from __future__ import annotations

import gc
import hashlib
import logging
from pathlib import Path
import sys

import torch
from tqdm.auto import tqdm

from .method_a_curvature import (
    _group_sensitivity,
    ground_truth_nll_sum,
    module_cache_name,
    row_group_ranges,
    teacher_score_sum,
)
from .method_a_gradient import (
    _disable_checkpointing_for_stats,
    _enable_checkpointing_for_stats,
    _restore_float_dtypes,
    _snapshot_float_dtypes,
    model_identity,
    tensor_fingerprint,
)


def lowrank_curvature_path(
    root: str | Path, layer_idx: int, module_name: str, source: str
) -> Path:
    return (
        Path(root)
        / f"l{layer_idx}"
        / module_cache_name(module_name)
        / f"{source}_lowrank.pt"
    )


def lowrank_source_completion_path(root: str | Path, source: str) -> Path:
    return Path(root) / f"_{source}_complete.pt"


def deterministic_sketch_seed(
    base_seed: int, model: str, layer_idx: int, module_name: str, group_idx: int
) -> int:
    identity = f"{base_seed}|{model}|{layer_idx}|{module_name}|{group_idx}"
    digest = hashlib.sha256(identity.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**63 - 1)


def make_rademacher_sketch(
    num_groups: int,
    in_features: int,
    rank: int,
    seeds: list[int],
    device: torch.device | str,
) -> torch.Tensor:
    if int(num_groups) < 1 or int(in_features) < 1 or int(rank) < 1:
        raise ValueError("num_groups, in_features, and rank must be positive")
    if len(seeds) != int(num_groups):
        raise ValueError("seeds must contain one entry per curvature group")
    actual_rank = min(int(rank), int(in_features))
    sketches = []
    for group_idx in range(num_groups):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seeds[group_idx]))
        signs = torch.randint(
            0, 2, (in_features, actual_rank), generator=generator, dtype=torch.int8
        )
        sketches.append(signs.float().mul_(2).sub_(1).div_(actual_rank**0.5))
    return torch.stack(sketches).to(device=device)


class StreamingDiagLowRankSketch:
    """Streaming sketch of grouped X.T @ diag(s) @ X without forming XTX."""

    def __init__(
        self,
        in_features: int,
        num_groups: int,
        rank: int,
        seeds: list[int],
        device: torch.device | str,
        token_chunk_size: int = 256,
    ) -> None:
        self.in_features = int(in_features)
        self.num_groups = int(num_groups)
        if self.in_features < 1 or self.num_groups < 1 or int(rank) < 1:
            raise ValueError("in_features, num_groups, and rank must be positive")
        self.rank = min(int(rank), self.in_features)
        self.device = torch.device(device)
        if int(token_chunk_size) < 1:
            raise ValueError("token_chunk_size must be positive")
        self.token_chunk_size = int(token_chunk_size)
        self.omega = make_rademacher_sketch(
            self.num_groups,
            self.in_features,
            self.rank,
            seeds,
            self.device,
        )
        self.c = torch.zeros(
            self.num_groups, self.in_features, self.rank,
            dtype=torch.float32, device=self.device,
        )
        self.b = torch.zeros(
            self.num_groups, self.rank, self.rank,
            dtype=torch.float32, device=self.device,
        )
        self.diag_h = torch.zeros(
            self.num_groups, self.in_features,
            dtype=torch.float32, device=self.device,
        )
        self.update_calls = 0
        self.observed_tokens = 0

    @torch.no_grad()
    def update(self, inputs: torch.Tensor, sensitivity: torch.Tensor) -> None:
        x = inputs.detach().reshape(-1, self.in_features)
        s = sensitivity.detach().reshape(-1, self.num_groups)
        if x.shape[0] != s.shape[0]:
            raise ValueError(
                f"Input/sensitivity token mismatch: {x.shape[0]} != {s.shape[0]}"
            )
        for start in range(0, x.shape[0], self.token_chunk_size):
            end = min(start + self.token_chunk_size, x.shape[0])
            x32 = x[start:end].float()
            s32 = s[start:end].float().clamp_min(0)
            z = x32[:, None, :] * s32.sqrt()[:, :, None]
            z_grouped = z.permute(1, 0, 2)
            p_grouped = torch.bmm(z_grouped, self.omega)
            self.c.add_(torch.bmm(z_grouped.transpose(1, 2), p_grouped))
            self.b.add_(torch.bmm(p_grouped.transpose(1, 2), p_grouped))
            self.diag_h.add_(z.square().sum(dim=0))
        self.update_calls += 1
        self.observed_tokens += int(x.shape[0])

    @torch.no_grad()
    def finalize(self, denominator: int) -> dict[str, torch.Tensor | int | float]:
        scale = 1.0 / max(1, int(denominator))
        c = self.c * scale
        b = self.b * scale
        diag_h = self.diag_h * scale
        factors = []
        effective_ranks = []
        thresholds = []
        for group_idx in range(self.num_groups):
            symmetric_b = 0.5 * (b[group_idx] + b[group_idx].T)
            evals, evecs = torch.linalg.eigh(symmetric_b)
            max_abs_eval = evals.abs().max()
            threshold = (
                torch.finfo(symmetric_b.dtype).eps
                * symmetric_b.shape[0]
                * max_abs_eval
            )
            active = evals > threshold
            if bool(active.any()):
                factor = (
                    c[group_idx]
                    @ evecs[:, active]
                    @ torch.diag(evals[active].rsqrt())
                )
            else:
                factor = c.new_zeros((self.in_features, 0))
            factors.append(factor)
            effective_ranks.append(int(active.sum()))
            thresholds.append(float(threshold))

        max_rank = max(effective_ranks, default=0)
        padded_factors = c.new_zeros(
            (self.num_groups, self.in_features, max_rank)
        )
        for group_idx, factor in enumerate(factors):
            padded_factors[group_idx, :, : factor.shape[1]] = factor
        diagonal = (
            diag_h - padded_factors.square().sum(dim=2)
        ).clamp_min_(0)
        column_l1 = padded_factors.abs().sum(dim=1)
        row_majorizer = diagonal + (
            padded_factors.abs() * column_l1[:, None, :]
        ).sum(dim=2)
        return {
            "diagonal": diagonal.cpu(),
            "factor": padded_factors.cpu(),
            "row_majorizer": row_majorizer.cpu(),
            "effective_ranks": effective_ranks,
            "eigenvalue_thresholds": thresholds,
            "target_diagonal": diag_h.cpu(),
        }


def _progress_kwargs() -> dict:
    return {
        "ascii": True,
        "leave": False,
        "dynamic_ncols": False,
        "ncols": 100,
        "mininterval": 5.0,
        "maxinterval": 30.0,
        "file": sys.stdout,
    }


def _expected_paths(analyzer, root: Path, source: str) -> list[Path]:
    return [
        lowrank_curvature_path(root, layer_idx, module_name, source)
        for layer_idx, layer in enumerate(analyzer.get_layers())
        for module_name in analyzer.get_modules(layer)
    ]


def _collect_source(
    analyzer,
    tokens: torch.Tensor,
    root: Path,
    source: str,
    batch_size: int,
    device: str,
    num_output_groups: int,
    rank: int,
    sketch_seed: int,
    token_chunk_size: int,
    kl_probes: int,
    random_state: int,
) -> None:
    model = analyzer.model
    layers = analyzer.get_layers()
    identity = model_identity(analyzer)
    accumulators = {}
    hooks = []

    for layer_idx, layer in enumerate(layers):
        for module_name, module in analyzer.get_modules(layer).items():
            group_count = len(row_group_ranges(module.weight.shape[0], num_output_groups))
            seeds = [
                deterministic_sketch_seed(
                    sketch_seed, identity, layer_idx, module_name, group_idx
                )
                for group_idx in range(group_count)
            ]
            key = (layer_idx, module_name)
            accumulators[key] = StreamingDiagLowRankSketch(
                module.weight.shape[1], group_count, rank, seeds, device,
                token_chunk_size,
            )

            def make_hook(accumulator):
                def forward_hook(_module, inputs, output):
                    if not inputs or not isinstance(inputs[0], torch.Tensor):
                        return
                    if not isinstance(output, torch.Tensor) or not output.requires_grad:
                        return
                    module_inputs = inputs[0].detach()

                    def output_gradient_hook(gradient):
                        sensitivity = _group_sensitivity(
                            gradient, accumulator.num_groups
                        )
                        accumulator.update(module_inputs, sensitivity)

                    output.register_hook(output_gradient_hook)
                return forward_hook

            hooks.append(module.register_forward_hook(make_hook(accumulators[key])))

    generator = torch.Generator(device=device)
    generator.manual_seed(int(random_state))
    probes = 1 if source == "nll" else max(1, int(kl_probes))
    total_valid_tokens = 0
    backward_calls = 0
    try:
        for start in tqdm(
            range(0, tokens.shape[0], max(1, batch_size)),
            desc=f"Method A direct {source.upper()} sketch (all layers)",
            **_progress_kwargs(),
        ):
            batch = tokens[start:start + max(1, batch_size)].to(device)
            valid_tokens = int(batch.shape[0]) * max(1, int(batch.shape[1]) - 1)
            total_valid_tokens += valid_tokens
            for _probe in range(probes):
                logits = model(input_ids=batch, use_cache=False).logits[:, :-1, :].float()
                if source == "nll":
                    objective = ground_truth_nll_sum(logits, batch[:, 1:])
                else:
                    objective, pseudo = teacher_score_sum(logits, generator)
                objective.backward()
                backward_calls += 1
                if backward_calls == 1:
                    missing = [
                        f"layer={layer_idx} module={module_name}"
                        for (layer_idx, module_name), accumulator in accumulators.items()
                        if accumulator.update_calls != 1
                    ]
                    if missing:
                        raise RuntimeError(
                            f"Missing first {source} sketch updates for "
                            f"{missing[:8]}. Check gradient-checkpointing hook semantics."
                        )
                model.zero_grad(set_to_none=True)
                if source == "kl":
                    del pseudo
                del logits, objective
            del batch
    finally:
        for hook in hooks:
            hook.remove()

    denominator = total_valid_tokens * probes
    expected_observed_tokens = int(tokens.numel()) * probes
    for (layer_idx, module_name), accumulator in accumulators.items():
        if accumulator.update_calls != backward_calls:
            raise RuntimeError(
                f"Expected one {source} sketch update per backward at "
                f"layer={layer_idx}, module={module_name}; got "
                f"{accumulator.update_calls}, expected {backward_calls}. "
                "Check gradient-checkpointing hook semantics."
            )
        if accumulator.observed_tokens != expected_observed_tokens:
            raise RuntimeError(
                f"Observed-token mismatch for {source} at layer={layer_idx}, "
                f"module={module_name}: got {accumulator.observed_tokens}, "
                f"expected {expected_observed_tokens}"
            )
        result = accumulator.finalize(denominator)
        module = analyzer.get_modules(layers[layer_idx])[module_name]
        result.update({
            "source": source,
            "model": identity,
            "layer_idx": int(layer_idx),
            "module_name": module_name,
            "weight_shape": tuple(module.weight.shape),
            "row_ranges": row_group_ranges(
                module.weight.shape[0], num_output_groups
            ),
            "normalization_denominator": int(denominator),
            "sketch_rank": int(accumulator.rank),
            "group_seeds": [
                deterministic_sketch_seed(
                    sketch_seed, identity, layer_idx, module_name, group_idx
                )
                for group_idx in range(accumulator.num_groups)
            ],
        })
        path = lowrank_curvature_path(root, layer_idx, module_name, source)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(result, path)
        diagonal = result["diagonal"]
        factor = result["factor"]
        logging.info(
            "Method A direct %s layer=%d module=%s ranks=%s trace=[%.6e,%.6e]",
            source.upper(), layer_idx, module_name, result["effective_ranks"],
            float((diagonal + factor.square().sum(2)).sum(1).min()),
            float((diagonal + factor.square().sum(2)).sum(1).max()),
        )
    del accumulators
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def collect_method_a_diag_lowrank_curvatures(
    analyzer,
    tokens: torch.Tensor,
    output_folder: str,
    batch_size: int,
    device: str,
    num_output_groups: int,
    rank: int,
    sketch_seed: int = 0,
    token_chunk_size: int = 256,
    kl_probes: int = 1,
    random_state: int = 0,
    calibration_dataset: str = "unknown",
    overwrite: bool = False,
) -> None:
    """Collect direct streaming D+UU.T NLL and KL curvature estimates."""
    if int(num_output_groups) < 1:
        raise ValueError("num_output_groups must be positive")
    if int(rank) < 1:
        raise ValueError("rank must be positive")
    if int(token_chunk_size) < 1:
        raise ValueError("token_chunk_size must be positive")
    if int(batch_size) < 1:
        raise ValueError("batch_size must be positive")
    if int(kl_probes) < 1:
        raise ValueError("kl_probes must be positive")
    root = Path(output_folder)
    forward_dtype = next(
        (
            str(param.dtype).replace("torch.", "")
            for param in analyzer.model.parameters()
            if torch.is_floating_point(param)
        ),
        "unknown",
    )
    config = {
        "schema": 1,
        "source": "method_a_direct_diag_lowrank_curvature",
        "model": model_identity(analyzer),
        "tokens_sha256": tensor_fingerprint(tokens),
        "num_examples": int(tokens.shape[0]),
        "seq_len": int(tokens.shape[1]),
        "calibration_dataset": calibration_dataset,
        "total_valid_tokens": (
            int(tokens.shape[0]) * max(1, int(tokens.shape[1]) - 1)
        ),
        "token_mask_policy": "all next-token positions; no padding mask",
        "num_output_groups": int(num_output_groups),
        "sketch_rank": int(rank),
        "sketch_seed": int(sketch_seed),
        "model_forward_dtype": forward_dtype,
        "accumulator_dtype": "float32",
        "factor_storage_dtype": "float32",
        "sketch_distribution": "rademacher_pm_1_over_sqrt_rank",
        "kl_probes": int(kl_probes),
        "random_state": int(random_state),
        "nll_loss": "ground_truth_next_token_cross_entropy_sum",
        "kl_loss": "teacher_pseudo_label_log_probability_sum",
        "loss_reduction": "sum",
        "normalization": "valid_prediction_tokens; KL additionally averages probes",
    }
    config_path = root / "_config.pt"
    cache_matches = (
        not overwrite
        and config_path.exists()
        and torch.load(config_path, map_location="cpu") == config
    )
    nll_expected = _expected_paths(analyzer, root, "nll")
    kl_expected = _expected_paths(analyzer, root, "kl")
    def source_complete(source, expected):
        marker = lowrank_source_completion_path(root, source)
        return (
            cache_matches
            and marker.exists()
            and torch.load(marker, map_location="cpu") == config
            and all(path.exists() for path in expected)
        )

    nll_complete = source_complete("nll", nll_expected)
    kl_complete = source_complete("kl", kl_expected)
    if nll_complete and kl_complete:
        logging.info("Reusing Method A direct diag-lowrank cache at %s", root)
        return
    root.mkdir(parents=True, exist_ok=True)
    # Saving before collection lets a completed NLL pass survive an
    # interruption during KL; completeness still requires every source file.
    torch.save(config, config_path)

    model = analyzer.model
    model.to(device).eval()
    original_param_dtypes, original_buffer_dtypes = _snapshot_float_dtypes(model)
    original_requires_grad = {id(param): param.requires_grad for param in model.parameters()}
    _enable_checkpointing_for_stats(model)
    for param in model.parameters():
        param.requires_grad_(False)
    try:
        if not nll_complete:
            lowrank_source_completion_path(root, "nll").unlink(missing_ok=True)
            _collect_source(
                analyzer, tokens, root, "nll", batch_size, device,
                num_output_groups, rank, sketch_seed, token_chunk_size,
                kl_probes, random_state,
            )
            torch.save(config, lowrank_source_completion_path(root, "nll"))
        if not kl_complete:
            lowrank_source_completion_path(root, "kl").unlink(missing_ok=True)
            _collect_source(
                analyzer, tokens, root, "kl", batch_size, device,
                num_output_groups, rank, sketch_seed, token_chunk_size,
                kl_probes, random_state,
            )
            torch.save(config, lowrank_source_completion_path(root, "kl"))
        torch.save(config, config_path)
    finally:
        for param in model.parameters():
            param.requires_grad_(original_requires_grad[id(param)])
        _restore_float_dtypes(model, original_param_dtypes, original_buffer_dtypes)
        _disable_checkpointing_for_stats(model)
        model.cpu().eval()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
