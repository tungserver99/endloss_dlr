from __future__ import annotations

from collections import defaultdict
import gc
import logging
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .activations import get_inps, update_outs_parallel
from .method_a_gradient import (
    _disable_checkpointing_for_stats,
    _enable_checkpointing_for_stats,
    _iter_layer_chunks,
    _restore_float_dtypes,
    _snapshot_float_dtypes,
    model_identity,
    tensor_fingerprint,
)


def row_group_ranges(num_rows: int, num_groups: int) -> list[tuple[int, int]]:
    groups = max(1, min(int(num_groups), int(num_rows)))
    base, remainder = divmod(int(num_rows), groups)
    ranges, start = [], 0
    for group_idx in range(groups):
        end = start + base + (1 if group_idx < remainder else 0)
        ranges.append((start, end))
        start = end
    return ranges


def module_cache_name(module_name: str) -> str:
    return module_name.replace("/", "__slash__")


def sensitivity_path(root: str | Path, layer_idx: int, module_name: str) -> Path:
    return Path(root) / f"l{layer_idx}" / f"{module_cache_name(module_name)}.pt"


def curvature_group_path(root: str | Path, layer_idx: int, module_name: str, group_idx: int) -> Path:
    return Path(root) / f"l{layer_idx}" / module_cache_name(module_name) / f"g{group_idx}.pt"


def _prediction_slice(tensor: torch.Tensor) -> torch.Tensor:
    return tensor[:, :-1, :] if tensor.ndim == 3 and tensor.shape[1] > 1 else tensor


def _group_sensitivity(gradient: torch.Tensor, num_groups: int) -> torch.Tensor:
    gradient = _prediction_slice(gradient.detach()).float()
    ranges = row_group_ranges(gradient.shape[-1], num_groups)
    return torch.stack(
        [gradient[..., start:end].square().mean(dim=-1) for start, end in ranges], dim=-1
    )


def ground_truth_nll_sum(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="sum"
    )


def teacher_score_sum(
    logits: torch.Tensor, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    log_probs = F.log_softmax(logits, dim=-1)
    pseudo = torch.multinomial(
        log_probs.detach().exp().reshape(-1, log_probs.shape[-1]),
        num_samples=1,
        generator=generator,
    ).reshape(log_probs.shape[:-1])
    return log_probs.gather(-1, pseudo[..., None]).sum(), pseudo


def _progress_kwargs() -> dict:
    return {
        "ascii": True, "leave": False, "dynamic_ncols": False, "ncols": 100,
        "mininterval": 5.0, "maxinterval": 30.0, "file": sys.stdout,
    }


def collect_method_a_sensitivities(
    analyzer,
    tokens: torch.Tensor,
    output_folder: str,
    batch_size: int,
    device: str,
    num_output_groups: int,
    kl_probes: int,
    layer_chunk_size: int = 1,
    random_state: int = 0,
    overwrite: bool = False,
) -> None:
    """Collect grouped squared d(end-loss)/d(linear output) signals.

    NLL uses summed ground-truth next-token cross entropy. KL uses independent
    pseudo-labels sampled from the frozen teacher and backpropagates summed log
    probabilities. No mean reduction is squared in either collector.
    """
    root = Path(output_folder)
    config = {
        "schema": 1,
        "source": "guidedquant_grouped_output_sensitivity",
        "num_output_groups": int(num_output_groups),
        "kl_probes": int(kl_probes),
        "num_examples": int(tokens.shape[0]),
        "seq_len": int(tokens.shape[1]),
        "random_state": int(random_state),
        "model": model_identity(analyzer),
        "tokens_sha256": tensor_fingerprint(tokens),
    }
    config_path = root / "_config.pt"
    expected = [
        sensitivity_path(root, layer_idx, name)
        for layer_idx, layer in enumerate(analyzer.get_layers())
        for name in analyzer.get_modules(layer)
    ]
    if not overwrite and expected and all(path.exists() for path in expected):
        if config_path.exists() and torch.load(config_path, map_location="cpu") == config:
            logging.info("Reusing Method A sensitivity cache at %s", root)
            return
    root.mkdir(parents=True, exist_ok=True)

    model = analyzer.model
    model.to(device).eval()
    original_param_dtypes, original_buffer_dtypes = _snapshot_float_dtypes(model)
    model.bfloat16()
    _enable_checkpointing_for_stats(model)
    original_requires_grad = {id(param): param.requires_grad for param in model.parameters()}
    for param in model.parameters():
        param.requires_grad_(False)

    layers = analyzer.get_layers()
    generator = torch.Generator(device=device)
    generator.manual_seed(int(random_state))
    kl_probes = max(1, int(kl_probes))

    for chunk_start, layer_chunk in _iter_layer_chunks(layers, max(1, layer_chunk_size)):
        modules = {}
        for local_idx, layer in enumerate(layer_chunk):
            layer_idx = chunk_start + local_idx
            for module_name, module in analyzer.get_modules(layer).items():
                modules[(layer_idx, module_name)] = module
        captured: dict[tuple[int, str], torch.Tensor] = {}
        hooks = []

        def make_hook(key):
            def forward_hook(_module, _inputs, output):
                if not isinstance(output, torch.Tensor) or not output.requires_grad:
                    return
                output.register_hook(
                    lambda grad: captured.__setitem__(key, _group_sensitivity(grad, num_output_groups).cpu())
                )
            return forward_hook

        for key, module in modules.items():
            hooks.append(module.register_forward_hook(make_hook(key)))

        nll_chunks = defaultdict(list)
        kl_chunks = defaultdict(list)
        for start in tqdm(
            range(0, tokens.shape[0], batch_size),
            desc=f"Method A sensitivities L{chunk_start}-{chunk_start + len(layer_chunk) - 1}",
            **_progress_kwargs(),
        ):
            batch = tokens[start:start + batch_size].to(device)
            captured.clear()
            logits = model(input_ids=batch, use_cache=False).logits[:, :-1, :].float()
            labels = batch[:, 1:]
            nll_sum = ground_truth_nll_sum(logits, labels)
            nll_sum.backward()
            missing = set(modules) - set(captured)
            if missing:
                raise RuntimeError(f"Missing NLL output gradients for {sorted(missing)}")
            for key, value in captured.items():
                nll_chunks[key].append(value.float())
            model.zero_grad(set_to_none=True)
            del logits, labels, nll_sum

            probe_sums = {}
            for _probe in range(kl_probes):
                captured.clear()
                logits = model(input_ids=batch, use_cache=False).logits[:, :-1, :].float()
                score_sum, pseudo = teacher_score_sum(logits, generator)
                score_sum.backward()
                missing = set(modules) - set(captured)
                if missing:
                    raise RuntimeError(f"Missing KL-score output gradients for {sorted(missing)}")
                for key, value in captured.items():
                    if key not in probe_sums:
                        probe_sums[key] = value.float()
                    else:
                        probe_sums[key].add_(value.float())
                model.zero_grad(set_to_none=True)
                del logits, pseudo, score_sum
            for key, value in probe_sums.items():
                kl_chunks[key].append((value / kl_probes).float())
            del batch, probe_sums
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        for hook in hooks:
            hook.remove()
        for key, module in modules.items():
            layer_idx, module_name = key
            path = sensitivity_path(root, layer_idx, module_name)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "nll": torch.cat(nll_chunks[key], dim=0),
                    "kl": torch.cat(kl_chunks[key], dim=0),
                    "row_ranges": row_group_ranges(module.weight.shape[0], num_output_groups),
                },
                path,
            )
        del nll_chunks, kl_chunks, captured
        gc.collect()

    for param in model.parameters():
        param.requires_grad_(original_requires_grad[id(param)])
    _restore_float_dtypes(model, original_param_dtypes, original_buffer_dtypes)
    _disable_checkpointing_for_stats(model)
    model.cpu().eval()
    torch.save(config, config_path)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@torch.no_grad()
def accumulate_method_a_curvatures(
    analyzer,
    tokens: torch.Tensor,
    sensitivity_folder: str,
    output_folder: str,
    device: str = "cuda",
    stats_chunk_size: int = 1024,
    overwrite: bool = False,
) -> None:
    """Build dense grouped X^T Diag(s) X matrices one group at a time."""
    root = Path(output_folder)
    sensitivity_config = torch.load(Path(sensitivity_folder) / "_config.pt", map_location="cpu")
    config = {**sensitivity_config, "schema": 1, "source": "guidedquant_dense_grouped"}
    config_path = root / "_config.pt"
    layers = analyzer.get_layers()
    expected = [
        curvature_group_path(root, layer_idx, name, group_idx)
        for layer_idx, layer in enumerate(layers)
        for name, module in analyzer.get_modules(layer).items()
        for group_idx in range(len(row_group_ranges(module.weight.shape[0], sensitivity_config["num_output_groups"])))
    ]
    cache_matches = (
        not overwrite
        and config_path.exists()
        and torch.load(config_path, map_location="cpu") == config
    )
    if cache_matches and expected and all(path.exists() for path in expected):
        logging.info("Reusing Method A curvature cache at %s", root)
        return
    root.mkdir(parents=True, exist_ok=True)

    devices = [torch.device(device)]
    data = [row.unsqueeze(0) for row in tokens.cpu()]
    inps, forward_args = get_inps(
        analyzer=analyzer,
        data=data,
        model_seqlen=tokens.shape[1],
        devices=devices,
        offload_activations=True,
    )
    outs = [torch.zeros_like(inps[0])]

    for layer_idx, layer in enumerate(layers):
        layer.to(device).eval()
        modules = analyzer.get_modules(layer)
        for module_name, module in modules.items():
            sensitivity = torch.load(
                sensitivity_path(sensitivity_folder, layer_idx, module_name), map_location="cpu"
            )
            for group_idx, _row_range in enumerate(sensitivity["row_ranges"]):
                out_path = curvature_group_path(root, layer_idx, module_name, group_idx)
                if out_path.exists() and cache_matches:
                    continue
                hessians = {}
                for source in ("nll", "kl"):
                    weighted = sensitivity[source][..., group_idx]
                    hessian = torch.zeros(
                        module.weight.shape[1], module.weight.shape[1], dtype=torch.float32, device=device
                    )
                    sample_index = 0

                    def input_hook(_module, inputs, _output):
                        nonlocal sample_index
                        x = inputs[0] if isinstance(inputs, tuple) else inputs
                        batch_samples = x.shape[0] if x.ndim == 3 else 1
                        x = _prediction_slice(x.detach()).reshape(-1, x.shape[-1]).float()
                        s = weighted[sample_index:sample_index + batch_samples].reshape(-1).to(device).float()
                        sample_index += batch_samples
                        for start in range(0, x.shape[0], max(1, stats_chunk_size)):
                            x_chunk = x[start:start + stats_chunk_size]
                            s_chunk = s[start:start + stats_chunk_size]
                            hessian.add_(x_chunk.T.matmul(x_chunk * s_chunk[:, None]))

                    hook = module.register_forward_hook(input_hook)
                    for sample in inps[0]:
                        layer(sample.to(device).unsqueeze(0), **forward_args)
                    hook.remove()
                    if sample_index != weighted.shape[0]:
                        raise RuntimeError(
                            f"Sensitivity/input sample mismatch at layer={layer_idx}, "
                            f"module={module_name}, group={group_idx}, source={source}: "
                            f"used {sample_index}, expected {weighted.shape[0]}"
                        )
                    total_tokens = max(1, weighted.numel())
                    hessian.div_(total_tokens)
                    hessian = 0.5 * (hessian + hessian.T)
                    hessians[f"H_{source}"] = hessian.cpu()
                    del hessian
                out_path.parent.mkdir(parents=True, exist_ok=True)
                logging.info(
                    "Method A curvature layer=%d module=%s group=%d "
                    "trace_H_nll=%.6e trace_H_kl=%.6e "
                    "diag_nll=[%.6e,%.6e,%.6e] diag_kl=[%.6e,%.6e,%.6e] "
                    "row_abs_nll=[%.6e,%.6e] row_abs_kl=[%.6e,%.6e]",
                    layer_idx,
                    module_name,
                    group_idx,
                    float(hessians["H_nll"].diagonal().sum()),
                    float(hessians["H_kl"].diagonal().sum()),
                    float(hessians["H_nll"].diagonal().min()),
                    float(hessians["H_nll"].diagonal().median()),
                    float(hessians["H_nll"].diagonal().max()),
                    float(hessians["H_kl"].diagonal().min()),
                    float(hessians["H_kl"].diagonal().median()),
                    float(hessians["H_kl"].diagonal().max()),
                    float(hessians["H_nll"].abs().sum(dim=1).min()),
                    float(hessians["H_nll"].abs().sum(dim=1).max()),
                    float(hessians["H_kl"].abs().sum(dim=1).min()),
                    float(hessians["H_kl"].abs().sum(dim=1).max()),
                )
                torch.save(hessians, out_path)
                del hessians
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            del sensitivity

        update_outs_parallel(
            devices=devices,
            layer=layer,
            inps=inps,
            outs=outs,
            compute_mse=False,
            is_after_quant=False,
            **forward_args,
        )
        layer.cpu()
        inps, outs = outs, inps
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    torch.save(config, config_path)




