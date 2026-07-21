#!/usr/bin/env python3
"""Method A constrained end-loss scalar quantization pipeline."""

from __future__ import annotations

import argparse
import gc
import logging
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

try:
    from env_utils import load_project_dotenv
except ModuleNotFoundError:
    def load_project_dotenv(*_args, **_kwargs):
        return None
from any_precision.analyzer import get_analyzer
from any_precision.quantization.method_a import (
    MethodAConfig,
    gather_quantized,
    nll_surrogate,
    quadratic_rows,
    quantize_rows_method_a,
)
from any_precision.quantization.method_a_curvature import (
    accumulate_method_a_curvatures,
    collect_method_a_sensitivities,
    curvature_group_path,
    row_group_ranges,
)
from any_precision.quantization.method_a_gradient import (
    collect_nll_gradients,
    model_identity,
    tensor_fingerprint,
)
from any_precision.quantization.method_a_sqllm_init import (
    build_sqllm_initialization,
    collect_sqllm_importance,
)
from any_precision.quantization.pack import pack


PIPELINE_STAGES = ("all", "tokens", "stats", "init", "quantize", "pack")


@dataclass
class QuantizationTotals:
    rows_total: int = 0
    rows_quantized: int = 0
    rows_q0: int = 0
    objective_sum: float = 0.0


def normalize_tokens(tokens, seq_len: int) -> torch.Tensor:
    if isinstance(tokens, torch.Tensor):
        if tokens.ndim == 2:
            return tokens.long()
        if tokens.ndim == 3 and tokens.shape[1] == 1:
            return tokens[:, 0, :].long()
        raise ValueError(f"Unexpected token tensor shape: {tuple(tokens.shape)}")
    normalized = []
    for item in tokens:
        item = item.detach().cpu()
        if item.ndim == 2 and item.shape[0] == 1:
            item = item[0]
        if item.ndim != 1 or item.numel() != seq_len:
            raise ValueError(f"Unexpected calibration item shape: {tuple(item.shape)}")
        normalized.append(item.long())
    if not normalized:
        raise ValueError("Calibration token cache is empty")
    return torch.stack(normalized)


def module_name_candidates(module_name: str) -> list[str]:
    candidates = [module_name]
    if module_name.startswith("self_attn."):
        candidates.append(module_name.replace("self_attn.", "linear_attn.", 1))
    elif module_name.startswith("linear_attn."):
        candidates.append(module_name.replace("linear_attn.", "self_attn.", 1))
    leaf = module_name.split(".")[-1]
    if leaf == "o_proj":
        candidates.append(module_name[:-len("o_proj")] + "out_proj")
    elif leaf == "out_proj":
        candidates.append(module_name[:-len("out_proj")] + "o_proj")
    return list(dict.fromkeys(candidates))


def resolve_entry(mapping, layer_idx: int, module_name: str, description: str):
    for candidate in module_name_candidates(module_name):
        if candidate in mapping:
            return mapping[candidate]
    leaves = {candidate.split(".")[-1] for candidate in module_name_candidates(module_name)}
    matches = [value for name, value in mapping.items() if name.split(".")[-1] in leaves]
    if len(matches) == 1:
        return matches[0]
    raise RuntimeError(
        f"Missing {description} at layer={layer_idx}, module={module_name}; "
        f"tried {module_name_candidates(module_name)}"
    )


def save_signed_gradients(
    analyzer,
    tokens: torch.Tensor,
    output_folder: str,
    batch_size: int,
    device: str,
    layer_chunk_size: int,
    overwrite: bool,
) -> None:
    root = Path(output_folder)
    config = {
        "schema": 1,
        "source": "mean_ground_truth_nll_signed_gradient",
        "num_examples": int(tokens.shape[0]),
        "seq_len": int(tokens.shape[1]),
        "batch_size": int(batch_size),
        "layer_chunk_size": int(layer_chunk_size),
        "model": model_identity(analyzer),
        "tokens_sha256": tensor_fingerprint(tokens),
    }
    expected = [root / f"l{i}.pt" for i in range(analyzer.num_layers)]
    config_path = root / "_config.pt"
    if not overwrite and expected and all(path.exists() for path in expected):
        if config_path.exists() and torch.load(config_path, map_location="cpu") == config:
            logging.info("Reusing signed NLL gradients at %s", root)
            return
    root.mkdir(parents=True, exist_ok=True)
    gradients = collect_nll_gradients(
        analyzer=analyzer,
        tokens=tokens,
        batch_size=batch_size,
        device=device,
        layer_chunk_size=layer_chunk_size,
    )
    for layer_idx in range(analyzer.num_layers):
        for module_name, gradient in gradients[layer_idx].items():
            absolute = gradient.abs()
            logging.info(
                "Method A gradient layer=%d module=%s "
                "max_abs_g=%.6e mean_abs_g=%.6e median_abs_g=%.6e",
                layer_idx, module_name, float(absolute.max()),
                float(absolute.mean()), float(absolute.median()),
            )
        torch.save(gradients[layer_idx], root / f"l{layer_idx}.pt")
    torch.save(config, config_path)
    del gradients
    gc.collect()


def quantize_method_a_cache(
    analyzer,
    gradient_folder: str,
    curvature_folder: str,
    initialization_folder: str,
    output_folder: str,
    bits: int,
    config: MethodAConfig,
    device: str,
    row_batch_size: int,
    num_output_groups: int,
    overwrite: bool,
    layer_range: tuple[int, int] | None = None,
) -> QuantizationTotals:
    gradient_config = torch.load(Path(gradient_folder) / "_config.pt", map_location="cpu")
    curvature_config = torch.load(Path(curvature_folder) / "_config.pt", map_location="cpu")
    initialization_config = torch.load(
        Path(initialization_folder) / "_config.pt", map_location="cpu"
    )
    quant_config = {
        "schema": 1,
        "source": "method_a_quantized",
        "bits": int(bits),
        "num_output_groups": int(num_output_groups),
        "solver": asdict(config),
        "gradient": gradient_config,
        "curvature": curvature_config,
        "initialization": initialization_config,
    }
    cached_groups = int(curvature_config["num_output_groups"])
    if cached_groups != int(num_output_groups):
        raise ValueError(
            f"Curvature cache has {cached_groups} output groups, requested {num_output_groups}"
        )
    output = Path(output_folder)
    output_config_path = output / "_config.pt"
    output_cache_matches = (
        not overwrite
        and output_config_path.exists()
        and torch.load(output_config_path, map_location="cpu") == quant_config
    )
    if layer_range is not None and output.exists() and not output_cache_matches and not overwrite:
        raise RuntimeError(
            "Partial quantization cannot reuse an output directory with a different config; "
            "use --overwrite-quantize or a new --quantized-path."
        )
    if overwrite and output.exists() and layer_range is None:
        shutil.rmtree(output)
    (output / "weights").mkdir(parents=True, exist_ok=True)
    (output / f"lut_{bits}").mkdir(parents=True, exist_ok=True)
    selected = range(analyzer.num_layers)
    if layer_range is not None:
        selected = range(layer_range[0], min(layer_range[1], analyzer.num_layers))

    totals = QuantizationTotals()
    for layer_idx in tqdm(selected, desc="Method A quantizing layers"):
        output_weights_path = output / "weights" / f"l{layer_idx}.pt"
        output_lut_path = output / f"lut_{bits}" / f"l{layer_idx}.pt"
        if output_cache_matches and output_weights_path.exists() and output_lut_path.exists():
            continue
        gradients = torch.load(Path(gradient_folder) / f"l{layer_idx}.pt", map_location="cpu")
        initial_labels = torch.load(
            Path(initialization_folder) / "weights" / f"l{layer_idx}.pt", map_location="cpu"
        )
        initial_luts = torch.load(
            Path(initialization_folder) / f"lut_{bits}" / f"l{layer_idx}.pt", map_location="cpu"
        )
        fp_weights = analyzer.get_layer_weights(layer_idx)
        output_labels, output_luts = {}, {}

        for module_name in gradients:
            fp_weight = resolve_entry(fp_weights, layer_idx, module_name, "FP weight")
            weight = fp_weight.to(device).float()
            gradient = resolve_entry(gradients, layer_idx, module_name, "signed NLL gradient").to(device).float()
            labels0 = torch.as_tensor(
                resolve_entry(initial_labels, layer_idx, module_name, "SqueezeLLM labels"), device=device
            ).long().reshape_as(weight)
            codebooks0 = torch.as_tensor(
                resolve_entry(initial_luts, layer_idx, module_name, "SqueezeLLM LUT"), device=device
            ).float().reshape(weight.shape[0], -1)
            labels = torch.empty_like(labels0, device="cpu", dtype=torch.uint8)
            codebooks = torch.empty_like(codebooks0, device="cpu", dtype=torch.float16)
            ranges = row_group_ranges(weight.shape[0], num_output_groups)

            for group_idx, (group_start, group_end) in enumerate(ranges):
                curvature = torch.load(
                    curvature_group_path(curvature_folder, layer_idx, module_name, group_idx),
                    map_location=device,
                )
                hessian_nll = curvature["H_nll"].float()
                hessian_kl = curvature["H_kl"].float()
                for start in range(group_start, group_end, max(1, row_batch_size)):
                    end = min(start + max(1, row_batch_size), group_end)
                    result = quantize_rows_method_a(
                        weight=weight[start:end],
                        gradient=gradient[start:end],
                        hessian_nll=hessian_nll,
                        hessian_kl=hessian_kl,
                        initial_labels=labels0[start:end],
                        initial_codebooks=codebooks0[start:end],
                        config=config,
                    )

                    # The packed representation uses FP16 LUTs. Constraints and
                    # objective must therefore be checked after that exact cast.
                    candidate_labels = result.labels
                    candidate_codebooks_fp16 = result.codebooks.to(torch.float16)
                    candidate_codebooks = candidate_codebooks_fp16.float()
                    candidate_q = gather_quantized(candidate_codebooks, candidate_labels)
                    candidate_error = candidate_q - weight[start:end]
                    candidate_cost_kl = 0.5 * quadratic_rows(candidate_error, hessian_kl)
                    candidate_cost_w = 0.5 * candidate_error.square().sum(dim=1)
                    candidate_loss = nll_surrogate(
                        weight[start:end],
                        gradient[start:end],
                        hessian_nll,
                        candidate_codebooks,
                        candidate_labels,
                    )

                    q0_labels = labels0[start:end]
                    q0_codebooks = codebooks0[start:end]
                    q0_q = gather_quantized(q0_codebooks, q0_labels)
                    q0_error = q0_q - weight[start:end]
                    eps_kl = 0.5 * quadratic_rows(q0_error, hessian_kl)
                    eps_w = 0.5 * q0_error.square().sum(dim=1)
                    q0_loss = nll_surrogate(
                        weight[start:end],
                        gradient[start:end],
                        hessian_nll,
                        q0_codebooks,
                        q0_labels,
                    )
                    kl_allowance = config.constraint_tol * eps_kl.abs().clamp_min(
                        config.numerical_eps
                    )
                    w_allowance = config.constraint_tol * eps_w.abs().clamp_min(
                        config.numerical_eps
                    )
                    finite = (
                        torch.isfinite(candidate_codebooks).all(dim=1)
                        & torch.isfinite(candidate_cost_kl)
                        & torch.isfinite(candidate_cost_w)
                        & torch.isfinite(candidate_loss)
                    )
                    feasible = (
                        finite
                        & (candidate_cost_kl <= eps_kl + kl_allowance)
                        & (candidate_cost_w <= eps_w + w_allowance)
                    )
                    use_candidate = feasible & (candidate_loss < q0_loss)
                    selected_labels = torch.where(
                        use_candidate[:, None], candidate_labels, q0_labels
                    )
                    selected_codebooks = torch.where(
                        use_candidate[:, None], candidate_codebooks_fp16, q0_codebooks.to(torch.float16)
                    )
                    selected_loss = torch.where(use_candidate, candidate_loss, q0_loss)
                    selected_cost_kl = torch.where(use_candidate, candidate_cost_kl, eps_kl)
                    selected_cost_w = torch.where(use_candidate, candidate_cost_w, eps_w)

                    labels[start:end] = selected_labels.detach().cpu().to(torch.uint8)
                    codebooks[start:end] = selected_codebooks.detach().cpu()
                    rows_in_batch = end - start
                    rows_q0 = int((~use_candidate).sum())
                    totals.rows_total += rows_in_batch
                    totals.rows_quantized += rows_in_batch
                    totals.rows_q0 += rows_q0
                    totals.objective_sum += float(selected_loss.sum())
                    logging.info(
                        "Method A layer=%d module=%s group=%d rows=[%d,%d) "
                        "candidate=%d q0=%d trace_H_nll=%.6e trace_H_kl=%.6e "
                        "max_abs_w=%.6e max_abs_q0=%.6e max_abs_e0=%.6e "
                        "max_eps_kl=%.6e max_eps_w=%.6e max_q0_nll=%.6e "
                        "max_KL_ratio=%.6e max_weight_ratio=%.6e",
                        layer_idx,
                        module_name,
                        group_idx,
                        start,
                        end,
                        int(use_candidate.sum()),
                        rows_q0,
                        float(hessian_nll.diagonal().sum()),
                        float(hessian_kl.diagonal().sum()),
                        float(weight[start:end].abs().max()),
                        float(q0_q.abs().max()),
                        float(q0_error.abs().max()),
                        float(eps_kl.max()),
                        float(eps_w.max()),
                        float(q0_loss.max()),
                        float(selected_cost_kl.div(eps_kl.clamp_min(config.numerical_eps)).max()),
                        float(selected_cost_w.div(eps_w.clamp_min(config.numerical_eps)).max()),
                    )
                del curvature, hessian_nll, hessian_kl
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            output_labels[module_name] = labels[:, None, :].numpy().astype(np.uint8)
            output_luts[module_name] = codebooks[:, None, :].numpy().astype(np.float16)
            del weight, gradient, labels0, codebooks0, labels, codebooks
            gc.collect()
        torch.save(output_labels, output_weights_path)
        torch.save(output_luts, output_lut_path)
    torch.save(quant_config, output_config_path)
    return totals


def parse_args():
    parser = argparse.ArgumentParser(description="Method A constrained end-loss scalar quantization")
    parser.add_argument("model", nargs="?", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--stage", choices=PIPELINE_STAGES, default="all")
    parser.add_argument("--bits", type=int, default=3)
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--dataset", default="redpajama")
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--num-examples", type=int, default=1024)
    parser.add_argument("--n-calib", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--stats-chunk-size", type=int, default=1024)
    parser.add_argument("--stats-layer-chunk-size", type=int, default=1)
    parser.add_argument("--sensitivity-layer-chunk-size", type=int, default=0)
    parser.add_argument("--num-output-groups", type=int, default=4)
    parser.add_argument("--kl-probes", type=int, default=1)
    parser.add_argument("--row-batch-size", type=int, default=64)
    parser.add_argument("--max-outer-iters", type=int, default=8)
    parser.add_argument("--max-inner-iters", type=int, default=8)
    parser.add_argument("--codebook-update-interval", type=int, default=1)
    parser.add_argument("--rel-tol", type=float, default=1e-7)
    parser.add_argument("--constraint-tol", type=float, default=1e-5)
    parser.add_argument("--numerical-eps", type=float, default=1e-12)
    parser.add_argument("--tie-tol", type=float, default=0.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cpu-count", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--redpajama-source", choices=["cache", "raw"], default="cache")
    parser.add_argument("--redpajama-dataset-repo", default=None)
    parser.add_argument("--tokens-path", default="")
    parser.add_argument("--stats-path", default="")
    parser.add_argument("--initialization-path", default="")
    parser.add_argument("--quantized-path", default="")
    parser.add_argument("--output-packed-path", default="")
    parser.add_argument("--layer-range", type=int, nargs=2, metavar=("START", "END"))
    parser.add_argument("--overwrite-tokens", action="store_true")
    parser.add_argument("--overwrite-stats", action="store_true")
    parser.add_argument("--overwrite-init", action="store_true")
    parser.add_argument("--overwrite-quantize", action="store_true")
    parser.add_argument("--overwrite-pack", action="store_true")
    return parser.parse_args()


def main():
    load_project_dotenv(Path(__file__).resolve().parent)
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s | %(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()
    if not 1 <= args.bits <= 8:
        raise ValueError("--bits must be in [1, 8] because packed labels are uint8")
    model_name = args.model.rstrip("/").split("/")[-1]
    data_tag = (
        f"{model_name}-{args.dataset}_s{args.num_examples}_blk{args.seq_len}"
        f"_seed{args.random_state}"
    )
    method_tag = (
        f"{data_tag}_n{args.n_calib}_g{args.num_output_groups}_klp{args.kl_probes}"
    )
    solver_tag = (
        f"o{args.max_outer_iters}i{args.max_inner_iters}"
        f"_rtol{args.rel_tol}_ctol{args.constraint_tol}"
        f"_cb{args.codebook_update_interval}"
        f"_eps{args.numerical_eps}_tie{args.tie_tol}"
    )
    run_tag = f"{model_name}-w{args.bits}-method-a-{method_tag}_{solver_tag}"
    args.tokens_path = args.tokens_path or f"{args.cache_dir}/tokens/{data_tag}.pt"
    args.stats_path = args.stats_path or f"{args.cache_dir}/method_a_stats/{method_tag}"
    args.initialization_path = (
        args.initialization_path
        or f"{args.cache_dir}/method_a_sqllm_init/{model_name}-w{args.bits}-{method_tag}"
    )
    args.quantized_path = args.quantized_path or f"{args.cache_dir}/method_a_quantized/{run_tag}"
    args.output_packed_path = args.output_packed_path or f"{args.cache_dir}/method_a_packed/anyprec-{run_tag}"

    logging.info("Loading model/analyzer: %s", args.model)
    analyzer = get_analyzer(args.model, include_tokenizer=True)
    if args.stage == "pack":
        analyzer.drop_original_weights()
        if args.overwrite_pack and Path(args.output_packed_path).exists():
            shutil.rmtree(args.output_packed_path)
        pack(
            analyzer=analyzer,
            lut_path=args.quantized_path,
            output_model_path=args.output_packed_path,
            seed_precision=args.bits,
            parent_precision=args.bits,
            cpu_count=args.cpu_count,
        )
        return

    if args.redpajama_dataset_repo:
        os.environ["REDPAJAMA_DATASET_REPO"] = args.redpajama_dataset_repo
    if args.overwrite_tokens and Path(args.tokens_path).exists():
        Path(args.tokens_path).unlink()
    
    from any_precision.quantization.datautils import get_tokens

    tokens = normalize_tokens(
        get_tokens(
            args.dataset,
            "train",
            analyzer.tokenizer,
            args.seq_len,
            args.num_examples,
            args.tokens_path,
            args.random_state,
            redpajama_source=args.redpajama_source,
        ),
        args.seq_len,
    )
    if args.stage == "tokens":
        return
    calib_tokens = tokens[:min(args.n_calib, tokens.shape[0])]
    gradient_tokens = calib_tokens
    stats_root = Path(args.stats_path)

    if args.stage in ("all", "stats"):
        save_signed_gradients(
            analyzer, gradient_tokens, str(stats_root / "gradients"), args.batch_size,
            args.device, args.stats_layer_chunk_size, args.overwrite_stats,
        )
        collect_method_a_sensitivities(
            analyzer, calib_tokens, str(stats_root / "sensitivities"), args.batch_size,
            args.device, args.num_output_groups, args.kl_probes,
            args.sensitivity_layer_chunk_size, args.random_state, args.overwrite_stats,
        )
        accumulate_method_a_curvatures(
            analyzer, calib_tokens, str(stats_root / "sensitivities"),
            str(stats_root / "curvatures"), args.device, args.stats_chunk_size,
            args.overwrite_stats,
        )
        if args.stage == "stats":
            return

    if args.stage in ("all", "init"):
        collect_sqllm_importance(
            analyzer, calib_tokens, str(stats_root / "sqllm_importance"),
            args.device, args.stats_layer_chunk_size, args.overwrite_init,
        )
        build_sqllm_initialization(
            analyzer, str(stats_root / "sqllm_importance"), args.initialization_path,
            args.bits, args.cpu_count, args.overwrite_init,
        )
        if args.stage == "init":
            return

    config = MethodAConfig(
        max_outer_iters=args.max_outer_iters,
        max_inner_iters=args.max_inner_iters,
        rel_tol=args.rel_tol,
        numerical_eps=args.numerical_eps,
        tie_tol=args.tie_tol,
        constraint_tol=args.constraint_tol,
        codebook_update_interval=args.codebook_update_interval,
    )
    totals = quantize_method_a_cache(
        analyzer, str(stats_root / "gradients"), str(stats_root / "curvatures"),
        args.initialization_path, args.quantized_path, args.bits, config, args.device,
        args.row_batch_size, args.num_output_groups, args.overwrite_quantize,
        tuple(args.layer_range) if args.layer_range else None,
    )
    logging.info(
        "Method A summary | rows=%d q0_rows=%d objective_sum=%.6e",
        totals.rows_quantized, totals.rows_q0, totals.objective_sum,
    )
    if args.stage == "quantize" or args.layer_range is not None:
        return

    analyzer.drop_original_weights()
    if args.overwrite_pack and Path(args.output_packed_path).exists():
        shutil.rmtree(args.output_packed_path)
    pack(
        analyzer=analyzer,
        lut_path=args.quantized_path,
        output_model_path=args.output_packed_path,
        seed_precision=args.bits,
        parent_precision=args.bits,
        cpu_count=args.cpu_count,
    )


if __name__ == "__main__":
    main()




