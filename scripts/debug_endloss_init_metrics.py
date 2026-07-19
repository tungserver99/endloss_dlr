#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from any_precision.analyzer import get_analyzer
from any_precision.quantization.endloss_dlr_batched import (
    _batched_loss,
    _continuous_target_batched,
    _exact_codebook_update_batched,
    _initialize_labels_batched,
    _initial_codebook_batched,
    _scatter_sum_batched,
    _sort_codebook_and_remap_batched,
)
from endloss_dlr_quantize import _resolve_layer_mapping_entry, _row_group_ranges
from env_utils import load_project_dotenv


def _fmt(value: float) -> str:
    return f"{value:.6e}"


def _quantile(values: torch.Tensor, q: float) -> float:
    if values.numel() == 0:
        return float("nan")
    return float(torch.quantile(values.float().flatten(), q).item())


def main() -> None:
    parser = argparse.ArgumentParser(description="Instrument EndLoss_DLR initialization metrics for one layer/module.")
    parser.add_argument("--model", default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--stats-path", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--module", required=True)
    parser.add_argument("--bits", type=int, default=3)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--row-batch-size", type=int, default=64)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--solver-d-min", type=float, default=1e-8)
    args = parser.parse_args()

    load_project_dotenv(Path(__file__).resolve().parent.parent)
    analyzer = get_analyzer(args.model, include_tokenizer=False)
    layer_stats = torch.load(Path(args.stats_path) / f"l{args.layer}.pt", map_location="cpu")
    if args.module not in layer_stats:
        raise KeyError(f"module {args.module!r} not found in layer stats keys={list(layer_stats)}")
    module_stats = layer_stats[args.module]
    fp_weights = analyzer.get_layer_weights(args.layer)
    W = _resolve_layer_mapping_entry(fp_weights, args.layer, args.module, "FP weights").to(args.device).float()
    g_all = module_stats["g"].to(args.device).float()
    K = 2 ** args.bits

    if "group_d" not in module_stats:
        raise RuntimeError("This diagnostic currently targets fast group_d/group_U stats.")
    group_d = module_stats["group_d"].to(args.device).float()
    group_U = module_stats["group_U"].to(args.device).float()
    row_ranges = _row_group_ranges(W.shape[0], int(module_stats.get("num_output_groups", group_d.shape[0])))

    print(f"model={args.model} stats_path={args.stats_path}")
    print(f"layer={args.layer} module={args.module} beta={args.beta:g} bits={args.bits} K={K}")
    print(f"W_shape={tuple(W.shape)} output_groups={len(row_ranges)} row_batch_size={args.row_batch_size}")

    rows_out = []
    module_max = {
        "w": 0.0,
        "g": 0.0,
        "x": 0.0,
        "x_minus_w": 0.0,
        "placeholder": 0.0,
        "exact": 0.0,
        "betaG_over_A": 0.0,
    }

    for group_idx, (group_start, group_end) in enumerate(row_ranges):
        d = group_d[group_idx].clamp_min(args.solver_d_min)
        U = group_U[group_idx]
        rho = d + U.square().sum(dim=-1)
        floor_thresh = max(args.solver_d_min * 1.01, float(d.median().item()) * 1e-3)
        frac_d_near_floor = float((d <= floor_thresh).float().mean().item())

        for start in range(group_start, group_end, args.row_batch_size):
            end = min(start + args.row_batch_size, group_end)
            w = W[start:end]
            g = g_all[start:end]
            alpha = torch.ones(end - start, device=args.device, dtype=torch.float32)
            x = _continuous_target_batched(w, g, d, U, alpha, args.beta)
            labels = _initialize_labels_batched(x, rho, K)
            placeholder = _initial_codebook_batched(x, labels, K)
            exact = _exact_codebook_update_batched(w, g, d, U, alpha, labels, placeholder, args.beta, K)
            exact_sorted, labels_sorted = _sort_codebook_and_remap_batched(exact, labels)
            loss = _batched_loss(w, g, d, U, alpha, exact_sorted, labels_sorted, args.beta)

            d_expand = d.unsqueeze(0).expand(end - start, -1).double()
            g64 = g.double()
            A = _scatter_sum_batched(d_expand, labels, K)
            G = _scatter_sum_batched(g64, labels, K)
            betaG_over_A = (abs(args.beta) * G.abs() / A.clamp_min(torch.finfo(A.dtype).tiny)).float()

            for local_idx in range(end - start):
                row_idx = start + local_idx
                row_metrics = {
                    "row": row_idx,
                    "group": group_idx,
                    "max_abs_w": float(w[local_idx].abs().max().item()),
                    "max_abs_g": float(g[local_idx].abs().max().item()),
                    "min_d": float(d.min().item()),
                    "median_d": float(d.median().item()),
                    "max_d": float(d.max().item()),
                    "frac_d_near_floor": frac_d_near_floor,
                    "max_abs_x": float(x[local_idx].abs().max().item()),
                    "max_abs_x_minus_w": float((x[local_idx] - w[local_idx]).abs().max().item()),
                    "max_abs_placeholder_codebook": float(placeholder[local_idx].abs().max().item()),
                    "max_abs_exact_codebook": float(exact[local_idx].abs().max().item()),
                    "min_A_k": float(A[local_idx][A[local_idx] > 0].min().item()),
                    "max_abs_G_k": float(G[local_idx].abs().max().item()),
                    "max_abs_betaG_over_A": float(betaG_over_A[local_idx].max().item()),
                    "loss": float(loss[local_idx].item()),
                }
                rows_out.append(row_metrics)
                module_max["w"] = max(module_max["w"], row_metrics["max_abs_w"])
                module_max["g"] = max(module_max["g"], row_metrics["max_abs_g"])
                module_max["x"] = max(module_max["x"], row_metrics["max_abs_x"])
                module_max["x_minus_w"] = max(module_max["x_minus_w"], row_metrics["max_abs_x_minus_w"])
                module_max["placeholder"] = max(module_max["placeholder"], row_metrics["max_abs_placeholder_codebook"])
                module_max["exact"] = max(module_max["exact"], row_metrics["max_abs_exact_codebook"])
                module_max["betaG_over_A"] = max(module_max["betaG_over_A"], row_metrics["max_abs_betaG_over_A"])

    rows_out.sort(key=lambda item: item["max_abs_exact_codebook"], reverse=True)
    exact_values = torch.tensor([item["max_abs_exact_codebook"] for item in rows_out])
    x_values = torch.tensor([item["max_abs_x"] for item in rows_out])
    ratio_values = torch.tensor([item["max_abs_betaG_over_A"] for item in rows_out])

    print("SUMMARY " + " ".join([
        f"max_abs_w={_fmt(module_max['w'])}",
        f"max_abs_g={_fmt(module_max['g'])}",
        f"max_abs_x={_fmt(module_max['x'])}",
        f"max_abs_x_minus_w={_fmt(module_max['x_minus_w'])}",
        f"max_abs_placeholder={_fmt(module_max['placeholder'])}",
        f"max_abs_exact={_fmt(module_max['exact'])}",
        f"max_abs_betaG_over_A={_fmt(module_max['betaG_over_A'])}",
        f"p50_exact={_fmt(_quantile(exact_values, 0.50))}",
        f"p99_exact={_fmt(_quantile(exact_values, 0.99))}",
        f"p99_x={_fmt(_quantile(x_values, 0.99))}",
        f"p99_betaG_over_A={_fmt(_quantile(ratio_values, 0.99))}",
    ]))

    print(f"TOP_ROWS_BY_EXACT_CODEBOOK topk={args.topk}")
    fields = [
        "row", "group", "max_abs_w", "max_abs_g", "min_d", "median_d", "max_d", "frac_d_near_floor",
        "max_abs_x", "max_abs_x_minus_w", "max_abs_placeholder_codebook", "max_abs_exact_codebook",
        "min_A_k", "max_abs_G_k", "max_abs_betaG_over_A", "loss",
    ]
    print("\t".join(fields))
    for item in rows_out[: args.topk]:
        print("\t".join(str(item[field]) if field in {"row", "group"} else _fmt(float(item[field])) for field in fields))


if __name__ == "__main__":
    main()