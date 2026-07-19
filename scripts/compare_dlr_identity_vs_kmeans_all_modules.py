#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numba
import numpy as np
import torch
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from any_precision.analyzer import get_analyzer
from any_precision.quantization.end_loss_dlr.row_objective_cuda import quantize_group_dlr_batched
from any_precision.quantization.quantize import _seed_and_upscale_layer
from env_utils import load_project_dotenv


def _dequantize_codes_and_lut(codes, lut) -> torch.Tensor:
    idx = torch.as_tensor(codes, dtype=torch.long)
    levels = torch.as_tensor(lut, dtype=torch.float32)
    rows, groups, width = idx.shape
    row_ids = torch.arange(rows).view(-1, 1, 1)
    group_ids = torch.arange(groups).view(1, -1, 1)
    return levels[row_ids, group_ids, idx].reshape(rows, groups * width)


def _dlr_identity_matrix(weight: torch.Tensor, num_bits: int, max_outer_iters: int, row_batch_size: int) -> torch.Tensor:
    device = torch.device("cuda")
    weight_cuda = weight.to(device=device, dtype=torch.float32)
    zeros = torch.zeros_like(weight_cuda)
    ones = torch.ones_like(weight_cuda)
    lowrank = weight_cuda.new_zeros((weight_cuda.shape[0], weight_cuda.shape[1], 0), dtype=torch.float32)

    quantized_rows = []
    for start in range(0, weight_cuda.shape[0], row_batch_size):
        end = min(start + row_batch_size, weight_cuda.shape[0])
        codebooks, labels, _ = quantize_group_dlr_batched(
            w=weight_cuda[start:end],
            g=zeros[start:end],
            d=ones[start:end],
            U=lowrank[start:end],
            K=2 ** num_bits,
            beta=0.0,
            max_outer_iters=max_outer_iters,
            rel_tol=1e-7,
            lambda_safety=1.01,
            tie_tol=0.0,
            eps=1e-12,
            log_prefix=f"[all-modules rows={start}-{end - 1}]",
        )
        quantized = torch.gather(codebooks, 1, labels.long())
        quantized_rows.append(quantized.cpu())
    return torch.cat(quantized_rows, dim=0)


def _kmeans_matrix(weight: torch.Tensor, num_bits: int, cpu_count: int | None) -> torch.Tensor:
    if cpu_count is not None:
        numba.set_num_threads(max(1, cpu_count))
    layer_gradients = [np.ones_like(weight.numpy(), dtype=np.float32)]
    layer_modules = [weight.numpy().astype(np.float32)]
    luts_by_bit_by_module, parent_weights = _seed_and_upscale_layer(
        layer_gradients,
        layer_modules,
        num_bits,
        num_bits,
        1,
        random_state=None,
    )
    lut = luts_by_bit_by_module[0][0]
    codes = parent_weights[0]
    return _dequantize_codes_and_lut(codes, lut)


def _full_module_name(analyzer, layer_idx: int, canonical_name: str) -> str:
    actual_path = analyzer.get_layer_module_paths(layer_idx)[canonical_name]
    return f"{analyzer.model_name}.{analyzer.layers_name}.{layer_idx}.{actual_path}"


def main():
    load_project_dotenv(verbose=True)

    parser = argparse.ArgumentParser(description="Compare DLR identity vs flash1dkmeans over all quantized modules")
    parser.add_argument("--model", default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--bits", type=int, default=3)
    parser.add_argument("--max-outer-iters", type=int, default=50)
    parser.add_argument("--row-batch-size", type=int, default=32)
    parser.add_argument("--cpu-count", type=int, default=None)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--output-file", default="")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the full-module DLR vs K-means comparison")

    analyzer = get_analyzer(args.model, include_tokenizer=False)
    results = []

    total_modules = sum(len(analyzer.get_layer_module_paths(layer_idx)) for layer_idx in range(analyzer.num_layers))
    progress = tqdm(total=total_modules, desc="Comparing modules", unit="module")

    for layer_idx in range(analyzer.num_layers):
        layer = analyzer.get_layers()[layer_idx]
        modules = analyzer.get_modules(layer)
        for canonical_name, module in modules.items():
            weight = module.weight.detach().float().cpu()
            q_dlr = _dlr_identity_matrix(weight, args.bits, args.max_outer_iters, args.row_batch_size)
            q_kmeans = _kmeans_matrix(weight, args.bits, args.cpu_count)

            mse_dlr = ((q_dlr.float() - weight.float()) ** 2).mean().item()
            mse_kmeans = ((q_kmeans.float() - weight.float()) ** 2).mean().item()
            ratio = mse_dlr / mse_kmeans if mse_kmeans != 0 else float("inf")

            result = {
                "layer": layer_idx,
                "canonical_name": canonical_name,
                "module_name": _full_module_name(analyzer, layer_idx, canonical_name),
                "shape": list(weight.shape),
                "mse_dlr": mse_dlr,
                "mse_kmeans": mse_kmeans,
                "ratio": ratio,
            }
            results.append(result)
            print(f"{result['module_name']}\tMSE_DLR={mse_dlr:.10f}\tMSE_KMEANS={mse_kmeans:.10f}\tRATIO={ratio:.6f}")
            progress.update(1)

    progress.close()

    ratios = [item["ratio"] for item in results]
    topk = sorted(results, key=lambda x: x["ratio"], reverse=True)[: args.topk]
    summary = {
        "model": args.model,
        "bits": args.bits,
        "max_outer_iters": args.max_outer_iters,
        "module_count": len(results),
        "median_ratio": float(statistics.median(ratios)),
        "mean_ratio": float(statistics.fmean(ratios)),
        "max_ratio": float(max(ratios)),
        "topk": topk,
        "results": results,
    }

    print("\n=== SUMMARY ===")
    print(f"module_count={summary['module_count']}")
    print(f"median_ratio={summary['median_ratio']:.6f}")
    print(f"mean_ratio={summary['mean_ratio']:.6f}")
    print(f"max_ratio={summary['max_ratio']:.6f}")
    print(f"\n=== TOP {args.topk} MODULES BY RATIO ===")
    for item in topk:
        print(
            f"{item['module_name']}\tRATIO={item['ratio']:.6f}\t"
            f"MSE_DLR={item['mse_dlr']:.10f}\tMSE_KMEANS={item['mse_kmeans']:.10f}"
        )

    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
