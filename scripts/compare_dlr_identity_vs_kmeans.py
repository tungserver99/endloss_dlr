#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import flash1dkmeans

from any_precision.analyzer import get_analyzer
from any_precision.quantization.end_loss_dlr.row_objective_cuda import quantize_group_dlr_batched
from env_utils import load_project_dotenv


def _resolve_module(analyzer, layer_idx: int, module_name: str | None):
    layer = analyzer.get_layers()[layer_idx]
    modules = analyzer.get_modules(layer)
    if module_name is None:
        first_name = next(iter(modules))
        return first_name, modules[first_name]
    if module_name not in modules:
        raise KeyError(f"Module {module_name!r} not found in layer {layer_idx}. Available: {list(modules)[:12]}")
    return module_name, modules[module_name]


def _kmeans_row(row: np.ndarray, num_bits: int, max_iter: int) -> np.ndarray:
    sorted_indices = np.argsort(row)
    sorted_x = row[sorted_indices].astype(np.float64)
    n = sorted_x.shape[0]
    if n == 0:
        return row.copy()

    weights_prefix_sum = np.arange(1, n + 1, dtype=np.float64)
    weighted_x_prefix_sum = np.cumsum(sorted_x)
    weighted_x_squared_prefix_sum = np.cumsum(sorted_x ** 2)

    n_clusters = 2 ** num_bits
    if n_clusters > 2:
        centroids, cluster_borders = flash1dkmeans.numba_kmeans_1d_k_cluster(
            sorted_X=sorted_x,
            n_clusters=n_clusters,
            max_iter=max_iter,
            weights_prefix_sum=weights_prefix_sum,
            weighted_X_prefix_sum=weighted_x_prefix_sum,
            weighted_X_squared_prefix_sum=weighted_x_squared_prefix_sum,
            start_idx=0,
            stop_idx=n,
        )
    else:
        centroids, cluster_borders = flash1dkmeans.numba_kmeans_1d_two_cluster(
            sorted_X=sorted_x,
            weights_prefix_sum=weights_prefix_sum,
            weighted_X_prefix_sum=weighted_x_prefix_sum,
            start_idx=0,
            stop_idx=n,
        )

    labels_sorted = np.empty(n, dtype=np.uint8)
    for idx in range(n_clusters):
        labels_sorted[cluster_borders[idx]:cluster_borders[idx + 1]] = idx
    labels = np.empty_like(labels_sorted)
    labels[sorted_indices] = labels_sorted
    q = centroids[labels.astype(np.int64)]
    return q.astype(np.float32)


def _dlr_identity_matrix(weight: torch.Tensor, num_bits: int, max_outer_iters: int, row_batch_size: int) -> torch.Tensor:
    device = torch.device("cuda")
    weight_cuda = weight.to(device=device, dtype=torch.float32)
    zeros = torch.zeros_like(weight_cuda)
    ones = torch.ones_like(weight_cuda)
    lowrank = weight_cuda.new_zeros((weight_cuda.shape[0], weight_cuda.shape[1], 0), dtype=torch.float32)

    quantized_rows = []
    for start in tqdm(range(0, weight_cuda.shape[0], row_batch_size), desc="DLR rows", unit="batch"):
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
            log_prefix=f"[test2 rows={start}-{end - 1}]",
        )
        quantized = torch.gather(codebooks, 1, labels.long())
        quantized_rows.append(quantized.cpu())
    return torch.cat(quantized_rows, dim=0)


def _kmeans_matrix(weight: torch.Tensor, num_bits: int, max_iter: int) -> torch.Tensor:
    row_outputs = []
    for row_idx in tqdm(range(weight.shape[0]), desc="KMeans rows", unit="row"):
        row_outputs.append(_kmeans_row(weight[row_idx].cpu().numpy().astype(np.float32), num_bits, max_iter))
    return torch.from_numpy(np.stack(row_outputs, axis=0))


def main():
    load_project_dotenv(verbose=True)

    parser = argparse.ArgumentParser(description="Compare DLR identity quantization against original flash1dkmeans")
    parser.add_argument("--model", default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--module", default="self_attn.q_proj")
    parser.add_argument("--bits", type=int, default=3)
    parser.add_argument("--max-outer-iters", type=int, default=50)
    parser.add_argument("--kmeans-iters", type=int, default=50)
    parser.add_argument("--row-batch-size", type=int, default=32)
    parser.add_argument("--limit-rows", type=int, default=None)
    parser.add_argument("--output-file", default="")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the DLR identity comparison test")

    analyzer = get_analyzer(args.model, include_tokenizer=False)
    module_name, module = _resolve_module(analyzer, args.layer, args.module)
    weight = module.weight.detach().float().cpu()
    if args.limit_rows is not None:
        weight = weight[: args.limit_rows]

    print(f"[test2] model={args.model}")
    print(f"[test2] layer={args.layer} module={module_name} shape={tuple(weight.shape)} bits={args.bits}")
    print(f"[test2] max_outer_iters={args.max_outer_iters} kmeans_iters={args.kmeans_iters}")

    q_dlr = _dlr_identity_matrix(weight, args.bits, args.max_outer_iters, args.row_batch_size)
    q_kmeans = _kmeans_matrix(weight, args.bits, args.kmeans_iters)

    mse_dlr = ((q_dlr.float() - weight.float()) ** 2).mean()
    mse_kmeans = ((q_kmeans.float() - weight.float()) ** 2).mean()
    ratio = mse_dlr / mse_kmeans

    summary = {
        "model": args.model,
        "layer": args.layer,
        "module": module_name,
        "shape": list(weight.shape),
        "bits": args.bits,
        "max_outer_iters": args.max_outer_iters,
        "kmeans_iters": args.kmeans_iters,
        "mse_dlr": float(mse_dlr.item()),
        "mse_kmeans": float(mse_kmeans.item()),
        "ratio": float(ratio.item()),
    }

    print(f"MSE_DLR={summary['mse_dlr']:.10f}")
    print(f"MSE_KMEANS={summary['mse_kmeans']:.10f}")
    print(f"RATIO={summary['ratio']:.10f}")

    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        print(f"[test2] saved {output_path}")


if __name__ == "__main__":
    main()
