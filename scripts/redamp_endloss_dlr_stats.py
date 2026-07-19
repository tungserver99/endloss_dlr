#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch


def _layer_idx(path: Path) -> int:
    return int(path.stem[1:])


def _redamp_group_d(group_d: torch.Tensor, group_U: torch.Tensor, old_ratio: float, new_ratio: float) -> torch.Tensor:
    d = group_d.float()
    U = group_U.float()
    lowrank_diag = U.square().sum(dim=-1)
    # Original fast stats used: d = residual + old_ratio * mean(diag_total),
    # with diag_total = residual + diag(UU^T). Recover the per-group mean algebraically.
    diag_mean = (d.mean(dim=1, keepdim=True) + lowrank_diag.mean(dim=1, keepdim=True)) / (1.0 + float(old_ratio))
    residual = (d - float(old_ratio) * diag_mean).clamp_min(0.0)
    return residual + float(new_ratio) * diag_mean.clamp_min(1e-12)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a redamped EndLoss_DLR stats cache without recomputing gradients/Fisher.")
    parser.add_argument("src")
    parser.add_argument("dst")
    parser.add_argument("--old-damping-ratio", type=float, default=1e-4)
    parser.add_argument("--new-damping-ratio", type=float, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    if not src.exists():
        raise FileNotFoundError(src)
    if dst.exists():
        if not args.overwrite:
            raise FileExistsError(f"Destination exists: {dst}. Use --overwrite to replace.")
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)

    stats_config = {
        "stats_method": "fast_weight_gradient_fisher_v2",
        "rank": 4,
        "oversampling": 4,
        "n_calib": 1024,
        "batch_size": 1,
        "device": "cuda",
        "fisher_probes": 16,
        "gradient_num_examples": None,
        "stats_layer_chunk_size": 8,
        "num_output_groups": 8,
        "damping_ratio": float(args.new_damping_ratio),
    }
    torch.save(stats_config, dst / "_config.pt")

    changed_modules = 0
    max_scale = 0.0
    for layer_file in sorted(src.glob("l*.pt"), key=_layer_idx):
        layer = torch.load(layer_file, map_location="cpu")
        out = {}
        for module_name, stats in layer.items():
            stats = dict(stats)
            if "group_d" in stats and "group_U" in stats:
                old_d = stats["group_d"].float()
                new_d = _redamp_group_d(old_d, stats["group_U"], args.old_damping_ratio, args.new_damping_ratio)
                scale = float((new_d / old_d.clamp_min(1e-30)).max().item())
                max_scale = max(max_scale, scale)
                stats["group_d"] = new_d.cpu()
                stats["damping_ratio"] = float(args.new_damping_ratio)
                stats["old_damping_ratio"] = float(args.old_damping_ratio)
                changed_modules += 1
            out[module_name] = stats
        torch.save(out, dst / layer_file.name)

    print(f"src={src}")
    print(f"dst={dst}")
    print(f"new_damping_ratio={args.new_damping_ratio:g} changed_modules={changed_modules} max_d_scale={max_scale:.6e}")


if __name__ == "__main__":
    main()