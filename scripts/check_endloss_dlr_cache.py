#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def _layer_idx(path: Path) -> int:
    return int(path.stem[1:])


def main() -> None:
    parser = argparse.ArgumentParser(description="Check EndLoss_DLR AnyPrecision cache for NaN/Inf and extreme LUT values.")
    parser.add_argument("quantized_path")
    parser.add_argument("--warn-abs", type=float, default=100.0)
    parser.add_argument("--topk", type=int, default=20)
    args = parser.parse_args()

    root = Path(args.quantized_path)
    weight_dir = root / "weights"
    lut_dirs = sorted(
        [path for path in root.iterdir() if path.is_dir() and path.name.startswith("lut_")],
        key=lambda p: int(p.name.split("_", 1)[1]),
    )
    if not weight_dir.exists() or not lut_dirs:
        raise FileNotFoundError(f"Not a valid quantized cache: {root}")
    lut_dir = lut_dirs[-1]

    worst = []
    total_luts = 0
    bad_luts = 0
    bad_indices = 0
    max_abs_all = 0.0

    for weight_file in sorted(weight_dir.glob("l*.pt"), key=_layer_idx):
        layer_idx = _layer_idx(weight_file)
        weights = torch.load(weight_file, map_location="cpu")
        luts = torch.load(lut_dir / weight_file.name, map_location="cpu")
        for module_name, idx in weights.items():
            lut = torch.as_tensor(luts[module_name]).float()
            idx_t = torch.as_tensor(idx)
            finite = torch.isfinite(lut)
            total_luts += lut.numel()
            bad_luts += int((~finite).sum().item())
            max_abs = float(lut[finite].abs().max().item()) if finite.any() else float("nan")
            max_abs_all = max(max_abs_all, 0.0 if max_abs != max_abs else max_abs)
            idx_min = int(idx_t.min().item())
            idx_max = int(idx_t.max().item())
            k = lut.shape[-1]
            idx_bad = idx_min < 0 or idx_max >= k
            bad_indices += int(idx_bad)
            if (not finite.all()) or max_abs > args.warn_abs or idx_bad:
                worst.append((max_abs, layer_idx, module_name, int((~finite).sum().item()), idx_min, idx_max, k))

    worst.sort(key=lambda item: (-1 if item[0] != item[0] else -item[0], item[1], item[2]))
    print(f"cache={root}")
    print(f"lut_values={total_luts} nonfinite_lut_values={bad_luts} bad_index_modules={bad_indices} max_abs_lut={max_abs_all:.6e}")
    if not worst:
        print(f"OK: all LUTs finite, all indices in range, max_abs_lut <= {args.warn_abs:g}")
        return

    print(f"Suspicious modules (top {args.topk}, warn_abs={args.warn_abs:g}):")
    for max_abs, layer_idx, module_name, nonfinite, idx_min, idx_max, k in worst[: args.topk]:
        print(
            f"layer={layer_idx:02d} module={module_name} max_abs_lut={max_abs:.6e} "
            f"nonfinite={nonfinite} idx_range=[{idx_min},{idx_max}] K={k}"
        )


if __name__ == "__main__":
    main()