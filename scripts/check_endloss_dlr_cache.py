#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from env_utils import load_project_dotenv
from endloss_dlr_quantize import _resolve_layer_mapping_entry


def _layer_idx(path: Path) -> int:
    return int(path.stem[1:])


def main() -> None:
    parser = argparse.ArgumentParser(description="Check EndLoss_DLR AnyPrecision cache for NaN/Inf and extreme LUT values.")
    parser.add_argument("quantized_path")
    parser.add_argument("--warn-abs", type=float, default=100.0)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--model", default="", help="Optional HF model path; when set, also reports original FP weight scale for outlier rows.")
    parser.add_argument("--device", default="cpu")
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

    analyzer = None
    fp_weights_cache = {}
    if args.model:
        from any_precision.analyzer import get_analyzer
        load_project_dotenv(ROOT_DIR)
        analyzer = get_analyzer(args.model, include_tokenizer=False)

    worst = []
    total_luts = 0
    bad_luts = 0
    bad_indices = 0
    max_abs_all = 0.0
    materialized_nonfinite = 0
    materialized_max = -1.0
    materialized_argmax = None
    layer_file_pattern = re.compile(r"^l\d+\.pt$")
    for weight_file in sorted((p for p in weight_dir.glob("l*.pt") if layer_file_pattern.match(p.name)), key=_layer_idx):
        layer_idx = _layer_idx(weight_file)
        weights = torch.load(weight_file, map_location="cpu")
        luts = torch.load(lut_dir / weight_file.name, map_location="cpu")
        for module_name, idx in weights.items():
            lut = torch.as_tensor(luts[module_name]).float()
            idx_t = torch.as_tensor(idx)
            finite = torch.isfinite(lut)
            total_luts += lut.numel()
            bad_luts += int((~finite).sum().item())
            abs_lut = lut.abs()
            max_abs = float(abs_lut[finite].max().item()) if finite.any() else float("nan")
            max_abs_all = max(max_abs_all, 0.0 if max_abs != max_abs else max_abs)
            idx_min = int(idx_t.min().item())
            idx_max = int(idx_t.max().item())
            k = lut.shape[-1]
            idx_bad = idx_min < 0 or idx_max >= k
            bad_indices += int(idx_bad)

            flat_pos = int(abs_lut.reshape(-1).argmax().item()) if lut.numel() else 0
            row = flat_pos // k
            codeword = flat_pos % k
            lut_value = float(lut.reshape(-1)[flat_pos].item()) if lut.numel() else float("nan")
            row_indices = idx_t[row].reshape(-1).long() if idx_t.numel() else torch.empty(0, dtype=torch.long)
            usage_count = int((row_indices == int(codeword)).sum().item())
            materialized_row = lut[row].index_select(0, row_indices.clamp(0, k - 1)) if row_indices.numel() else torch.empty(0)
            max_abs_materialized_row = float(materialized_row.abs().max().item()) if materialized_row.numel() else float("nan")
            row_col = int(materialized_row.abs().argmax().item()) if materialized_row.numel() else -1
            materialized_value = float(materialized_row[row_col].item()) if row_col >= 0 else float("nan")
            max_abs_original_row = float("nan")
            max_abs_original_module = float("nan")
            if analyzer is not None:
                if layer_idx not in fp_weights_cache:
                    fp_weights_cache[layer_idx] = analyzer.get_layer_weights(layer_idx)
                fp_weight = _resolve_layer_mapping_entry(fp_weights_cache[layer_idx], layer_idx, module_name, "FP weights").float()
                max_abs_original_module = float(fp_weight.abs().max().item())
                max_abs_original_row = float(fp_weight[row].abs().max().item())

            safe_idx = idx_t.long().clamp(0, k - 1)
            if idx_t.ndim != 3 or lut.ndim != 3:
                materialized = torch.empty(0)
            else:
                row_ids = torch.arange(lut.shape[0]).view(-1, 1, 1)
                group_ids = torch.arange(lut.shape[1]).view(1, -1, 1)
                materialized = lut[row_ids, group_ids, safe_idx].float()
            if materialized.numel():
                mat_finite = torch.isfinite(materialized)
                materialized_nonfinite += int((~mat_finite).sum().item())
                if mat_finite.any():
                    mat_abs = materialized.abs()
                    local_flat = int(mat_abs.reshape(-1).argmax().item())
                    local_max = float(mat_abs.reshape(-1)[local_flat].item())
                    if local_max > materialized_max:
                        row_count, group_count, col_count = materialized.shape
                        mat_row = local_flat // (group_count * col_count)
                        rem = local_flat % (group_count * col_count)
                        mat_group = rem // col_count
                        mat_col = rem % col_count
                        materialized_max = local_max
                        materialized_argmax = (layer_idx, module_name, mat_row, mat_group, mat_col, float(materialized.reshape(-1)[local_flat].item()))
            if (not finite.all()) or max_abs > args.warn_abs or idx_bad:
                worst.append((
                    max_abs, layer_idx, module_name, row, codeword, lut_value, usage_count,
                    max_abs_materialized_row, row_col, materialized_value,
                    max_abs_original_row, max_abs_original_module, int((~finite).sum().item()), idx_min, idx_max, k,
                ))

    worst.sort(key=lambda item: (-1 if item[0] != item[0] else -item[0], item[1], item[2]))
    print(f"cache={root}")
    print(f"lut_values={total_luts} nonfinite_lut_values={bad_luts} bad_index_modules={bad_indices} max_abs_lut={max_abs_all:.6e}")
    if materialized_argmax is None:
        print(f"materialized_nonfinite_weights={materialized_nonfinite} max_abs_materialized_weight=nan")
    else:
        layer_idx, module_name, row, group, col, value = materialized_argmax
        print(
            "MATERIALIZED_MAX "
            f"nonfinite={materialized_nonfinite} max_abs={materialized_max:.6e} "
            f"layer={layer_idx:02d} module={module_name} row={row} group={group} col={col} value={value:.6e}"
        )
    if not worst:
        print(f"OK: all LUTs finite, all indices in range, max_abs_lut <= {args.warn_abs:g}")
        return

    print(f"Suspicious LUT outliers (top {args.topk}, warn_abs={args.warn_abs:g}):")
    print("layer\tmodule\trow\tcodeword\tlut_value\tusage_count\tmax_abs_lut\tmax_abs_materialized_row\tmaterialized_argmax_col\tmaterialized_value\tmax_abs_original_row\tmax_abs_original_module\tnonfinite\tidx_range\tK")
    for item in worst[: args.topk]:
        (
            max_abs, layer_idx, module_name, row, codeword, lut_value, usage_count,
            max_abs_materialized_row, row_col, materialized_value,
            max_abs_original_row, max_abs_original_module, nonfinite, idx_min, idx_max, k,
        ) = item
        print(
            f"{layer_idx:02d}\t{module_name}\t{row}\t{codeword}\t{lut_value:.6e}\t{usage_count}\t{max_abs:.6e}\t"
            f"{max_abs_materialized_row:.6e}\t{row_col}\t{materialized_value:.6e}\t"
            f"{max_abs_original_row:.6e}\t{max_abs_original_module:.6e}\t{nonfinite}\t[{idx_min},{idx_max}]\t{k}"
        )


if __name__ == "__main__":
    main()