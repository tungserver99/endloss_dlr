#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect AnyPrecision cache layer file names.")
    parser.add_argument("quantized_path")
    args = parser.parse_args()
    root = Path(args.quantized_path)
    weight_dir = root / "weights"
    lut_dirs = sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("lut_")]) if root.exists() else []
    pat = re.compile(r"^l\d+\.pt$")
    all_weights = sorted(weight_dir.glob("*.pt")) if weight_dir.exists() else []
    good_weights = [p for p in all_weights if pat.match(p.name)]
    bad_weights = [p for p in all_weights if not pat.match(p.name)]
    print(f"root={root}")
    print(f"weight_dir_exists={weight_dir.exists()} all_weight_pt={len(all_weights)} layer_weight_pt={len(good_weights)} non_layer_weight_pt={len(bad_weights)}")
    if good_weights:
        print(f"first_layer_files={[p.name for p in good_weights[:10]]}")
        print(f"last_layer_files={[p.name for p in good_weights[-10:]]}")
    if bad_weights:
        print(f"bad_weight_files_sample={[p.name for p in bad_weights[:30]]}")
    for lut_dir in lut_dirs:
        all_luts = sorted(lut_dir.glob("*.pt"))
        good_luts = [p for p in all_luts if pat.match(p.name)]
        bad_luts = [p for p in all_luts if not pat.match(p.name)]
        print(f"lut_dir={lut_dir.name} all_lut_pt={len(all_luts)} layer_lut_pt={len(good_luts)} non_layer_lut_pt={len(bad_luts)}")
        if bad_luts:
            print(f"bad_lut_files_sample={[p.name for p in bad_luts[:30]]}")


if __name__ == "__main__":
    main()