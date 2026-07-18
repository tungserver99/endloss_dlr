from __future__ import annotations

import json
import os
from pathlib import Path

import torch


def quantized_cache_path(cache_dir: str, model_name: str, bits: int, dataset: str, num_examples: int, seq_len: int) -> str:
    return os.path.join(cache_dir, "endloss_dlr_quantized", f"{model_name}-w{bits}-{dataset}_s{num_examples}_blk{seq_len}")


def packed_model_output_path(cache_dir: str, model_name: str, bits: int, dataset: str, num_examples: int, seq_len: int) -> str:
    return os.path.join(cache_dir, "packed", f"anyprec-{model_name}-endloss-dlr-w{bits}-{dataset}_s{num_examples}_blk{seq_len}")


def save_layer_artifacts(output_dir: str, layer_idx: int, module_codebooks: dict[str, torch.Tensor], module_labels: dict[str, torch.Tensor]):
    num_levels = next(iter(module_codebooks.values())).shape[-1]
    bit_width = int(round(torch.log2(torch.tensor(num_levels, dtype=torch.float32)).item()))
    lut_dir = Path(output_dir) / f"lut_{bit_width}"
    weight_dir = Path(output_dir) / "weights"
    lut_dir.mkdir(parents=True, exist_ok=True)
    weight_dir.mkdir(parents=True, exist_ok=True)
    torch.save({name: tensor.float().cpu() for name, tensor in module_codebooks.items()}, lut_dir / f"l{layer_idx}.pt")
    torch.save({name: tensor.to(torch.uint8).cpu() for name, tensor in module_labels.items()}, weight_dir / f"l{layer_idx}.pt")


def save_metadata(output_dir: str, metadata: dict):
    path = Path(output_dir) / "metadata.json"
    serializable = {}
    for key, value in metadata.items():
        if isinstance(value, dict):
            serializable[key] = value
        elif isinstance(value, (int, float, str, bool)) or value is None:
            serializable[key] = value
        else:
            serializable[key] = str(value)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(serializable, handle, indent=2)
