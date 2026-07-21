from __future__ import annotations

import logging
import os
from pathlib import Path

import numba
import torch

from .gradients import get_gradients
from .quantize import seed_and_upscale


def ensure_sqllm_initialization(
    analyzer,
    tokens: torch.Tensor,
    output_folder: str,
    bits: int,
    gradients_path: str,
    cpu_count: int | None = None,
    overwrite: bool = False,
) -> None:
    """Create q0 by running the repository's original SqueezeLLM path."""
    output = Path(output_folder)
    weights_dir = output / "weights"
    lut_dir = output / f"lut_{bits}"
    expected = [
        (weights_dir / f"l{layer_idx}.pt", lut_dir / f"l{layer_idx}.pt")
        for layer_idx in range(analyzer.num_layers)
    ]
    if not overwrite and expected and all(w.exists() and l.exists() for w, l in expected):
        logging.info("Reusing original SqueezeLLM q0 cache at %s", output)
        return

    cpu_count = int(cpu_count or os.cpu_count() or 1)
    numba.set_num_threads(max(1, cpu_count))
    logging.info("Building q0 with the repository's original SqueezeLLM implementation")
    gradients = get_gradients(
        analyzer=analyzer,
        input_tokens=tokens,
        save_path=gradients_path,
    )
    seed_and_upscale(
        analyzer=analyzer,
        gradients=gradients,
        output_folder=str(output),
        seed_precision=int(bits),
        parent_precision=int(bits),
        cpu_count=cpu_count,
    )
