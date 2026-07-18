from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class EndLossDLRConfig:
    bits: int = 3
    beta: float = 0.0
    rank: int = 4
    oversample: int = 4
    subspace_iterations: int = 1
    num_output_groups: int = 8
    damping_ratio: float = 1e-4
    eps: float = 1e-12
    device: str = "cuda"
    row_batch_size: int = 128
    calibration_batch_size: int = 1
    fisher_probes: int = 16
    gradient_num_examples: int | None = None
    stats_layer_chunk_size: int = 8
    max_outer_iters: int = 8
    rel_tol: float = 1e-7
    lambda_safety: float = 1.01
    tie_tol: float = 0.0
    stats_dtype: str = "float32"
    small_matrix_dtype: str = "float32"
    cache_dir: str = "./cache"
    dataset: str = "redpajama"
    seq_len: int = 4096
    num_examples: int = 128
    identity_curvature: bool = False

    @property
    def num_levels(self) -> int:
        return 1 << self.bits

    def validate(self):
        if not self.device.startswith("cuda"):
            raise ValueError(f"EndLossDLRConfig.device must be CUDA, got {self.device!r}")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for EndLossDLRConfig")
        if not (0.0 <= self.beta <= 1.0):
            raise ValueError(f"beta must be in [0, 1], got {self.beta}")
        if self.rank < 0:
            raise ValueError(f"rank must be >= 0, got {self.rank}")
        if self.bits < 1:
            raise ValueError(f"bits must be >= 1, got {self.bits}")
        if self.row_batch_size < 1:
            raise ValueError(f"row_batch_size must be >= 1, got {self.row_batch_size}")
        if self.max_outer_iters < 1:
            raise ValueError(f"max_outer_iters must be >= 1, got {self.max_outer_iters}")
        if self.lambda_safety < 1.0:
            raise ValueError(f"lambda_safety must be >= 1.0, got {self.lambda_safety}")
        if self.gradient_num_examples is not None and self.gradient_num_examples < 1:
            raise ValueError(f"gradient_num_examples must be >= 1, got {self.gradient_num_examples}")
        if self.stats_layer_chunk_size < 1:
            raise ValueError(f"stats_layer_chunk_size must be >= 1, got {self.stats_layer_chunk_size}")

