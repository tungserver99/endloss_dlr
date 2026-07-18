from __future__ import annotations

import torch

from .row_objective_cuda import require_cuda


def fit_diagonal_plus_lowrank(
    probe_matrix: torch.Tensor,
    rank: int,
    damping_ratio: float,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    require_cuda(probe_matrix)
    if probe_matrix.ndim != 2:
        raise ValueError(f"Expected probe_matrix [num_rows, in_features], got {tuple(probe_matrix.shape)}")

    probe_matrix = probe_matrix.float()
    diagonal_total = probe_matrix.square().sum(dim=0)
    if rank <= 0 or probe_matrix.shape[0] == 0:
        damping = damping_ratio * diagonal_total.mean().clamp_min(eps)
        return diagonal_total + damping, probe_matrix.new_zeros((probe_matrix.shape[1], 0), dtype=torch.float32)

    _, singular_values, vh = torch.linalg.svd(probe_matrix, full_matrices=False)
    kept = min(rank, singular_values.numel())
    if kept == 0:
        damping = damping_ratio * diagonal_total.mean().clamp_min(eps)
        return diagonal_total + damping, probe_matrix.new_zeros((probe_matrix.shape[1], 0), dtype=torch.float32)

    factors = vh[:kept].transpose(0, 1).contiguous() * singular_values[:kept].unsqueeze(0)
    residual_diag = (diagonal_total - factors.square().sum(dim=1)).clamp_min(0.0)
    damping = damping_ratio * diagonal_total.mean().clamp_min(eps)
    return residual_diag + damping, factors
