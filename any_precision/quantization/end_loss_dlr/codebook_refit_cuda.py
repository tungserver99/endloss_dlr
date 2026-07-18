from __future__ import annotations

import torch

from .row_objective_cuda import (
    cluster_stats_from_labels,
    quantized_from_labels,
    require_cuda,
    solve_codebook,
)


def sort_codebook_and_remap(codebook: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    require_cuda(codebook, labels)
    sorted_codebook, permutation = torch.sort(codebook.float())
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(permutation.numel(), device=permutation.device)
    remapped = inverse[labels.long()]
    return sorted_codebook, remapped


def refit_codebook(
    weight: torch.Tensor,
    gradient: torch.Tensor,
    diagonal: torch.Tensor,
    lowrank: torch.Tensor,
    labels: torch.Tensor,
    num_levels: int,
    beta: float,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    require_cuda(weight, gradient, diagonal, lowrank, labels)
    stats = cluster_stats_from_labels(weight, gradient, diagonal, lowrank, labels, num_levels, beta)
    if torch.any(stats.A <= 0):
        raise RuntimeError("Encountered an empty cluster during refit; RBVT should preserve non-empty clusters")
    weight_sum_lowrank = lowrank.float().transpose(0, 1).matmul(weight.float()) if lowrank.numel() else lowrank.new_zeros((0,))
    codebook = solve_codebook(stats, weight_sum_lowrank, eps)
    codebook, labels = sort_codebook_and_remap(codebook, labels)
    quantized = quantized_from_labels(codebook, labels)
    return codebook, labels, quantized
