from __future__ import annotations

from dataclasses import dataclass

import torch

from .codebook_refit_cuda import sort_codebook_and_remap
from .row_objective_cuda import (
    ClusterStats,
    build_prefix_sums,
    interval_stats,
    partition_score,
    quantized_from_labels,
    require_cuda,
    solve_codebook,
)


@dataclass
class GreedySplitResult:
    codebook: torch.Tensor
    labels: torch.Tensor
    quantized: torch.Tensor
    boundaries: list[int]
    score: torch.Tensor


def _stats_from_boundaries(prefix: dict[str, torch.Tensor], boundaries: list[int]) -> ClusterStats:
    As = []
    bs = []
    ms = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        A, b, m = interval_stats(prefix, start, end)
        As.append(A)
        bs.append(b)
        ms.append(m)
    A_tensor = torch.stack(As)
    b_tensor = torch.stack(bs)
    if ms and ms[0].numel():
        m_tensor = torch.stack(ms)
    else:
        device = prefix["d"].device
        m_tensor = torch.zeros((len(As), 0), device=device, dtype=torch.float32)
    return ClusterStats(A=A_tensor, b=b_tensor, m=m_tensor)


def _make_sorted_labels(boundaries: list[int], device: torch.device) -> torch.Tensor:
    labels = torch.empty(boundaries[-1], device=device, dtype=torch.long)
    for idx, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        labels[start:end] = idx
    return labels


def greedy_split_row(
    sorted_weight: torch.Tensor,
    sorted_gradient: torch.Tensor,
    sorted_diagonal: torch.Tensor,
    sorted_lowrank: torch.Tensor,
    beta: float,
    num_levels: int,
    min_cluster_size: int,
    eps: float,
    candidate_chunk: int,
) -> GreedySplitResult:
    require_cuda(sorted_weight, sorted_gradient, sorted_diagonal, sorted_lowrank)
    prefix = build_prefix_sums(sorted_weight, sorted_gradient, sorted_diagonal, sorted_lowrank, beta)
    boundaries = [0, int(sorted_weight.numel())]
    weight_sum_lowrank = (
        sorted_lowrank.float().transpose(0, 1).matmul(sorted_weight.float()) if sorted_lowrank.numel() else sorted_lowrank.new_zeros((0,))
    )

    while len(boundaries) - 1 < num_levels:
        best_boundaries = None
        best_score = None
        candidate_specs: list[tuple[int, int]] = []
        for parent_idx, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            if end - start >= 2 * min_cluster_size:
                for split in range(start + min_cluster_size, end - min_cluster_size + 1):
                    candidate_specs.append((parent_idx, split))

        if not candidate_specs:
            raise RuntimeError("No valid split candidates remain; try reducing min_cluster_size or bit-width")

        for chunk_start in range(0, len(candidate_specs), candidate_chunk):
            chunk = candidate_specs[chunk_start:chunk_start + candidate_chunk]
            for parent_idx, split in chunk:
                local_boundaries = boundaries[: parent_idx + 1] + [split] + boundaries[parent_idx + 1 :]
                stats = _stats_from_boundaries(prefix, local_boundaries)
                score = partition_score(stats, weight_sum_lowrank, eps)
                if best_score is None or score > best_score:
                    best_score = score
                    best_boundaries = local_boundaries

        boundaries = best_boundaries

    final_stats = _stats_from_boundaries(prefix, boundaries)
    codebook = solve_codebook(final_stats, weight_sum_lowrank, eps)
    labels = _make_sorted_labels(boundaries, sorted_weight.device)
    codebook, labels = sort_codebook_and_remap(codebook, labels)
    quantized = quantized_from_labels(codebook, labels)
    score = partition_score(final_stats, weight_sum_lowrank, eps)
    return GreedySplitResult(codebook=codebook, labels=labels, quantized=quantized, boundaries=boundaries, score=score)
