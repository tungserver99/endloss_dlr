from __future__ import annotations

import torch

from .row_objective_cuda import quantized_from_labels, require_cuda


def rbvt_coordinate_descent(
    weight: torch.Tensor,
    gradient: torch.Tensor,
    diagonal: torch.Tensor,
    lowrank: torch.Tensor,
    codebook: torch.Tensor,
    labels: torch.Tensor,
    beta: float,
    eps: float,
    preserve_nonempty_clusters: bool = True,
    reverse: bool = False,
) -> torch.Tensor:
    require_cuda(weight, gradient, diagonal, lowrank, codebook, labels)
    labels = labels.long().clone()
    quantized = quantized_from_labels(codebook.float(), labels)
    error = quantized - weight.float()
    h = lowrank.float().transpose(0, 1).matmul(error) if lowrank.numel() else lowrank.new_zeros((0,))
    counts = torch.bincount(labels, minlength=codebook.numel()).to(weight.device)
    norm_u_sq = lowrank.float().square().sum(dim=1) if lowrank.numel() else weight.new_zeros(weight.shape[0], dtype=torch.float32)
    order = range(weight.shape[0] - 1, -1, -1) if reverse else range(weight.shape[0])

    for i in order:
        current = int(labels[i].item())
        current_q = quantized[i]
        delta = codebook.float() - current_q
        local_linear = beta * gradient[i].float() + diagonal[i].float() * error[i] + (
            lowrank[i].float().dot(h) if lowrank.numel() else weight.new_zeros(())
        )
        delta_loss = delta * local_linear + 0.5 * delta.square() * (diagonal[i].float() + norm_u_sq[i])
        if preserve_nonempty_clusters and counts[current].item() <= 1:
            mask = torch.ones_like(delta_loss, dtype=torch.bool)
            mask[current] = False
            delta_loss = delta_loss.masked_fill(mask, torch.inf)
        best = torch.argmin(delta_loss)
        best_delta = delta[best]
        if delta_loss[best] < -eps and int(best.item()) != current:
            labels[i] = best
            quantized[i] = codebook.float()[best]
            error[i] = quantized[i] - weight.float()[i]
            counts[current] -= 1
            counts[best] += 1
            if lowrank.numel():
                h = h + best_delta * lowrank[i].float()
    return labels
