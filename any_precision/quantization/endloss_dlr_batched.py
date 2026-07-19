from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch

from .endloss_dlr import DLRConfig, quantize_group_dlr


@dataclass
class BatchedDLRResult:
    labels: torch.Tensor
    codebooks: torch.Tensor
    losses: torch.Tensor
    fallback_rows: int = 0


def _scatter_sum_batched(values: torch.Tensor, labels: torch.Tensor, K: int) -> torch.Tensor:
    out = values.new_zeros(values.shape[0], K)
    out.scatter_add_(1, labels.long(), values)
    return out


def _scatter_sum_rows_batched(values: torch.Tensor, labels: torch.Tensor, K: int) -> torch.Tensor:
    out = values.new_zeros(values.shape[0], K, values.shape[-1])
    index = labels.long().unsqueeze(-1).expand(-1, -1, values.shape[-1])
    out.scatter_add_(1, index, values)
    return out


def _batched_loss(
    w: torch.Tensor,
    g: torch.Tensor,
    d_A: torch.Tensor,
    U_A: torch.Tensor,
    alpha: torch.Tensor,
    codebook: torch.Tensor,
    labels: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    q = torch.gather(codebook, 1, labels.long())
    e = q - w
    h = e @ U_A
    curvature = (d_A.unsqueeze(0) * e.square()).sum(dim=1) + h.square().sum(dim=1)
    return beta * (g * e).sum(dim=1) + 0.5 * alpha * curvature


def _continuous_target_batched(
    w: torch.Tensor,
    g: torch.Tensor,
    d_A: torch.Tensor,
    U_A: torch.Tensor,
    alpha: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    inv_d = 1.0 / d_A.clamp_min(torch.finfo(d_A.dtype).tiny)
    y = g * inv_d.unsqueeze(0)
    if U_A.shape[-1] == 0:
        Ainv_g = y
    else:
        DU = inv_d.unsqueeze(-1) * U_A
        R = torch.eye(U_A.shape[-1], device=U_A.device, dtype=U_A.dtype) + U_A.transpose(0, 1) @ DU
        rhs = y @ U_A
        v = torch.linalg.solve(R, rhs.transpose(0, 1)).transpose(0, 1)
        Ainv_g = y - v @ DU.transpose(0, 1)
    return w - beta * Ainv_g / alpha.clamp_min(torch.finfo(alpha.dtype).tiny).unsqueeze(1)


def _initialize_labels_batched(x: torch.Tensor, rho_base: torch.Tensor, K: int) -> torch.Tensor:
    rows, n = x.shape
    if K <= 1:
        return torch.zeros(rows, n, device=x.device, dtype=torch.long)
    K = min(K, n)
    order = torch.argsort(x, dim=1)
    rho_sorted = rho_base.unsqueeze(0).expand(rows, -1).gather(1, order)
    cumsum = torch.cumsum(rho_sorted, dim=1)
    total = cumsum[:, -1:].clamp_min(torch.finfo(cumsum.dtype).tiny)
    thresholds = total * torch.arange(1, K, device=x.device, dtype=x.dtype).unsqueeze(0) / K
    boundaries = torch.searchsorted(cumsum.contiguous(), thresholds.contiguous(), right=True).long()

    lower = torch.arange(1, K, device=x.device, dtype=torch.long).unsqueeze(0)
    upper = torch.arange(n - K + 1, n, device=x.device, dtype=torch.long).unsqueeze(0)
    boundaries = torch.maximum(boundaries, lower)
    boundaries = torch.minimum(boundaries, upper)
    for i in range(1, K - 1):
        boundaries[:, i] = torch.maximum(boundaries[:, i], boundaries[:, i - 1] + 1)
    for i in range(K - 3, -1, -1):
        boundaries[:, i] = torch.minimum(boundaries[:, i], boundaries[:, i + 1] - 1)

    positions = torch.arange(n, device=x.device).unsqueeze(0).expand(rows, -1)
    labels_sorted = torch.zeros(rows, n, device=x.device, dtype=torch.long)
    for k in range(K - 1):
        labels_sorted += (positions >= boundaries[:, k:k + 1]).long()

    labels = torch.empty_like(labels_sorted)
    labels.scatter_(1, order, labels_sorted)
    return labels


def _initial_codebook_batched(x: torch.Tensor, labels: torch.Tensor, K: int) -> torch.Tensor:
    one = torch.ones_like(x)
    sums = _scatter_sum_batched(x, labels, K)
    counts = _scatter_sum_batched(one, labels, K).clamp_min(1.0)
    return sums / counts


def _exact_codebook_update_batched(
    w: torch.Tensor,
    g: torch.Tensor,
    d_A: torch.Tensor,
    U_A: torch.Tensor,
    alpha: torch.Tensor,
    labels: torch.Tensor,
    old_codebook: torch.Tensor,
    beta: float,
    K: int,
) -> torch.Tensor:
    rows = w.shape[0]
    d_expand = d_A.unsqueeze(0).expand(rows, -1)
    U_expand = U_A.unsqueeze(0).expand(rows, -1, -1)

    A_base = _scatter_sum_batched(d_expand, labels, K)
    B_base = _scatter_sum_batched(d_expand * w, labels, K)
    G = _scatter_sum_batched(g, labels, K)
    M_base = _scatter_sum_rows_batched(U_expand, labels, K)

    A = alpha.unsqueeze(1) * A_base
    b = alpha.unsqueeze(1) * B_base - beta * G
    sqrt_alpha = alpha.sqrt()
    M = sqrt_alpha[:, None, None] * M_base
    z = sqrt_alpha[:, None] * (w @ U_A)
    active = A > 0

    inv_A = torch.where(active, 1.0 / A.clamp_min(torch.finfo(A.dtype).tiny), torch.zeros_like(A))
    Q = torch.einsum("bkr,bks,bk->brs", M, M, inv_A)
    v = (M * (b * inv_A).unsqueeze(-1)).sum(dim=1) - z
    eye = torch.eye(U_A.shape[-1], device=w.device, dtype=w.dtype).unsqueeze(0).expand(rows, -1, -1)
    if U_A.shape[-1] == 0:
        h = v
    else:
        h = torch.linalg.solve(eye + Q, v.unsqueeze(-1)).squeeze(-1)
    updated = (b - torch.einsum("bkr,br->bk", M, h)) / A.clamp_min(torch.finfo(A.dtype).tiny)
    return torch.where(active, updated, old_codebook)


def _sort_codebook_and_remap_batched(codebook: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    sorted_codebook, perm = torch.sort(codebook, dim=1)
    inv_perm = torch.empty_like(perm)
    arange_k = torch.arange(perm.shape[1], device=perm.device).unsqueeze(0).expand_as(perm)
    inv_perm.scatter_(1, perm, arange_k)
    return sorted_codebook, torch.gather(inv_perm, 1, labels.long())


def _assignment_batched(
    w: torch.Tensor,
    g: torch.Tensor,
    d_A: torch.Tensor,
    U_A: torch.Tensor,
    alpha: torch.Tensor,
    lambda_A: torch.Tensor,
    codebook: torch.Tensor,
    labels: torch.Tensor,
    beta: float,
    tie_tol: float,
) -> torch.Tensor:
    q = torch.gather(codebook, 1, labels.long())
    e = q - w
    lowrank_grad = (e @ U_A) @ U_A.transpose(0, 1)
    curvature_grad = d_A.unsqueeze(0) * e + lowrank_grad
    grad = beta * g + alpha.unsqueeze(1) * curvature_grad
    denom = alpha.unsqueeze(1) * (d_A + lambda_A).unsqueeze(0)
    target = q - grad / denom.clamp_min(torch.finfo(denom.dtype).tiny)

    dist = (target.unsqueeze(-1) - codebook.unsqueeze(1)).abs()
    best_dist, best_labels = dist.min(dim=-1)
    current_dist = dist.gather(2, labels.long().unsqueeze(-1)).squeeze(-1)
    keep_current = current_dist <= best_dist + tie_tol
    return torch.where(keep_current, labels.long(), best_labels.long())


def quantize_rows_dlr_batched(
    w: torch.Tensor,
    g: torch.Tensor,
    d_A: torch.Tensor,
    U_A: torch.Tensor,
    alpha: torch.Tensor,
    K: int,
    config: DLRConfig,
) -> BatchedDLRResult:
    w = w.float()
    g = g.float()
    d_A = d_A.float().clamp_min(config.d_min)
    U_A = U_A.float()
    alpha = alpha.float().clamp_min(torch.finfo(torch.float32).tiny)
    rows = w.shape[0]
    K = min(int(K), int(w.shape[1]))

    if U_A.shape[-1] == 0:
        lambda_A = w.new_zeros(())
    else:
        gram = U_A.transpose(0, 1) @ U_A
        lambda_A = config.lambda_safety * torch.linalg.eigvalsh(gram)[-1].clamp_min(0.0)

    x = _continuous_target_batched(w, g, d_A, U_A, alpha, config.beta)
    rho_base = d_A + U_A.square().sum(dim=-1)
    labels = _initialize_labels_batched(x, rho_base, K)
    codebook = _initial_codebook_batched(x, labels, K)
    codebook = _exact_codebook_update_batched(w, g, d_A, U_A, alpha, labels, codebook, config.beta, K)
    codebook, labels = _sort_codebook_and_remap_batched(codebook, labels)
    old_loss = _batched_loss(w, g, d_A, U_A, alpha, codebook, labels, config.beta)

    active_mask = torch.ones(rows, device=w.device, dtype=torch.bool)
    fallback_rows = 0

    for _ in range(config.max_outer_iters):
        if not active_mask.any():
            break

        old_labels = labels.clone()
        candidate_labels = _assignment_batched(
            w=w,
            g=g,
            d_A=d_A,
            U_A=U_A,
            alpha=alpha,
            lambda_A=lambda_A,
            codebook=codebook,
            labels=labels,
            beta=config.beta,
            tie_tol=config.tie_tol,
        )
        candidate_codebook = _exact_codebook_update_batched(
            w, g, d_A, U_A, alpha, candidate_labels, codebook, config.beta, K
        )
        candidate_codebook, candidate_labels = _sort_codebook_and_remap_batched(candidate_codebook, candidate_labels)
        candidate_loss = _batched_loss(w, g, d_A, U_A, alpha, candidate_codebook, candidate_labels, config.beta)

        loss_scale = old_loss.abs().clamp_min(1.0)
        increased = active_mask & (candidate_loss > old_loss + 1e-6 * loss_scale)
        if increased.any():
            bad_rows = torch.nonzero(increased, as_tuple=False).flatten()
            fallback_rows += int(bad_rows.numel())
            for row in bad_rows.tolist():
                U_row = alpha[row].sqrt() * U_A
                d_row = alpha[row] * d_A
                row_codebook, row_labels, row_loss = quantize_group_dlr(
                    w=w[row],
                    g=g[row],
                    d=d_row,
                    U=U_row,
                    K=K,
                    config=config,
                )
                candidate_codebook[row] = row_codebook
                candidate_labels[row] = row_labels
                candidate_loss[row] = row_loss

        labels = torch.where(active_mask.unsqueeze(1), candidate_labels, labels)
        codebook = torch.where(active_mask.unsqueeze(1), candidate_codebook, codebook)
        new_loss = torch.where(active_mask, candidate_loss, old_loss)

        relative_drop = (old_loss - new_loss).abs() / loss_scale
        labels_unchanged = (labels == old_labels).all(dim=1)
        row_done = active_mask & (labels_unchanged | (relative_drop <= config.rel_tol) | increased)
        old_loss = new_loss
        active_mask = active_mask & ~row_done

    return BatchedDLRResult(labels=labels, codebooks=codebook, losses=old_loss, fallback_rows=fallback_rows)
