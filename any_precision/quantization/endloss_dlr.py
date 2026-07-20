from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass
class DLRConfig:
    beta: float = 0.5
    rank: int = 4
    max_outer_iters: int = 8
    rel_tol: float = 1e-7
    lambda_safety: float = 1.01
    d_min: float = 1e-8
    tie_tol: float = 0.0


def _scatter_sum(values: torch.Tensor, labels: torch.Tensor, K: int) -> torch.Tensor:
    out = torch.zeros(K, device=values.device, dtype=values.dtype)
    out.scatter_add_(0, labels.long(), values)
    return out


def _scatter_sum_rows(values: torch.Tensor, labels: torch.Tensor, K: int) -> torch.Tensor:
    out = torch.zeros(K, values.shape[-1], device=values.device, dtype=values.dtype)
    index = labels.long().unsqueeze(-1).expand(-1, values.shape[-1])
    out.scatter_add_(0, index, values)
    return out


def dlr_loss(
    w: torch.Tensor,
    g: torch.Tensor,
    d: torch.Tensor,
    U: torch.Tensor,
    codebook: torch.Tensor,
    labels: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    w64 = w.double()
    g64 = g.double()
    d64 = d.double()
    U64 = U.double()
    codebook64 = codebook.double()
    q = codebook64[labels.long()]
    e = q - w64
    h = U64.transpose(-1, -2) @ e
    return beta * torch.dot(g64, e) + 0.5 * torch.dot(d64, e.square()) + 0.5 * torch.dot(h, h)

def spectral_lambda(U: torch.Tensor, safety: float = 1.01) -> torch.Tensor:
    if U.numel() == 0:
        return torch.zeros((), device=U.device, dtype=U.dtype)
    gram = U.transpose(-1, -2) @ U
    eigvals = torch.linalg.eigvalsh(gram)
    return safety * eigvals[-1].clamp_min(0.0)


def continuous_dlr_target(
    w: torch.Tensor,
    g: torch.Tensor,
    d: torch.Tensor,
    U: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    inv_d = 1.0 / d.clamp_min(torch.finfo(d.dtype).tiny)
    y = inv_d * g
    DU = inv_d.unsqueeze(-1) * U
    if U.shape[-1] == 0:
        h_inv_g = y
    else:
        R = torch.eye(U.shape[-1], device=U.device, dtype=U.dtype) + U.transpose(-1, -2) @ DU
        rhs = U.transpose(-1, -2) @ y
        v = torch.linalg.solve(R, rhs)
        h_inv_g = y - DU @ v
    return w - beta * h_inv_g


def _enforce_nonempty_intervals(boundaries: torch.Tensor, n: int, K: int) -> torch.Tensor:
    if K <= 1:
        return boundaries
    boundaries = boundaries.to(torch.long).clone()
    lower = torch.arange(1, K, device=boundaries.device, dtype=boundaries.dtype)
    upper = torch.arange(n - K + 1, n, device=boundaries.device, dtype=boundaries.dtype)
    boundaries = torch.maximum(boundaries, lower)
    boundaries = torch.minimum(boundaries, upper)
    for i in range(1, boundaries.numel()):
        min_allowed = boundaries[i - 1] + 1
        if boundaries[i] < min_allowed:
            boundaries[i] = min_allowed
    for i in range(boundaries.numel() - 2, -1, -1):
        max_allowed = boundaries[i + 1] - 1
        if boundaries[i] > max_allowed:
            boundaries[i] = max_allowed
    return boundaries


def initialize_labels_from_target(x: torch.Tensor, d: torch.Tensor, U: torch.Tensor, K: int) -> torch.Tensor:
    n = x.numel()
    if K <= 1 or n == 0:
        return torch.zeros(n, device=x.device, dtype=torch.long)
    K = min(K, n)
    rho = d + U.square().sum(dim=-1)
    order = torch.argsort(x)
    rho_sorted = rho[order]
    cumsum = torch.cumsum(rho_sorted, dim=0)
    total = cumsum[-1].clamp_min(torch.finfo(cumsum.dtype).tiny)
    thresholds = total * torch.arange(1, K, device=x.device, dtype=x.dtype) / K
    boundaries = torch.searchsorted(cumsum, thresholds, right=True)
    boundaries = _enforce_nonempty_intervals(boundaries, n=n, K=K)

    labels_sorted = torch.empty(n, device=x.device, dtype=torch.long)
    start = 0
    for cluster_id, stop in enumerate(boundaries.tolist() + [n]):
        labels_sorted[start:stop] = cluster_id
        start = stop

    labels = torch.empty_like(labels_sorted)
    labels[order] = labels_sorted
    return labels


def initial_placeholder_codebook(
    x: torch.Tensor,
    labels: torch.Tensor,
    K: int,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if weights is None:
        weights = torch.ones_like(x)
    codebook = torch.empty(K, device=x.device, dtype=x.dtype)
    fallback = (weights * x).sum() / weights.sum().clamp_min(torch.finfo(x.dtype).tiny)
    for k in range(K):
        mask = labels == k
        if mask.any():
            cluster_weights = weights[mask]
            codebook[k] = (cluster_weights * x[mask]).sum() / cluster_weights.sum().clamp_min(torch.finfo(x.dtype).tiny)
        else:
            codebook[k] = fallback
    return codebook


def lloyd_initialize_codebook(
    x: torch.Tensor,
    labels: torch.Tensor,
    codebook: torch.Tensor,
    K: int,
    weights: torch.Tensor | None = None,
    max_iters: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    if weights is None:
        weights = torch.ones_like(x)
    for _ in range(max(0, int(max_iters))):
        old_labels = labels
        labels = (x.unsqueeze(-1) - codebook.unsqueeze(0)).square().argmin(dim=-1)
        codebook = initial_placeholder_codebook(x, labels, K, weights=weights)
        codebook, labels = sort_codebook_and_remap_labels(codebook, labels)
        if torch.equal(labels, old_labels):
            break
    return codebook, labels


def _solve_box_qp(
    H: torch.Tensor,
    f: torch.Tensor,
    lb: torch.Tensor,
    ub: torch.Tensor,
    x0: torch.Tensor,
    active: torch.Tensor,
) -> torch.Tensor:
    x = x0.clamp(min=lb, max=ub)
    if not active.any():
        return x

    tol = 1e-9
    tiny = torch.finfo(H.dtype).tiny
    for _ in range(int(active.numel()) + 2):
        at_lower = active & (x <= lb + 1e-10)
        at_upper = active & (x >= ub - 1e-10)
        free = active & ~(at_lower | at_upper)
        if free.any():
            free_ids = torch.nonzero(free, as_tuple=False).flatten()
            fixed_ids = torch.nonzero(active & ~free, as_tuple=False).flatten()
            rhs = f.index_select(0, free_ids).clone()
            if fixed_ids.numel() > 0:
                rhs = rhs - H.index_select(0, free_ids).index_select(1, fixed_ids) @ x.index_select(0, fixed_ids)
            H_ff = H.index_select(0, free_ids).index_select(1, free_ids)
            H_ff = H_ff + torch.eye(free_ids.numel(), device=H.device, dtype=H.dtype) * tiny
            try:
                x_free = torch.linalg.solve(H_ff, rhs.unsqueeze(-1)).squeeze(-1)
            except RuntimeError:
                x_free = torch.linalg.lstsq(H_ff, rhs.unsqueeze(-1)).solution.squeeze(-1)
            x[free_ids] = x_free.clamp(min=lb[free_ids], max=ub[free_ids])

        grad = H @ x - f
        release_lower = at_lower & (grad < -tol)
        release_upper = at_upper & (grad > tol)
        if not release_lower.any() and not release_upper.any():
            break
        release_ids = torch.cat((torch.nonzero(release_lower, as_tuple=False).flatten(), torch.nonzero(release_upper, as_tuple=False).flatten()))
        x[release_ids] = x0[release_ids].clamp(min=lb[release_ids], max=ub[release_ids])
    return x


def exact_dlr_codebook_update(
    w: torch.Tensor,
    g: torch.Tensor,
    d: torch.Tensor,
    U: torch.Tensor,
    labels: torch.Tensor,
    old_codebook: torch.Tensor,
    beta: float,
    K: int,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
) -> torch.Tensor:
    w64 = w.double()
    g64 = g.double()
    d64 = d.double()
    U64 = U.double()
    old_codebook64 = old_codebook.double()
    A = _scatter_sum(d64, labels, K)
    B = _scatter_sum(d64 * w64, labels, K)
    G = _scatter_sum(g64, labels, K)
    M = _scatter_sum_rows(U64, labels, K)

    b = B - beta * G
    z = U64.transpose(-1, -2) @ w64
    active = A > 0

    new_codebook = old_codebook64.clone()
    if not active.any():
        return new_codebook.to(dtype=old_codebook.dtype)

    A_a = A[active]
    b_a = b[active]
    M_a = M[active]

    if U64.shape[-1] == 0:
        new_codebook[active] = b_a / A_a.clamp_min(torch.finfo(A_a.dtype).tiny)
    else:
        Q = torch.einsum("kr,ks,k->rs", M_a, M_a, 1.0 / A_a)
        v = (M_a * (b_a / A_a).unsqueeze(-1)).sum(dim=0) - z
        R = torch.eye(U64.shape[-1], device=U64.device, dtype=U64.dtype) + Q
        h = torch.linalg.solve(R, v)
        new_codebook[active] = (b_a - M_a @ h) / A_a

    if lower_bound is None or upper_bound is None:
        return new_codebook.to(dtype=old_codebook.dtype)

    H = M @ M.transpose(0, 1)
    H.diagonal().add_(A)
    f_box = b + M @ z
    lb = torch.full_like(new_codebook, float(lower_bound))
    ub = torch.full_like(new_codebook, float(upper_bound))
    solved = _solve_box_qp(H=H, f=f_box, lb=lb, ub=ub, x0=new_codebook, active=active)
    new_codebook[active] = solved[active]
    return new_codebook.to(dtype=old_codebook.dtype)

def sort_codebook_and_remap_labels(codebook: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    sorted_codebook, perm = torch.sort(codebook)
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(perm.numel(), device=perm.device, dtype=perm.dtype)
    return sorted_codebook, inv_perm[labels.long()]


def nearest_codeword_with_tie_break(
    target: torch.Tensor,
    codebook: torch.Tensor,
    current_labels: torch.Tensor,
    tie_tol: float = 0.0,
) -> torch.Tensor:
    dist = (target.unsqueeze(-1) - codebook.unsqueeze(0)).abs()
    best_dist, best_labels = dist.min(dim=-1)
    current_dist = dist.gather(dim=-1, index=current_labels.long().unsqueeze(-1)).squeeze(-1)
    keep_current = current_dist <= best_dist + tie_tol
    return torch.where(keep_current, current_labels.long(), best_labels.long())


def parallel_mm_assignment(
    w: torch.Tensor,
    g: torch.Tensor,
    d: torch.Tensor,
    U: torch.Tensor,
    codebook: torch.Tensor,
    labels: torch.Tensor,
    beta: float,
    lambda_: torch.Tensor,
    tie_tol: float = 0.0,
) -> torch.Tensor:
    q = codebook[labels.long()]
    e = q - w
    h = U.transpose(-1, -2) @ e
    grad = beta * g + d * e + U @ h
    target = q - grad / (d + lambda_).clamp_min(torch.finfo(d.dtype).tiny)
    return nearest_codeword_with_tie_break(target, codebook, labels, tie_tol=tie_tol)


def quantize_group_dlr(
    w: torch.Tensor,
    g: torch.Tensor,
    d: torch.Tensor,
    U: torch.Tensor,
    K: int,
    config: DLRConfig | None = None,
    initial_labels: torch.Tensor | None = None,
    initial_codebook: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cfg = config or DLRConfig()
    w = w.float()
    g = g.float()
    d = d.float().clamp_min(cfg.d_min)
    U = U.float()
    K = min(int(K), int(w.numel()))
    if K <= 0:
        raise ValueError("K must be positive")

    lambda_ = spectral_lambda(U, safety=cfg.lambda_safety)

    if initial_labels is None:
        x = continuous_dlr_target(w, g, d, U, cfg.beta)
        init_weights = torch.ones_like(d) if cfg.max_outer_iters == 0 else d + U.square().sum(dim=-1)
        labels = initialize_labels_from_target(x, init_weights, x.new_zeros((x.numel(), 0)), K)
        codebook = initial_placeholder_codebook(x, labels, K, weights=init_weights) if initial_codebook is None else initial_codebook.float()
        if cfg.max_outer_iters == 0:
            codebook, labels = lloyd_initialize_codebook(x, labels, codebook, K, weights=init_weights, max_iters=8)
    else:
        labels = initial_labels.long().clone()
        if initial_codebook is None:
            codebook = initial_placeholder_codebook(w, labels, K)
        else:
            codebook = initial_codebook.float().clone()

    codebook, labels = sort_codebook_and_remap_labels(codebook, labels)
    if cfg.max_outer_iters == 0:
        init_loss = dlr_loss(w, g, d, U, codebook, labels, cfg.beta)
        return codebook, labels, init_loss

    codebook = exact_dlr_codebook_update(
        w=w,
        g=g,
        d=d,
        U=U,
        labels=labels,
        old_codebook=codebook,
        beta=cfg.beta,
        K=K,
    )
    codebook, labels = sort_codebook_and_remap_labels(codebook, labels)
    old_loss = dlr_loss(w, g, d, U, codebook, labels, cfg.beta)

    for _ in range(cfg.max_outer_iters):
        old_labels = labels.clone()
        labels = parallel_mm_assignment(
            w=w,
            g=g,
            d=d,
            U=U,
            codebook=codebook,
            labels=labels,
            beta=cfg.beta,
            lambda_=lambda_,
            tie_tol=cfg.tie_tol,
        )
        codebook = exact_dlr_codebook_update(
            w=w,
            g=g,
            d=d,
            U=U,
            labels=labels,
            old_codebook=codebook,
            beta=cfg.beta,
            K=K,
        )
        codebook, labels = sort_codebook_and_remap_labels(codebook, labels)
        new_loss = dlr_loss(w, g, d, U, codebook, labels, cfg.beta)
        loss_scale = old_loss.abs().clamp_min(1.0)
        if new_loss > old_loss + 1e-6 * loss_scale:
            raise RuntimeError("DLR loss unexpectedly increased")
        relative_drop = (old_loss - new_loss).abs() / loss_scale
        old_loss = new_loss
        if torch.equal(labels, old_labels) or relative_drop <= cfg.rel_tol:
            break

    return codebook, labels, old_loss


def factorize_hessian_dlr(H: torch.Tensor, rank: int, d_min: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor]:
    H = H.float()
    diag_H = torch.diag(H)
    if rank <= 0:
        U = H.new_zeros(H.shape[0], 0)
        return diag_H.clamp_min(d_min), U

    q = min(rank, H.shape[0])
    try:
        eigvecs, eigvals, _ = torch.svd_lowrank(H, q=q)
    except RuntimeError:
        eigvals_full, eigvecs_full = torch.linalg.eigh(H)
        eigvals = eigvals_full[-q:].clamp_min(0.0)
        eigvecs = eigvecs_full[:, -q:]
    else:
        eigvals = eigvals.clamp_min(0.0)

    U = eigvecs * eigvals.sqrt().unsqueeze(0)
    d = (diag_H - U.square().sum(dim=-1)).clamp_min(d_min)
    return d, U

