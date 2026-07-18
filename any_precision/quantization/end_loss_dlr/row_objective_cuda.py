from __future__ import annotations

import logging

import torch


def require_cuda(*tensors: torch.Tensor):
    for tensor in tensors:
        if tensor is None:
            continue
        if not isinstance(tensor, torch.Tensor) or not tensor.is_cuda:
            raise ValueError("All tensors in the End-Loss DLR solver must be CUDA tensors")


def _ensure_batch_row(x: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if x.ndim == 1:
        return x.unsqueeze(0), True
    return x, False


def _ensure_batch_factor(U: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if U.ndim == 2:
        return U.unsqueeze(0), True
    return U, False


def dlr_loss(
    w: torch.Tensor,
    g: torch.Tensor,
    d: torch.Tensor,
    U: torch.Tensor,
    codebook: torch.Tensor,
    labels: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    require_cuda(w, g, d, U, codebook, labels)
    w_b, squeezed = _ensure_batch_row(w.float())
    g_b, _ = _ensure_batch_row(g.float())
    d_b, _ = _ensure_batch_row(d.float())
    U_b, _ = _ensure_batch_factor(U.float())
    codebook_b, _ = _ensure_batch_row(codebook.float())
    labels_b, _ = _ensure_batch_row(labels.long())
    q = torch.gather(codebook_b, 1, labels_b)
    e = q - w_b
    if U_b.numel() and U_b.shape[-1] > 0:
        h = torch.einsum("bnr,bn->br", U_b, e)
        lowrank_term = 0.5 * (h.square().sum(dim=1))
    else:
        lowrank_term = torch.zeros(w_b.shape[0], device=w_b.device, dtype=torch.float32)
    loss = beta * (g_b * e).sum(dim=1) + 0.5 * (d_b * e.square()).sum(dim=1) + lowrank_term
    return loss[0] if squeezed else loss


def spectral_lambda(U: torch.Tensor, safety: float = 1.01) -> torch.Tensor:
    require_cuda(U)
    U_b, squeezed = _ensure_batch_factor(U.float())
    if U_b.numel() == 0 or U_b.shape[-1] == 0:
        out = torch.zeros(U_b.shape[0], device=U_b.device, dtype=torch.float32)
        return out[0] if squeezed else out
    gram = torch.einsum("bnr,bns->brs", U_b, U_b)
    eigvals = torch.linalg.eigvalsh(gram)
    out = safety * eigvals[:, -1].clamp_min(0.0)
    return out[0] if squeezed else out


def continuous_dlr_target(
    w: torch.Tensor,
    g: torch.Tensor,
    d: torch.Tensor,
    U: torch.Tensor,
    beta: float,
    eps: float,
) -> torch.Tensor:
    require_cuda(w, g, d, U)
    w_b, squeezed = _ensure_batch_row(w.float())
    g_b, _ = _ensure_batch_row(g.float())
    d_b, _ = _ensure_batch_row(d.float())
    U_b, _ = _ensure_batch_factor(U.float())
    inv_d = d_b.clamp_min(eps).reciprocal()
    y = inv_d * g_b
    if U_b.numel() == 0 or U_b.shape[-1] == 0:
        out = w_b - beta * y
        return out[0] if squeezed else out
    DU = inv_d.unsqueeze(-1) * U_b
    eye = torch.eye(U_b.shape[-1], device=U_b.device, dtype=torch.float32).expand(U_b.shape[0], -1, -1)
    R = eye + torch.einsum("bnr,bns->brs", U_b, DU)
    rhs = torch.einsum("bnr,bn->br", U_b, y)
    v = torch.linalg.solve(R, rhs.unsqueeze(-1)).squeeze(-1)
    Hinv_g = y - torch.einsum("bnr,br->bn", DU, v)
    out = w_b - beta * Hinv_g
    return out[0] if squeezed else out


def enforce_nonempty_intervals(boundaries: torch.Tensor, n: int, K: int) -> torch.Tensor:
    boundaries = boundaries.long().clone()
    if K <= 1:
        return boundaries.new_empty((0,))
    lower = torch.arange(1, K, device=boundaries.device, dtype=torch.long)
    upper = n - (K - 1 - torch.arange(K - 1, device=boundaries.device, dtype=torch.long))
    boundaries = boundaries.clamp(1, n - 1)
    for i in range(K - 1):
        min_val = lower[i]
        if i > 0:
            min_val = torch.maximum(min_val, boundaries[i - 1] + 1)
        boundaries[i] = torch.maximum(boundaries[i], min_val)
    for i in range(K - 2, -1, -1):
        max_val = upper[i]
        if i < K - 2:
            max_val = torch.minimum(max_val, boundaries[i + 1] - 1)
        boundaries[i] = torch.minimum(boundaries[i], max_val)
    return boundaries


def initialize_labels_from_target(
    x: torch.Tensor,
    d: torch.Tensor,
    U: torch.Tensor,
    K: int,
) -> torch.Tensor:
    require_cuda(x, d, U)
    if x.ndim != 1:
        raise ValueError("initialize_labels_from_target expects a single row")
    if K == 1:
        return torch.zeros(x.numel(), device=x.device, dtype=torch.long)
    rho = d.float() + (U.float().square().sum(dim=-1) if U.numel() else torch.zeros_like(d.float()))
    order = torch.argsort(x.float(), stable=True)
    rho_sorted = rho[order]
    cumsum = torch.cumsum(rho_sorted, dim=0)
    total = cumsum[-1]
    thresholds = total * torch.arange(1, K, device=x.device, dtype=torch.float32) / K
    boundaries = torch.searchsorted(cumsum, thresholds).long()
    boundaries = enforce_nonempty_intervals(boundaries, n=x.numel(), K=K)
    labels_sorted = torch.empty(x.numel(), device=x.device, dtype=torch.long)
    start = 0
    for cluster_id, end in enumerate(boundaries.tolist() + [x.numel()]):
        labels_sorted[start:end] = cluster_id
        start = end
    labels = torch.empty_like(labels_sorted)
    labels[order] = labels_sorted
    return labels


def initialize_labels_from_target_batched(
    x: torch.Tensor,
    d: torch.Tensor,
    U: torch.Tensor,
    K: int,
) -> torch.Tensor:
    require_cuda(x, d, U)
    x_b, squeezed = _ensure_batch_row(x.float())
    d_b, _ = _ensure_batch_row(d.float())
    U_b, _ = _ensure_batch_factor(U.float())
    labels = []
    for b in range(x_b.shape[0]):
        labels.append(initialize_labels_from_target(x_b[b], d_b[b], U_b[b], K))
    out = torch.stack(labels, dim=0)
    return out[0] if squeezed else out


def initial_placeholder_codebook(x: torch.Tensor, labels: torch.Tensor, K: int) -> torch.Tensor:
    require_cuda(x, labels)
    if x.ndim != 1:
        raise ValueError("initial_placeholder_codebook expects a single row")
    sums = torch.zeros(K, device=x.device, dtype=torch.float32)
    counts = torch.zeros(K, device=x.device, dtype=torch.float32)
    sums.scatter_add_(0, labels.long(), x.float())
    counts.scatter_add_(0, labels.long(), torch.ones_like(x.float()))
    codebook = sums / counts.clamp_min(1.0)
    if torch.any(counts == 0):
        filler = torch.linspace(x.float().min(), x.float().max(), steps=K, device=x.device, dtype=torch.float32)
        codebook = torch.where(counts > 0, codebook, filler)
    return codebook


def initial_placeholder_codebook_batched(x: torch.Tensor, labels: torch.Tensor, K: int) -> torch.Tensor:
    require_cuda(x, labels)
    x_b, squeezed = _ensure_batch_row(x.float())
    labels_b, _ = _ensure_batch_row(labels.long())
    B = x_b.shape[0]
    sums = torch.zeros((B, K), device=x_b.device, dtype=torch.float32)
    counts = torch.zeros((B, K), device=x_b.device, dtype=torch.float32)
    sums.scatter_add_(1, labels_b, x_b)
    counts.scatter_add_(1, labels_b, torch.ones_like(x_b))
    codebook = sums / counts.clamp_min(1.0)
    if torch.any(counts == 0):
        mins = x_b.min(dim=1).values.unsqueeze(1)
        maxs = x_b.max(dim=1).values.unsqueeze(1)
        filler = torch.linspace(0.0, 1.0, steps=K, device=x_b.device, dtype=torch.float32).unsqueeze(0)
        filler = mins + (maxs - mins) * filler
        codebook = torch.where(counts > 0, codebook, filler)
    return codebook[0] if squeezed else codebook


def nearest_codeword_with_current_tie_break(
    target: torch.Tensor,
    codebook: torch.Tensor,
    current_labels: torch.Tensor,
    tie_tol: float = 0.0,
) -> torch.Tensor:
    require_cuda(target, codebook, current_labels)
    target_b, squeezed = _ensure_batch_row(target.float())
    codebook_b, _ = _ensure_batch_row(codebook.float())
    current_b, _ = _ensure_batch_row(current_labels.long())
    dist = (target_b.unsqueeze(-1) - codebook_b.unsqueeze(1)).abs()
    best_dist = dist.min(dim=-1).values
    current_dist = dist.gather(dim=-1, index=current_b.unsqueeze(-1)).squeeze(-1)
    keep_current = current_dist <= (best_dist + tie_tol)
    new_labels = dist.argmin(dim=-1)
    out = torch.where(keep_current, current_b, new_labels.long())
    return out[0] if squeezed else out


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
    require_cuda(w, g, d, U, codebook, labels)
    w_b, squeezed = _ensure_batch_row(w.float())
    g_b, _ = _ensure_batch_row(g.float())
    d_b, _ = _ensure_batch_row(d.float())
    U_b, _ = _ensure_batch_factor(U.float())
    codebook_b, _ = _ensure_batch_row(codebook.float())
    labels_b, _ = _ensure_batch_row(labels.long())
    lambda_b = lambda_.reshape(-1).float() if isinstance(lambda_, torch.Tensor) and lambda_.ndim > 0 else torch.as_tensor(lambda_, device=w_b.device, dtype=torch.float32).reshape(1)
    if lambda_b.numel() == 1 and w_b.shape[0] > 1:
        lambda_b = lambda_b.expand(w_b.shape[0])
    q = torch.gather(codebook_b, 1, labels_b)
    e = q - w_b
    if U_b.numel() and U_b.shape[-1] > 0:
        h = torch.einsum("bnr,bn->br", U_b, e)
        grad = beta * g_b + d_b * e + torch.einsum("bnr,br->bn", U_b, h)
    else:
        grad = beta * g_b + d_b * e
    target = q - grad / (d_b + lambda_b.unsqueeze(1))
    out = nearest_codeword_with_current_tie_break(target, codebook_b, labels_b, tie_tol=tie_tol)
    return out[0] if squeezed else out


def exact_dlr_codebook_update(
    w: torch.Tensor,
    g: torch.Tensor,
    d: torch.Tensor,
    U: torch.Tensor,
    labels: torch.Tensor,
    old_codebook: torch.Tensor,
    beta: float,
    K: int,
    eps: float,
) -> torch.Tensor:
    require_cuda(w, g, d, U, labels, old_codebook)
    codebook_b = exact_dlr_codebook_update_batched(
        w.unsqueeze(0),
        g.unsqueeze(0),
        d.unsqueeze(0),
        U.unsqueeze(0) if U.ndim == 2 else U,
        labels.unsqueeze(0),
        old_codebook.unsqueeze(0),
        beta,
        K,
        eps,
    )
    return codebook_b[0]


def exact_dlr_codebook_update_batched(
    w: torch.Tensor,
    g: torch.Tensor,
    d: torch.Tensor,
    U: torch.Tensor,
    labels: torch.Tensor,
    old_codebook: torch.Tensor,
    beta: float,
    K: int,
    eps: float,
) -> torch.Tensor:
    require_cuda(w, g, d, U, labels, old_codebook)
    w_b, _ = _ensure_batch_row(w.float())
    g_b, _ = _ensure_batch_row(g.float())
    d_b, _ = _ensure_batch_row(d.float())
    U_b, _ = _ensure_batch_factor(U.float())
    labels_b, _ = _ensure_batch_row(labels.long())
    codebook_b, _ = _ensure_batch_row(old_codebook.float())
    B, n = w_b.shape
    A = torch.zeros((B, K), device=w_b.device, dtype=torch.float32)
    Bsum = torch.zeros((B, K), device=w_b.device, dtype=torch.float32)
    Gsum = torch.zeros((B, K), device=w_b.device, dtype=torch.float32)
    A.scatter_add_(1, labels_b, d_b)
    Bsum.scatter_add_(1, labels_b, d_b * w_b)
    Gsum.scatter_add_(1, labels_b, g_b)
    b = Bsum - beta * Gsum
    new_codebook = codebook_b.clone()
    if U_b.numel() == 0 or U_b.shape[-1] == 0:
        active = A > 0
        new_codebook = torch.where(active, b / A.clamp_min(eps), new_codebook)
        return new_codebook
    r = U_b.shape[-1]
    M = torch.zeros((B, K, r), device=w_b.device, dtype=torch.float32)
    expanded_labels = labels_b.unsqueeze(-1).expand(-1, -1, r)
    M.scatter_add_(1, expanded_labels, U_b)
    z = torch.einsum("bnr,bn->br", U_b, w_b)
    eye = torch.eye(r, device=w_b.device, dtype=torch.float32)
    for bidx in range(B):
        active = A[bidx] > 0
        if not torch.any(active):
            continue
        A_a = A[bidx, active]
        b_a = b[bidx, active]
        M_a = M[bidx, active]
        Q = torch.einsum("kr,ks,k->rs", M_a, M_a, A_a.reciprocal())
        v = (M_a * (b_a / A_a).unsqueeze(1)).sum(dim=0) - z[bidx]
        h = torch.linalg.solve(eye + Q, v.unsqueeze(1)).squeeze(1)
        new_codebook[bidx, active] = (b_a - M_a.matmul(h)) / A_a
    return new_codebook


def sort_codebook_and_remap_labels(codebook: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    require_cuda(codebook, labels)
    codebook_b, squeezed = _ensure_batch_row(codebook.float())
    labels_b, _ = _ensure_batch_row(labels.long())
    sorted_codebook, perm = torch.sort(codebook_b, dim=1)
    inv_perm = torch.empty_like(perm)
    inv_perm.scatter_(1, perm, torch.arange(perm.shape[1], device=perm.device, dtype=perm.dtype).unsqueeze(0).expand_as(perm))
    remapped = torch.gather(inv_perm, 1, labels_b)
    return (sorted_codebook[0], remapped[0]) if squeezed else (sorted_codebook, remapped)


def quantize_group_dlr(
    w: torch.Tensor,
    g: torch.Tensor,
    d: torch.Tensor,
    U: torch.Tensor,
    K: int,
    beta: float = 0.5,
    max_outer_iters: int = 8,
    rel_tol: float = 1e-7,
    lambda_safety: float = 1.01,
    tie_tol: float = 0.0,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, list[float]]:
    codebook, labels, losses = quantize_group_dlr_batched(
        w.unsqueeze(0),
        g.unsqueeze(0),
        d.unsqueeze(0),
        U.unsqueeze(0) if U.ndim == 2 else U,
        K,
        beta,
        max_outer_iters,
        rel_tol,
        lambda_safety,
        tie_tol,
        eps,
    )
    return codebook[0], labels[0], [float(x) for x in losses[0].tolist() if x == x]


def quantize_group_dlr_batched(
    w: torch.Tensor,
    g: torch.Tensor,
    d: torch.Tensor,
    U: torch.Tensor,
    K: int,
    beta: float = 0.5,
    max_outer_iters: int = 8,
    rel_tol: float = 1e-7,
    lambda_safety: float = 1.01,
    tie_tol: float = 0.0,
    eps: float = 1e-12,
    log_prefix: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    require_cuda(w, g, d, U)
    w_b, _ = _ensure_batch_row(w.float())
    g_b, _ = _ensure_batch_row(g.float())
    d_b, _ = _ensure_batch_row(d.float())
    U_b, _ = _ensure_batch_factor(U.float())
    d_b = d_b.clamp_min(eps)


    lambda_b = spectral_lambda(U_b, safety=lambda_safety)
    x = continuous_dlr_target(w_b, g_b, d_b, U_b, beta, eps)
    labels = initialize_labels_from_target_batched(x, d_b, U_b, K)
    codebook = initial_placeholder_codebook_batched(x, labels, K)
    codebook = exact_dlr_codebook_update_batched(w_b, g_b, d_b, U_b, labels, codebook, beta, K, eps)
    codebook, labels = sort_codebook_and_remap_labels(codebook, labels)

    old_loss = dlr_loss(w_b, g_b, d_b, U_b, codebook, labels, beta)
    history = [old_loss]
    active_mask = torch.ones(w_b.shape[0], device=w_b.device, dtype=torch.bool)

    for iteration in range(max_outer_iters):
        prev_labels = labels.clone()
        previous_codebook = codebook
        updated_labels = parallel_mm_assignment(w_b, g_b, d_b, U_b, codebook, labels, beta, lambda_b, tie_tol=tie_tol)
        labels = torch.where(active_mask.unsqueeze(1), updated_labels, labels)
        assignment_labels = labels
        updated_codebook = exact_dlr_codebook_update_batched(w_b, g_b, d_b, U_b, labels, codebook, beta, K, eps)
        codebook = torch.where(active_mask.unsqueeze(1), updated_codebook, codebook)
        codebook, labels = sort_codebook_and_remap_labels(codebook, labels)
        new_loss = dlr_loss(w_b, g_b, d_b, U_b, codebook, labels, beta)
        loss_scale = old_loss.abs().clamp_min(1.0)

        increased = new_loss > old_loss + 1e-6 * loss_scale
        if torch.any(increased):
            assignment_loss = dlr_loss(
                w_b, g_b, d_b, U_b, previous_codebook, assignment_labels, beta
            )
            assignment_increased = assignment_loss > old_loss + 1e-6 * loss_scale
            codebook_scale = assignment_loss.abs().clamp_min(1.0)
            codebook_increased = new_loss > assignment_loss + 1e-6 * codebook_scale
            max_increase = float((new_loss - old_loss)[increased].max().item())
            count = int(increased.sum().item())
            raise RuntimeError(
                "DLR loss unexpectedly increased "
                f"after outer iteration {iteration + 1} for {count} rows; "
                f"max increase={max_increase:.3e}; "
                f"assignment violations={int(assignment_increased.sum().item())}; "
                f"codebook violations={int(codebook_increased.sum().item())}"
            )

        history.append(new_loss)
        labels_unchanged = (labels == prev_labels).all(dim=1)
        relative_drop = (old_loss - new_loss).abs() / loss_scale
        done = labels_unchanged | (relative_drop <= rel_tol)
        active_mask = active_mask & (~done)
        old_loss = new_loss
        if not torch.any(active_mask):
            break

    loss_history = torch.stack(history, dim=1)
    return codebook, labels, loss_history
