from __future__ import annotations

import logging

import torch

from .method_a import (
    MethodAConfig,
    MethodAResult,
    gather_quantized,
    nearest_sorted_codeword,
    sort_codebooks_and_remap,
)


def lowrank_matmul_rows(error: torch.Tensor, diagonal: torch.Tensor, factor: torch.Tensor) -> torch.Tensor:
    return error * diagonal[None, :] + (error @ factor) @ factor.T


def lowrank_quadratic_rows(error: torch.Tensor, diagonal: torch.Tensor, factor: torch.Tensor) -> torch.Tensor:
    projected = error @ factor
    return (error.square() * diagonal[None, :]).sum(dim=1) + projected.square().sum(dim=1)


def lowrank_row_majorizer(diagonal: torch.Tensor, factor: torch.Tensor) -> torch.Tensor:
    column_l1 = factor.abs().sum(dim=0)
    return diagonal + factor.abs() @ column_l1


def lowrank_nll_surrogate(weight, gradient, d_nll, u_nll, codebooks, labels):
    error = gather_quantized(codebooks, labels) - weight
    return (gradient * error).sum(dim=1) + 0.5 * lowrank_quadratic_rows(error, d_nll, u_nll)


def lowrank_fixed_dual_objective(
    weight, gradient, d_nll, u_nll, d_kl, u_kl, mu, nu, codebooks, labels,
):
    error = gather_quantized(codebooks, labels) - weight
    return (
        (gradient * error).sum(dim=1)
        + 0.5 * lowrank_quadratic_rows(error, d_nll, u_nll)
        + 0.5 * mu * lowrank_quadratic_rows(error, d_kl, u_kl)
        + 0.5 * nu * error.square().sum(dim=1)
    )


def parallel_mm_assignment_lowrank(
    weight, gradient, d_nll, u_nll, d_kl, u_kl, row_majorizer_nll,
    row_majorizer_kl, mu, nu, codebooks, labels, numerical_eps, tie_tol,
    return_target=False,
):
    quantized = gather_quantized(codebooks, labels)
    error = quantized - weight
    slope = (
        gradient
        + lowrank_matmul_rows(error, d_nll, u_nll)
        + mu[:, None] * lowrank_matmul_rows(error, d_kl, u_kl)
        + nu[:, None] * error
    )
    majorizer = (
        row_majorizer_nll[None, :]
        + mu[:, None] * row_majorizer_kl[None, :]
        + nu[:, None]
    )
    positive = majorizer > numerical_eps
    target = torch.where(
        positive,
        quantized - slope / majorizer.clamp_min(numerical_eps),
        quantized,
    )
    assigned = nearest_sorted_codeword(target, codebooks, labels, tie_tol)
    assigned = torch.where(
        (~positive) & (slope > numerical_eps), torch.zeros_like(assigned), assigned
    )
    assigned = torch.where(
        (~positive) & (slope < -numerical_eps),
        torch.full_like(assigned, codebooks.shape[1] - 1),
        assigned,
    )
    return (assigned, target) if return_target else assigned


def _scatter_rows(values: torch.Tensor, labels: torch.Tensor, clusters: int) -> torch.Tensor:
    output = values.new_zeros(clusters, values.shape[1])
    output.index_add_(0, labels, values)
    return output


def exact_codebook_update_lowrank(
    weight, gradient, d_nll, u_nll, d_kl, u_kl, labels, old_codebooks, mu, nu,
):
    clusters = old_codebooks.shape[1]
    output = old_codebooks.clone()
    for row_idx in range(weight.shape[0]):
        row_labels = labels[row_idx].long()
        counts = torch.bincount(row_labels, minlength=clusters)
        active_ids = torch.nonzero(counts > 0, as_tuple=False).flatten()
        if active_ids.numel() == 0:
            continue

        d_a = d_nll + mu[row_idx] * d_kl + nu[row_idx]
        diagonal_l = torch.bincount(
            row_labels, weights=d_a, minlength=clusters
        )
        v_nll = _scatter_rows(u_nll, row_labels, clusters)
        v_kl = _scatter_rows(u_kl, row_labels, clusters)
        lhs = torch.diag(diagonal_l)
        lhs.add_(v_nll @ v_nll.T)
        lhs.add_(v_kl @ v_kl.T, alpha=float(mu[row_idx]))

        row_weight = weight[row_idx]
        aw_minus_g = (
            d_a * row_weight
            + u_nll @ (u_nll.T @ row_weight)
            + mu[row_idx] * (u_kl @ (u_kl.T @ row_weight))
            - gradient[row_idx]
        )
        rhs = torch.bincount(
            row_labels, weights=aw_minus_g, minlength=clusters
        )
        active_lhs = lhs.index_select(0, active_ids).index_select(1, active_ids)
        active_rhs = rhs.index_select(0, active_ids)
        solution, info = torch.linalg.solve_ex(active_lhs, active_rhs)
        if int(info.item()) != 0 or not torch.isfinite(solution).all():
            solution = torch.linalg.lstsq(active_lhs, active_rhs[:, None]).solution[:, 0]
        if torch.isfinite(solution).all():
            output[row_idx, active_ids] = solution
    return output


def _validate_inputs(
    weight, gradient, d_nll, u_nll, d_kl, u_kl, initial_labels, initial_codebooks,
):
    if weight.ndim != 2 or gradient.shape != weight.shape:
        raise ValueError("weight and gradient must have matching [rows, in_features] shapes")
    n = weight.shape[1]
    for name, diagonal, factor in (
        ("nll", d_nll, u_nll),
        ("kl", d_kl, u_kl),
    ):
        if diagonal.shape != (n,) or factor.ndim != 2 or factor.shape[0] != n:
            raise ValueError(f"{name} curvature must have D=[n] and U=[n, rank]")
        if not torch.isfinite(diagonal).all() or not torch.isfinite(factor).all():
            raise ValueError(f"{name} curvature contains non-finite values")
        if bool((diagonal < 0).any()):
            raise ValueError(f"{name} diagonal must be non-negative")
    if initial_labels.shape != weight.shape:
        raise ValueError("initial_labels must match weight shape")
    if initial_codebooks.ndim != 2 or initial_codebooks.shape[0] != weight.shape[0]:
        raise ValueError("initial_codebooks must have shape [rows, K]")
    if initial_labels.numel() and (
        int(initial_labels.min()) < 0
        or int(initial_labels.max()) >= initial_codebooks.shape[1]
    ):
        raise ValueError("initial_labels contains an index outside the codebook")
    for name, tensor in (
        ("weight", weight),
        ("gradient", gradient),
        ("initial_codebooks", initial_codebooks),
    ):
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} contains non-finite values")


def quantize_rows_method_a_lowrank(
    weight, gradient, d_nll, u_nll, d_kl, u_kl, initial_labels,
    initial_codebooks, config: MethodAConfig | None = None,
    row_majorizer_nll: torch.Tensor | None = None,
    row_majorizer_kl: torch.Tensor | None = None,
):
    cfg = config or MethodAConfig()
    _validate_inputs(
        weight, gradient, d_nll, u_nll, d_kl, u_kl,
        initial_labels, initial_codebooks,
    )
    weight, gradient = weight.float(), gradient.float()
    d_nll, u_nll = d_nll.float(), u_nll.float()
    d_kl, u_kl = d_kl.float(), u_kl.float()
    labels = initial_labels.long().clone()
    codebooks, labels = sort_codebooks_and_remap(
        initial_codebooks.float().clone(), labels
    )

    q0 = gather_quantized(codebooks, labels)
    error0 = q0 - weight
    epsilon_kl = 0.5 * lowrank_quadratic_rows(error0, d_kl, u_kl)
    epsilon_weight = 0.5 * error0.square().sum(dim=1)
    best_labels, best_codebooks = labels.clone(), codebooks.clone()
    best_loss = lowrank_nll_surrogate(
        weight, gradient, d_nll, u_nll, codebooks, labels
    )
    q0_loss = best_loss.clone()

    rows, n = weight.shape
    mu = weight.new_full((rows,), float(cfg.dual_init))
    nu = weight.new_full((rows,), float(cfg.dual_init))
    trace_nll = (d_nll.sum() + u_nll.square().sum()) / n
    trace_kl = (d_kl.sum() + u_kl.square().sum()) / n
    trace_nll = trace_nll.clamp_min(cfg.numerical_eps)
    trace_kl = trace_kl.clamp_min(cfg.numerical_eps)
    eta_mu = trace_nll / trace_kl
    eta_nu = trace_nll
    if row_majorizer_nll is None:
        row_majorizer_nll = lowrank_row_majorizer(d_nll, u_nll)
    else:
        row_majorizer_nll = row_majorizer_nll.float()
    if row_majorizer_kl is None:
        row_majorizer_kl = lowrank_row_majorizer(d_kl, u_kl)
    else:
        row_majorizer_kl = row_majorizer_kl.float()
    if row_majorizer_nll.shape != (n,) or row_majorizer_kl.shape != (n,):
        raise ValueError("row majorizers must have shape [in_features]")
    if not torch.isfinite(row_majorizer_nll).all() or not torch.isfinite(row_majorizer_kl).all():
        raise ValueError("row majorizers contain non-finite values")
    assignment_rejections = codebook_rejections = 0
    inner_history, outer_history = [], []
    previous_outer_labels = labels_two_outer_iters_ago = None

    for outer in range(max(0, int(cfg.max_outer_iters))):
        objective = lowrank_fixed_dual_objective(
            weight, gradient, d_nll, u_nll, d_kl, u_kl,
            mu, nu, codebooks, labels,
        )
        for inner in range(max(1, int(cfg.max_inner_iters))):
            labels_before = labels.clone()
            codebooks_before = codebooks.clone()
            objective_before = objective
            candidate_labels, mm_target = parallel_mm_assignment_lowrank(
                weight, gradient, d_nll, u_nll, d_kl, u_kl,
                row_majorizer_nll, row_majorizer_kl, mu, nu,
                codebooks, labels, cfg.numerical_eps, cfg.tie_tol,
                return_target=True,
            )
            assignment_objective = lowrank_fixed_dual_objective(
                weight, gradient, d_nll, u_nll, d_kl, u_kl,
                mu, nu, codebooks, candidate_labels,
            )
            accept_assignment = assignment_objective <= objective
            assignment_rejections += int((~accept_assignment).sum())
            labels = torch.where(accept_assignment[:, None], candidate_labels, labels)
            objective_after_assignment = torch.where(
                accept_assignment, assignment_objective, objective
            )

            objective_after_codebook = objective_after_assignment
            codebook_updated = False
            if cfg.codebook_update_interval <= 1 or (inner + 1) % cfg.codebook_update_interval == 0:
                codebook_updated = True
                candidate_codebooks = exact_codebook_update_lowrank(
                    weight, gradient, d_nll, u_nll, d_kl, u_kl,
                    labels, codebooks, mu, nu,
                )
                candidate_codebooks, candidate_labels = sort_codebooks_and_remap(
                    candidate_codebooks, labels
                )
                codebook_objective = lowrank_fixed_dual_objective(
                    weight, gradient, d_nll, u_nll, d_kl, u_kl,
                    mu, nu, candidate_codebooks, candidate_labels,
                )
                accept_codebook = codebook_objective <= objective_after_assignment
                codebook_rejections += int((~accept_codebook).sum())
                labels = torch.where(accept_codebook[:, None], candidate_labels, labels)
                codebooks = torch.where(
                    accept_codebook[:, None], candidate_codebooks, codebooks
                )
                objective_after_codebook = torch.where(
                    accept_codebook, codebook_objective, objective_after_assignment
                )

            scale = objective_before.abs().clamp_min(1.0)
            relative_improvement = (objective_before - objective_after_codebook) / scale
            labels_stable = torch.equal(labels, labels_before)
            codebook_stable = torch.allclose(
                codebooks, codebooks_before, rtol=0.0, atol=cfg.numerical_eps
            )
            inner_history.append({
                "outer": outer,
                "inner": inner,
                "objective_before": objective_before.detach().cpu(),
                "objective_after_assignment": objective_after_assignment.detach().cpu(),
                "objective_after_codebook": objective_after_codebook.detach().cpu(),
                "label_changes": int((labels != labels_before).sum()),
                "max_abs_target": mm_target.abs().max().detach().cpu(),
                "max_abs_codebook": codebooks.abs().max().detach().cpu(),
            })
            objective = objective_after_codebook
            if codebook_updated and (
                (labels_stable and codebook_stable)
                or bool((relative_improvement <= cfg.rel_tol).all())
            ):
                break

        error = gather_quantized(codebooks, labels) - weight
        cost_kl = 0.5 * lowrank_quadratic_rows(error, d_kl, u_kl)
        cost_weight = 0.5 * error.square().sum(dim=1)
        kl_allowance = cfg.constraint_tol * epsilon_kl.abs().clamp_min(cfg.numerical_eps)
        weight_allowance = cfg.constraint_tol * epsilon_weight.abs().clamp_min(cfg.numerical_eps)
        feasible = (cost_kl <= epsilon_kl + kl_allowance) & (
            cost_weight <= epsilon_weight + weight_allowance
        )
        candidate_loss = lowrank_nll_surrogate(
            weight, gradient, d_nll, u_nll, codebooks, labels
        )
        improve = feasible & (candidate_loss < best_loss)
        best_loss = torch.where(improve, candidate_loss, best_loss)
        best_labels = torch.where(improve[:, None], labels, best_labels)
        best_codebooks = torch.where(improve[:, None], codebooks, best_codebooks)

        violation_kl = cost_kl / epsilon_kl.clamp_min(cfg.numerical_eps) - 1.0
        violation_weight = cost_weight / epsilon_weight.clamp_min(cfg.numerical_eps) - 1.0
        violation_kl = torch.where(
            epsilon_kl > cfg.numerical_eps,
            violation_kl,
            torch.where(cost_kl <= kl_allowance, torch.zeros_like(cost_kl), torch.ones_like(cost_kl)),
        )
        violation_weight = torch.where(
            epsilon_weight > cfg.numerical_eps,
            violation_weight,
            torch.where(cost_weight <= weight_allowance, torch.zeros_like(cost_weight), torch.ones_like(cost_weight)),
        )
        ratio_kl = cost_kl / epsilon_kl.clamp_min(cfg.numerical_eps)
        ratio_weight = cost_weight / epsilon_weight.clamp_min(cfg.numerical_eps)
        outer_history.append({
            "outer": outer,
            "ratio_kl": ratio_kl.detach().cpu(),
            "ratio_weight": ratio_weight.detach().cpu(),
            "feasible": feasible.detach().cpu(),
        })
        oscillating = (
            labels_two_outer_iters_ago is not None
            and torch.equal(labels, labels_two_outer_iters_ago)
            and not torch.equal(labels, previous_outer_labels)
        )
        if oscillating:
            logging.debug("Method A low-rank stopped on a two-cycle at outer=%d", outer)
            break
        labels_two_outer_iters_ago = previous_outer_labels
        previous_outer_labels = labels.clone()
        mu = (mu + eta_mu * violation_kl).clamp_min(0.0)
        nu = (nu + eta_nu * violation_weight).clamp_min(0.0)

    final_q = gather_quantized(best_codebooks, best_labels)
    final_error = final_q - weight
    final_cost_kl = 0.5 * lowrank_quadratic_rows(final_error, d_kl, u_kl)
    final_cost_weight = 0.5 * final_error.square().sum(dim=1)
    q0_rows = (final_q == q0).all(dim=1)
    return MethodAResult(
        labels=best_labels,
        codebooks=best_codebooks,
        losses=best_loss,
        fallback_rows=int(q0_rows.sum()),
        diagnostics={
            "epsilon_kl": epsilon_kl.detach(),
            "epsilon_weight": epsilon_weight.detach(),
            "cost_kl": final_cost_kl.detach(),
            "cost_weight": final_cost_weight.detach(),
            "q0_loss": q0_loss.detach(),
            "trace_h_nll": (d_nll.sum() + u_nll.square().sum()).detach(),
            "trace_h_kl": (d_kl.sum() + u_kl.square().sum()).detach(),
            "row_abs_nll": row_majorizer_nll.detach(),
            "row_abs_kl": row_majorizer_kl.detach(),
            "assignment_rejections": assignment_rejections,
            "codebook_rejections": codebook_rejections,
            "inner_history": inner_history,
            "outer_history": outer_history,
        },
    )
