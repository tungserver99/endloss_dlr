from __future__ import annotations

from dataclasses import dataclass, field
import logging

import torch


@dataclass
class MethodAConfig:
    max_outer_iters: int = 8
    max_inner_iters: int = 8
    rel_tol: float = 1e-7
    numerical_eps: float = 1e-12
    tie_tol: float = 0.0
    constraint_tol: float = 1e-5
    dual_init: float = 0.0
    codebook_update_interval: int = 1


@dataclass
class MethodAResult:
    labels: torch.Tensor
    codebooks: torch.Tensor
    losses: torch.Tensor
    fallback_rows: int
    diagnostics: dict[str, object] = field(default_factory=dict)


def gather_quantized(codebooks: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return torch.gather(codebooks, 1, labels.long())


def quadratic_rows(error: torch.Tensor, hessian: torch.Tensor) -> torch.Tensor:
    return (error.matmul(hessian) * error).sum(dim=1)


def nll_surrogate(weight, gradient, hessian_nll, codebooks, labels):
    error = gather_quantized(codebooks, labels) - weight
    return (gradient * error).sum(dim=1) + 0.5 * quadratic_rows(error, hessian_nll)


def fixed_dual_objective(weight, gradient, hessian_nll, hessian_kl, mu, nu, codebooks, labels):
    error = gather_quantized(codebooks, labels) - weight
    return (
        (gradient * error).sum(dim=1)
        + 0.5 * quadratic_rows(error, hessian_nll)
        + 0.5 * mu * quadratic_rows(error, hessian_kl)
        + 0.5 * nu * error.square().sum(dim=1)
    )


def sort_codebooks_and_remap(codebooks: torch.Tensor, labels: torch.Tensor):
    sorted_codebooks, permutation = torch.sort(codebooks, dim=1)
    inverse = torch.empty_like(permutation)
    indices = torch.arange(permutation.shape[1], device=permutation.device).expand_as(permutation)
    inverse.scatter_(1, permutation, indices)
    return sorted_codebooks, torch.gather(inverse, 1, labels.long())


def nearest_sorted_codeword(target, codebooks, current_labels, tie_tol=0.0):
    """Batched O(B*n*log(K)) projection without materializing [B,n,K]."""
    clusters = codebooks.shape[1]
    if clusters == 1:
        return torch.zeros_like(current_labels, dtype=torch.long)
    insertion = torch.searchsorted(codebooks.contiguous(), target.contiguous(), right=False)
    right = insertion.clamp(max=clusters - 1)
    left = (insertion - 1).clamp(min=0)
    left_value = torch.gather(codebooks, 1, left)
    right_value = torch.gather(codebooks, 1, right)
    projected = torch.where((target - left_value).abs() <= (target - right_value).abs(), left, right)
    current_value = torch.gather(codebooks, 1, current_labels.long())
    projected_value = torch.gather(codebooks, 1, projected)
    keep_current = (target - current_value).abs() <= (target - projected_value).abs() + tie_tol
    return torch.where(keep_current, current_labels.long(), projected.long())


def parallel_mm_assignment(
    weight, gradient, hessian_nll, hessian_kl, row_abs_nll, row_abs_kl,
    mu, nu, codebooks, labels, numerical_eps, tie_tol, return_target=False,
):
    quantized = gather_quantized(codebooks, labels)
    error = quantized - weight
    slope = (
        gradient + error.matmul(hessian_nll)
        + mu[:, None] * error.matmul(hessian_kl) + nu[:, None] * error
    )
    majorizer = row_abs_nll[None, :] + mu[:, None] * row_abs_kl[None, :] + nu[:, None]
    positive = majorizer > numerical_eps
    target = torch.where(positive, quantized - slope / majorizer.clamp_min(numerical_eps), quantized)
    assigned = nearest_sorted_codeword(target, codebooks, labels, tie_tol)
    assigned = torch.where((~positive) & (slope > numerical_eps), torch.zeros_like(assigned), assigned)
    assigned = torch.where(
        (~positive) & (slope < -numerical_eps),
        torch.full_like(assigned, codebooks.shape[1] - 1), assigned,
    )
    return (assigned, target) if return_target else assigned


def _aggregate_hessian_by_labels(hessian, labels, clusters):
    # P^T H P via two grouped reductions; the temporary is [n,K], not [n,n] int64.
    columns = hessian.new_zeros(hessian.shape[0], clusters)
    columns.index_add_(1, labels, hessian)
    output = hessian.new_zeros(clusters, clusters)
    output.index_add_(0, labels, columns)
    return output


def _scatter_vector_by_labels(values, labels, clusters):
    return torch.bincount(labels, weights=values, minlength=clusters)


def exact_codebook_update(
    weight, gradient, hessian_nll, hessian_kl, labels, old_codebooks, mu, nu,
):
    """Solve (P^T A P)c = P^T A w - P^T g on active clusters."""
    clusters = old_codebooks.shape[1]
    output = old_codebooks.clone()
    for row_idx in range(weight.shape[0]):
        row_labels = labels[row_idx].long()
        counts = torch.bincount(row_labels, minlength=clusters)
        active_ids = torch.nonzero(counts > 0, as_tuple=False).flatten()
        if active_ids.numel() == 0:
            continue
        lhs = _aggregate_hessian_by_labels(hessian_nll, row_labels, clusters)
        lhs.add_(_aggregate_hessian_by_labels(hessian_kl, row_labels, clusters), alpha=float(mu[row_idx]))
        lhs.diagonal().add_(counts.to(lhs.dtype), alpha=float(nu[row_idx]))
        aw_minus_g = (
            hessian_nll.matmul(weight[row_idx])
            + mu[row_idx] * hessian_kl.matmul(weight[row_idx])
            + nu[row_idx] * weight[row_idx] - gradient[row_idx]
        )
        rhs = _scatter_vector_by_labels(aw_minus_g, row_labels, clusters)
        active_lhs = lhs.index_select(0, active_ids).index_select(1, active_ids)
        active_rhs = rhs.index_select(0, active_ids)
        solution, info = torch.linalg.solve_ex(active_lhs, active_rhs)
        if int(info.item()) != 0 or not torch.isfinite(solution).all():
            solution = torch.linalg.lstsq(active_lhs, active_rhs[:, None]).solution[:, 0]
        if torch.isfinite(solution).all():
            output[row_idx, active_ids] = solution
    return output


def _validate_inputs(weight, gradient, hessian_nll, hessian_kl, initial_labels, initial_codebooks):
    if weight.ndim != 2 or gradient.shape != weight.shape:
        raise ValueError("weight and gradient must have matching [rows, in_features] shapes")
    n = weight.shape[1]
    if hessian_nll.shape != (n, n) or hessian_kl.shape != (n, n):
        raise ValueError("each curvature must have shape [in_features, in_features]")
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
        ("hessian_nll", hessian_nll),
        ("hessian_kl", hessian_kl),
        ("initial_codebooks", initial_codebooks),
    ):
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} contains non-finite values")


def quantize_rows_method_a(
    weight, gradient, hessian_nll, hessian_kl, initial_labels, initial_codebooks,
    config: MethodAConfig | None = None,
):
    """Quantize rows sharing one pair of GuidedQuant grouped curvatures."""
    cfg = config or MethodAConfig()
    _validate_inputs(weight, gradient, hessian_nll, hessian_kl, initial_labels, initial_codebooks)
    weight, gradient = weight.float(), gradient.float()
    hessian_nll = hessian_nll.float()
    hessian_nll = 0.5 * (hessian_nll + hessian_nll.T)
    hessian_kl = hessian_kl.float()
    hessian_kl = 0.5 * (hessian_kl + hessian_kl.T)
    labels = initial_labels.long().clone()
    codebooks, labels = sort_codebooks_and_remap(initial_codebooks.float().clone(), labels)

    q0 = gather_quantized(codebooks, labels)
    error0 = q0 - weight
    epsilon_kl = 0.5 * quadratic_rows(error0, hessian_kl)
    epsilon_weight = 0.5 * error0.square().sum(dim=1)
    best_labels, best_codebooks = labels.clone(), codebooks.clone()
    best_loss = nll_surrogate(weight, gradient, hessian_nll, codebooks, labels)
    q0_loss = best_loss.clone()

    rows = weight.shape[0]
    mu = weight.new_full((rows,), float(cfg.dual_init))
    nu = weight.new_full((rows,), float(cfg.dual_init))
    trace_nll = hessian_nll.diagonal().mean().clamp_min(cfg.numerical_eps)
    trace_kl = hessian_kl.diagonal().mean().clamp_min(cfg.numerical_eps)
    eta_mu = trace_nll / trace_kl
    eta_nu = trace_nll
    row_abs_nll = hessian_nll.abs().sum(dim=1)
    row_abs_kl = hessian_kl.abs().sum(dim=1)
    assignment_rejections = codebook_rejections = 0
    inner_history = []
    outer_history = []
    previous_outer_labels = None
    labels_two_outer_iters_ago = None

    for outer in range(max(0, int(cfg.max_outer_iters))):
        objective = fixed_dual_objective(
            weight, gradient, hessian_nll, hessian_kl, mu, nu, codebooks, labels,
        )
        for inner in range(max(1, int(cfg.max_inner_iters))):
            labels_before = labels.clone()
            codebooks_before = codebooks.clone()
            objective_before = objective

            candidate_labels, mm_target = parallel_mm_assignment(
                weight, gradient, hessian_nll, hessian_kl, row_abs_nll, row_abs_kl,
                mu, nu, codebooks, labels, cfg.numerical_eps, cfg.tie_tol,
                return_target=True,
            )
            assignment_objective = fixed_dual_objective(
                weight, gradient, hessian_nll, hessian_kl, mu, nu, codebooks, candidate_labels,
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
                candidate_codebooks = exact_codebook_update(
                    weight, gradient, hessian_nll, hessian_kl, labels, codebooks, mu, nu,
                )
                candidate_codebooks, candidate_labels = sort_codebooks_and_remap(
                    candidate_codebooks, labels
                )
                codebook_objective = fixed_dual_objective(
                    weight, gradient, hessian_nll, hessian_kl, mu, nu,
                    candidate_codebooks, candidate_labels,
                )
                accept_codebook = codebook_objective <= objective_after_assignment
                codebook_rejections += int((~accept_codebook).sum())
                labels = torch.where(accept_codebook[:, None], candidate_labels, labels)
                codebooks = torch.where(accept_codebook[:, None], candidate_codebooks, codebooks)
                objective_after_codebook = torch.where(
                    accept_codebook, codebook_objective, objective_after_assignment
                )

            scale = objective_before.abs().clamp_min(1.0)
            relative_improvement = (objective_before - objective_after_codebook) / scale
            label_changes = int((labels != labels_before).sum())
            codebook_stable = torch.allclose(
                codebooks, codebooks_before, rtol=0.0, atol=cfg.numerical_eps
            )
            labels_stable = torch.equal(labels, labels_before)
            inner_history.append({
                "outer": outer,
                "inner": inner,
                "objective_before": objective_before.detach().cpu(),
                "objective_after_assignment": objective_after_assignment.detach().cpu(),
                "objective_after_codebook": objective_after_codebook.detach().cpu(),
                "label_changes": label_changes,
                "max_abs_target": mm_target.abs().max().detach().cpu(),
                "max_abs_codebook": codebooks.abs().max().detach().cpu(),
            })
            logging.debug(
                "Method A inner outer=%d inner=%d labels_changed=%d "
                "max_J_before=%.6e max_J_assignment=%.6e max_J_codebook=%.6e "
                "max_abs_target=%.6e max_abs_codebook=%.6e",
                outer,
                inner,
                label_changes,
                float(objective_before.max()),
                float(objective_after_assignment.max()),
                float(objective_after_codebook.max()),
                float(mm_target.abs().max()),
                float(codebooks.abs().max()),
            )
            objective = objective_after_codebook
            if codebook_updated and (
                (labels_stable and codebook_stable)
                or bool((relative_improvement <= cfg.rel_tol).all())
            ):
                break

        error = gather_quantized(codebooks, labels) - weight
        cost_kl = 0.5 * quadratic_rows(error, hessian_kl)
        cost_weight = 0.5 * error.square().sum(dim=1)
        kl_allowance = cfg.constraint_tol * epsilon_kl.abs().clamp_min(cfg.numerical_eps)
        weight_allowance = cfg.constraint_tol * epsilon_weight.abs().clamp_min(cfg.numerical_eps)
        feasible = (cost_kl <= epsilon_kl + kl_allowance) & (
            cost_weight <= epsilon_weight + weight_allowance
        )
        candidate_loss = nll_surrogate(weight, gradient, hessian_nll, codebooks, labels)
        improve = feasible & (candidate_loss < best_loss)
        best_loss = torch.where(improve, candidate_loss, best_loss)
        best_labels = torch.where(improve[:, None], labels, best_labels)
        best_codebooks = torch.where(improve[:, None], codebooks, best_codebooks)

        violation_kl = cost_kl / epsilon_kl.clamp_min(cfg.numerical_eps) - 1.0
        violation_weight = cost_weight / epsilon_weight.clamp_min(cfg.numerical_eps) - 1.0
        violation_kl = torch.where(
            epsilon_kl > cfg.numerical_eps,
            violation_kl,
            torch.where(
                cost_kl <= kl_allowance,
                torch.zeros_like(cost_kl),
                torch.ones_like(cost_kl),
            ),
        )
        violation_weight = torch.where(
            epsilon_weight > cfg.numerical_eps,
            violation_weight,
            torch.where(
                cost_weight <= weight_allowance,
                torch.zeros_like(cost_weight),
                torch.ones_like(cost_weight),
            ),
        )
        ratio_kl = cost_kl / epsilon_kl.clamp_min(cfg.numerical_eps)
        ratio_weight = cost_weight / epsilon_weight.clamp_min(cfg.numerical_eps)
        outer_history.append({
            "outer": outer,
            "ratio_kl": ratio_kl.detach().cpu(),
            "ratio_weight": ratio_weight.detach().cpu(),
            "feasible": feasible.detach().cpu(),
        })
        logging.debug(
            "Method A outer=%d feasible=%d/%d max_KL_ratio=%.6e max_weight_ratio=%.6e",
            outer,
            int(feasible.sum()),
            rows,
            float(ratio_kl.max()),
            float(ratio_weight.max()),
        )
        oscillating = (
            labels_two_outer_iters_ago is not None
            and torch.equal(labels, labels_two_outer_iters_ago)
            and not torch.equal(labels, previous_outer_labels)
        )
        if oscillating:
            logging.debug("Method A stopped on a two-cycle at outer=%d", outer)
            break
        labels_two_outer_iters_ago = previous_outer_labels
        previous_outer_labels = labels.clone()
        mu = (mu + eta_mu * violation_kl).clamp_min(0.0)
        nu = (nu + eta_nu * violation_weight).clamp_min(0.0)

    final_q = gather_quantized(best_codebooks, best_labels)
    final_error = final_q - weight
    final_cost_kl = 0.5 * quadratic_rows(final_error, hessian_kl)
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
            "trace_h_nll": hessian_nll.diagonal().sum().detach(),
            "trace_h_kl": hessian_kl.diagonal().sum().detach(),
            "row_abs_nll": row_abs_nll.detach(),
            "row_abs_kl": row_abs_kl.detach(),
            "assignment_rejections": assignment_rejections,
            "codebook_rejections": codebook_rejections,
            "inner_history": inner_history,
            "outer_history": outer_history,
        },
    )