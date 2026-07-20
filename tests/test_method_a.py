import torch
import torch.nn.functional as F

from any_precision.quantization.method_a import (
    MethodAConfig,
    exact_codebook_update,
    fixed_dual_objective,
    gather_quantized,
    nearest_sorted_codeword,
    parallel_mm_assignment,
    quadratic_rows,
    quantize_rows_method_a,
    sort_codebooks_and_remap,
)
from any_precision.quantization.method_a_curvature import (
    _group_sensitivity,
    ground_truth_nll_sum,
    teacher_score_sum,
)


def _psd_matrix(n, seed):
    generator = torch.Generator().manual_seed(seed)
    factor = torch.randn(n + 2, n, generator=generator)
    return factor.T @ factor / factor.shape[0]


def test_grouped_curvature_construction_is_psd_and_normalized():
    generator = torch.Generator().manual_seed(1)
    x = torch.randn(17, 6, generator=generator)
    sensitivity = torch.rand(17, generator=generator)
    hessian = x.T @ (x * sensitivity[:, None]) / x.shape[0]
    eigenvalues = torch.linalg.eigvalsh(0.5 * (hessian + hessian.T))
    assert eigenvalues.min() >= -1e-6
    expected_trace = (x.square().sum(dim=1) * sensitivity).mean()
    torch.testing.assert_close(hessian.trace(), expected_trace)


def test_absolute_row_sum_majorizes_fixed_dual_hessian():
    n = 9
    h_nll = _psd_matrix(n, 2)
    h_kl = _psd_matrix(n, 3)
    mu, nu = 0.7, 0.2
    matrix = h_nll + mu * h_kl + nu * torch.eye(n)
    diagonal = h_nll.abs().sum(1) + mu * h_kl.abs().sum(1) + nu
    difference = torch.diag(diagonal) - matrix
    assert torch.linalg.eigvalsh(difference).min() >= -2e-5


def test_parallel_mm_assignment_is_monotone():
    generator = torch.Generator().manual_seed(4)
    rows, n, clusters = 5, 12, 4
    weight = torch.randn(rows, n, generator=generator)
    gradient = 0.03 * torch.randn(rows, n, generator=generator)
    h_nll = _psd_matrix(n, 5)
    h_kl = _psd_matrix(n, 6)
    codebooks = torch.sort(torch.randn(rows, clusters, generator=generator), dim=1).values
    labels = torch.randint(clusters, (rows, n), generator=generator)
    mu = torch.rand(rows, generator=generator)
    nu = 0.1 + torch.rand(rows, generator=generator)
    before = fixed_dual_objective(weight, gradient, h_nll, h_kl, mu, nu, codebooks, labels)
    updated = parallel_mm_assignment(
        weight,
        gradient,
        h_nll,
        h_kl,
        h_nll.abs().sum(1),
        h_kl.abs().sum(1),
        mu,
        nu,
        codebooks,
        labels,
        1e-12,
        0.0,
    )
    after = fixed_dual_objective(weight, gradient, h_nll, h_kl, mu, nu, codebooks, updated)
    assert torch.all(after <= before + 2e-5 * before.abs().clamp_min(1.0))


def test_zero_majorizer_uses_finite_linear_solution():
    weight = torch.zeros(1, 3)
    gradient = torch.tensor([[1.0, -1.0, 0.0]])
    zero = torch.zeros(3, 3)
    codebooks = torch.tensor([[-2.0, 0.0, 3.0]])
    labels = torch.tensor([[1, 1, 1]])
    updated = parallel_mm_assignment(
        weight,
        gradient,
        zero,
        zero,
        torch.zeros(3),
        torch.zeros(3),
        torch.zeros(1),
        torch.zeros(1),
        codebooks,
        labels,
        1e-12,
        0.0,
    )
    assert updated.tolist() == [[0, 2, 1]]


def test_nearest_codeword_keeps_current_label_on_tie():
    codebooks = torch.tensor([[0.0, 2.0, 4.0]])
    target = torch.tensor([[1.0, 3.0]])
    current = torch.tensor([[1, 1]])
    labels = nearest_sorted_codeword(target, codebooks, current)
    assert labels.tolist() == [[1, 1]]


def test_exact_codebook_update_decreases_objective():
    generator = torch.Generator().manual_seed(7)
    rows, n, clusters = 3, 10, 3
    weight = torch.randn(rows, n, generator=generator)
    gradient = 0.1 * torch.randn(rows, n, generator=generator)
    h_nll = _psd_matrix(n, 8)
    h_kl = _psd_matrix(n, 9)
    labels = torch.tensor([[0, 1, 2, 0, 1, 2, 0, 1, 2, 0]]).expand(rows, -1).clone()
    codebooks = torch.sort(torch.randn(rows, clusters, generator=generator), dim=1).values
    mu = 0.2 + torch.rand(rows, generator=generator)
    nu = 0.3 + torch.rand(rows, generator=generator)
    before = fixed_dual_objective(weight, gradient, h_nll, h_kl, mu, nu, codebooks, labels)
    updated = exact_codebook_update(
        weight, gradient, h_nll, h_kl, labels, codebooks, mu, nu
    )
    for row in range(rows):
        assignment = F.one_hot(labels[row], num_classes=clusters).float()
        matrix = h_nll + mu[row] * h_kl + nu[row] * torch.eye(n)
        lhs = assignment.T @ matrix @ assignment
        rhs = assignment.T @ (matrix @ weight[row] - gradient[row])
        torch.testing.assert_close(lhs @ updated[row], rhs, rtol=2e-4, atol=2e-4)
    updated, remapped = sort_codebooks_and_remap(updated, labels)
    after = fixed_dual_objective(weight, gradient, h_nll, h_kl, mu, nu, updated, remapped)
    assert torch.all(after <= before + 2e-5 * before.abs().clamp_min(1.0))


def test_q0_is_exact_fallback_and_defines_budgets():
    generator = torch.Generator().manual_seed(10)
    rows, n, clusters = 4, 11, 4
    weight = torch.randn(rows, n, generator=generator)
    gradient = torch.randn(rows, n, generator=generator) * 0.02
    h_nll = _psd_matrix(n, 11)
    h_kl = _psd_matrix(n, 12)
    codebooks = torch.sort(torch.randn(rows, clusters, generator=generator), dim=1).values
    labels = torch.randint(clusters, (rows, n), generator=generator)
    q0 = gather_quantized(codebooks, labels)
    result = quantize_rows_method_a(
        weight,
        gradient,
        h_nll,
        h_kl,
        labels,
        codebooks,
        MethodAConfig(max_outer_iters=0),
    )
    q_final = gather_quantized(result.codebooks, result.labels)
    torch.testing.assert_close(q_final, q0)
    error = q0 - weight
    torch.testing.assert_close(
        result.diagnostics["epsilon_kl"], 0.5 * quadratic_rows(error, h_kl)
    )
    torch.testing.assert_close(
        result.diagnostics["epsilon_weight"], 0.5 * error.square().sum(dim=1)
    )


def test_full_solver_returns_best_feasible_candidate():
    generator = torch.Generator().manual_seed(13)
    rows, n, clusters = 5, 13, 4
    weight = torch.randn(rows, n, generator=generator)
    gradient = 0.05 * torch.randn(rows, n, generator=generator)
    h_nll = _psd_matrix(n, 14)
    h_kl = _psd_matrix(n, 15)
    codebooks = torch.sort(torch.randn(rows, clusters, generator=generator), dim=1).values
    labels = torch.randint(clusters, (rows, n), generator=generator)
    config = MethodAConfig(max_outer_iters=4, max_inner_iters=5, constraint_tol=1e-5)
    result = quantize_rows_method_a(
        weight, gradient, h_nll, h_kl, labels, codebooks, config
    )
    eps_kl = result.diagnostics["epsilon_kl"]
    eps_w = result.diagnostics["epsilon_weight"]
    allowance_kl = config.constraint_tol * eps_kl.abs().clamp_min(config.numerical_eps)
    allowance_w = config.constraint_tol * eps_w.abs().clamp_min(config.numerical_eps)
    assert torch.all(result.diagnostics["cost_kl"] <= eps_kl + allowance_kl)
    assert torch.all(result.diagnostics["cost_weight"] <= eps_w + allowance_w)

def test_statistics_sources_are_distinct():
    logits = torch.tensor(
        [[[2.0, -1.0, 0.5], [0.1, 1.2, -0.7]]], requires_grad=True
    )
    ground_truth = torch.tensor([[0, 2]])
    nll = ground_truth_nll_sum(logits, ground_truth)
    expected_nll = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), ground_truth.reshape(-1), reduction="sum"
    )
    torch.testing.assert_close(nll, expected_nll)

    generator = torch.Generator().manual_seed(123)
    score, pseudo = teacher_score_sum(logits, generator)
    expected = F.log_softmax(logits, dim=-1).gather(-1, pseudo[..., None]).sum()
    torch.testing.assert_close(score, expected)
    assert pseudo.shape == ground_truth.shape


def test_sum_loss_curvature_matches_mean_loss_after_undoing_reduction():
    generator = torch.Generator().manual_seed(21)
    token_count, channels = 7, 6
    logits = torch.randn(token_count, channels, generator=generator, requires_grad=True)
    labels = torch.randint(channels, (token_count,), generator=generator)

    sum_gradient = torch.autograd.grad(
        F.cross_entropy(logits, labels, reduction="sum"), logits, retain_graph=True
    )[0]
    mean_gradient = torch.autograd.grad(
        F.cross_entropy(logits, labels, reduction="mean"), logits
    )[0]
    sum_curvature = _group_sensitivity(sum_gradient, num_groups=3) / token_count
    restored_mean_curvature = (
        _group_sensitivity(mean_gradient * token_count, num_groups=3) / token_count
    )
    torch.testing.assert_close(sum_curvature, restored_mean_curvature)

