import torch

from any_precision.quantization.method_a import (
    MethodAConfig,
    fixed_dual_objective,
    gather_quantized,
)
from any_precision.quantization.method_a_lowrank import (
    exact_codebook_update_lowrank,
    lowrank_fixed_dual_objective,
    lowrank_matmul_rows,
    lowrank_quadratic_rows,
    lowrank_row_majorizer,
    parallel_mm_assignment_lowrank,
    quantize_rows_method_a_lowrank,
)
from any_precision.quantization.method_a_lowrank_curvature import (
    StreamingDiagLowRankSketch,
)


def _curvature(n, rank, seed):
    generator = torch.Generator().manual_seed(seed)
    diagonal = torch.rand(n, generator=generator)
    factor = torch.randn(n, rank, generator=generator) / rank**0.5
    return diagonal, factor


def test_lowrank_ops_equal_dense_matrix():
    generator = torch.Generator().manual_seed(1)
    error = torch.randn(5, 9, generator=generator)
    diagonal, factor = _curvature(9, 4, 2)
    dense = torch.diag(diagonal) + factor @ factor.T
    torch.testing.assert_close(lowrank_matmul_rows(error, diagonal, factor), error @ dense)
    torch.testing.assert_close(
        lowrank_quadratic_rows(error, diagonal, factor),
        (error @ dense * error).sum(1),
    )


def test_streaming_statistics_match_direct_accumulation():
    generator = torch.Generator().manual_seed(3)
    x = torch.randn(19, 7, generator=generator)
    sensitivity = torch.rand(19, 3, generator=generator)
    sketch = StreamingDiagLowRankSketch(7, 3, 4, [11, 12, 13], "cpu", 5)
    sketch.update(x[:8], sensitivity[:8])
    sketch.update(x[8:], sensitivity[8:])

    for group_idx in range(3):
        z = x * sensitivity[:, group_idx].sqrt()[:, None]
        p = z @ sketch.omega[group_idx]
        torch.testing.assert_close(sketch.c[group_idx], z.T @ p)
        torch.testing.assert_close(sketch.b[group_idx], p.T @ p)
        torch.testing.assert_close(sketch.diag_h[group_idx], z.square().sum(0))


def test_finalized_curvature_is_psd_and_preserves_target_diagonal():
    generator = torch.Generator().manual_seed(4)
    x = torch.randn(31, 6, generator=generator)
    sensitivity = torch.rand(31, 2, generator=generator)
    sketch = StreamingDiagLowRankSketch(6, 2, 3, [21, 22], "cpu", 7)
    sketch.update(x, sensitivity)
    result = sketch.finalize(x.shape[0])
    for group_idx in range(2):
        diagonal = result["diagonal"][group_idx]
        factor = result["factor"][group_idx]
        matrix = torch.diag(diagonal) + factor @ factor.T
        assert torch.linalg.eigvalsh(matrix).min() >= -1e-6
        torch.testing.assert_close(
            matrix.diagonal(), result["target_diagonal"][group_idx],
            rtol=2e-5, atol=2e-6,
        )
        expected_diagonal = (
            x.square() * sensitivity[:, group_idx, None]
        ).sum(0) / x.shape[0]
        torch.testing.assert_close(
            result["target_diagonal"][group_idx], expected_diagonal
        )
        torch.testing.assert_close(
            result["row_majorizer"][group_idx],
            lowrank_row_majorizer(diagonal, factor),
        )


def test_lowrank_majorizer_is_safe():
    diagonal, factor = _curvature(10, 5, 5)
    dense = torch.diag(diagonal) + factor @ factor.T
    majorizer = lowrank_row_majorizer(diagonal, factor)
    assert torch.linalg.eigvalsh(torch.diag(majorizer) - dense).min() >= -2e-5


def test_lowrank_parallel_mm_is_monotone():
    generator = torch.Generator().manual_seed(6)
    rows, n, clusters = 4, 12, 4
    weight = torch.randn(rows, n, generator=generator)
    gradient = 0.03 * torch.randn(rows, n, generator=generator)
    d_nll, u_nll = _curvature(n, 4, 7)
    d_kl, u_kl = _curvature(n, 3, 8)
    codebooks = torch.sort(torch.randn(rows, clusters, generator=generator), 1).values
    labels = torch.randint(clusters, (rows, n), generator=generator)
    mu = torch.rand(rows, generator=generator)
    nu = 0.1 + torch.rand(rows, generator=generator)
    before = lowrank_fixed_dual_objective(
        weight, gradient, d_nll, u_nll, d_kl, u_kl,
        mu, nu, codebooks, labels,
    )
    updated = parallel_mm_assignment_lowrank(
        weight, gradient, d_nll, u_nll, d_kl, u_kl,
        lowrank_row_majorizer(d_nll, u_nll),
        lowrank_row_majorizer(d_kl, u_kl),
        mu, nu, codebooks, labels, 1e-12, 0.0,
    )
    after = lowrank_fixed_dual_objective(
        weight, gradient, d_nll, u_nll, d_kl, u_kl,
        mu, nu, codebooks, updated,
    )
    assert torch.all(after <= before + 2e-5 * before.abs().clamp_min(1))


def test_exact_lowrank_codebook_update_matches_dense_objective():
    generator = torch.Generator().manual_seed(9)
    rows, n, clusters = 3, 11, 4
    weight = torch.randn(rows, n, generator=generator)
    gradient = 0.05 * torch.randn(rows, n, generator=generator)
    d_nll, u_nll = _curvature(n, 3, 10)
    d_kl, u_kl = _curvature(n, 2, 11)
    labels = torch.randint(clusters, (rows, n), generator=generator)
    codebooks = torch.randn(rows, clusters, generator=generator)
    mu = torch.rand(rows, generator=generator)
    nu = 0.2 + torch.rand(rows, generator=generator)
    before = lowrank_fixed_dual_objective(
        weight, gradient, d_nll, u_nll, d_kl, u_kl,
        mu, nu, codebooks, labels,
    )
    updated = exact_codebook_update_lowrank(
        weight, gradient, d_nll, u_nll, d_kl, u_kl,
        labels, codebooks, mu, nu,
    )
    after = lowrank_fixed_dual_objective(
        weight, gradient, d_nll, u_nll, d_kl, u_kl,
        mu, nu, updated, labels,
    )
    assert torch.all(after <= before + 2e-5 * before.abs().clamp_min(1))
    h_nll = torch.diag(d_nll) + u_nll @ u_nll.T
    h_kl = torch.diag(d_kl) + u_kl @ u_kl.T
    lowrank_value = lowrank_fixed_dual_objective(
        weight, gradient, d_nll, u_nll, d_kl, u_kl,
        mu, nu, updated, labels,
    )
    dense_value = fixed_dual_objective(
        weight, gradient, h_nll, h_kl, mu, nu, updated, labels,
    )
    torch.testing.assert_close(lowrank_value, dense_value, rtol=2e-5, atol=2e-5)


def test_lowrank_q0_defines_budgets_and_is_feasible():
    generator = torch.Generator().manual_seed(12)
    rows, n, clusters = 4, 10, 4
    weight = torch.randn(rows, n, generator=generator)
    gradient = 0.02 * torch.randn(rows, n, generator=generator)
    d_nll, u_nll = _curvature(n, 3, 13)
    d_kl, u_kl = _curvature(n, 3, 14)
    codebooks = torch.sort(torch.randn(rows, clusters, generator=generator), 1).values
    labels = torch.randint(clusters, (rows, n), generator=generator)
    q0 = gather_quantized(codebooks, labels)
    result = quantize_rows_method_a_lowrank(
        weight, gradient, d_nll, u_nll, d_kl, u_kl,
        labels, codebooks, MethodAConfig(max_outer_iters=0),
    )
    torch.testing.assert_close(gather_quantized(result.codebooks, result.labels), q0)
    error = q0 - weight
    torch.testing.assert_close(
        result.diagnostics["epsilon_kl"],
        0.5 * lowrank_quadratic_rows(error, d_kl, u_kl),
    )
