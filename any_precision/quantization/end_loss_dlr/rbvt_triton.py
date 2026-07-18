from __future__ import annotations

import torch

from .rbvt_cuda_debug import rbvt_coordinate_descent


def rbvt_triton(
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
    return rbvt_coordinate_descent(
        weight=weight,
        gradient=gradient,
        diagonal=diagonal,
        lowrank=lowrank,
        codebook=codebook,
        labels=labels,
        beta=beta,
        eps=eps,
        preserve_nonempty_clusters=preserve_nonempty_clusters,
        reverse=reverse,
    )
