from __future__ import annotations

from collections import defaultdict

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .nll_gradient_cuda import (
    _disable_checkpointing_for_stats,
    _enable_checkpointing_for_stats,
    _iter_layer_chunks,
    _restore_float_dtypes,
    _snapshot_float_dtypes,
)


def _row_groups(out_features: int, requested_groups: int, device: str) -> list[torch.Tensor]:
    groups = min(max(1, requested_groups), out_features)
    chunk = (out_features + groups - 1) // groups
    return [torch.arange(start, min(start + chunk, out_features), device=device) for start in range(0, out_features, chunk)]


def _make_group_omega(in_features: int, rank: int, oversample: int, device: str, seed: int) -> torch.Tensor:
    sketch_rank = max(1, rank + oversample)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return torch.randn((in_features, sketch_rank), generator=generator, device=device, dtype=torch.float32)


def _fit_group_dlr_from_streaming_sketch(diag_total: torch.Tensor, Y: torch.Tensor, rank: int, damping_ratio: float, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    del damping_ratio, eps
    diag_total = diag_total.float()
    if rank <= 0:
        return diag_total, diag_total.new_zeros((diag_total.numel(), 0))

    if not torch.any(Y.abs() > 0):
        return diag_total, diag_total.new_zeros((diag_total.numel(), 0))

    Q, _ = torch.linalg.qr(Y, mode="reduced")
    if Q.numel() == 0 or Q.shape[1] == 0:
        return diag_total, diag_total.new_zeros((diag_total.numel(), 0))

    return diag_total, Q


def collect_fisher_curvature(analyzer, tokens: torch.Tensor, config) -> dict[int, dict[str, dict[str, torch.Tensor]]]:
    model = analyzer.model
    model.to(config.device)
    model.eval()
    original_param_dtypes, original_buffer_dtypes = _snapshot_float_dtypes(model)
    model = model.bfloat16()
    _enable_checkpointing_for_stats(model)

    layers = analyzer.get_layers()
    fisher_stats: dict[int, dict[str, dict[str, torch.Tensor]]] = defaultdict(dict)
    original_requires_grad = {id(param): param.requires_grad for param in model.parameters()}

    max_samples = min(tokens.shape[0], config.fisher_probes)
    sketch_rank = max(1, config.rank + config.oversample)

    for chunk_start, layer_chunk in _iter_layer_chunks(layers, config.stats_layer_chunk_size):
        target_weights = set()
        group_meta: dict[int, dict[str, dict[str, list[torch.Tensor] | torch.Tensor | int]]] = defaultdict(dict)

        for local_idx, layer in enumerate(layer_chunk):
            layer_idx = chunk_start + local_idx
            for module_name, module in analyzer.get_modules(layer).items():
                target_weights.add(module.weight)
                groups = _row_groups(module.weight.shape[0], config.num_output_groups, config.device)
                module_meta = {
                    "groups": groups,
                    "omega": [],
                    "diag_total": [],
                    "Y": [],
                }
                for group_id, _rows in enumerate(groups):
                    seed = 17_000_003 * (layer_idx + 1) + 1_000_003 * (group_id + 1) + 101 * (len(module_name) + 1)
                    omega = _make_group_omega(module.weight.shape[1], config.rank, config.oversample, config.device, seed)
                    module_meta["omega"].append(omega)
                    module_meta["diag_total"].append(torch.zeros(module.weight.shape[1], dtype=torch.float32))
                    module_meta["Y"].append(torch.zeros((module.weight.shape[1], sketch_rank), dtype=torch.float32))
                group_meta[layer_idx][module_name] = module_meta

        for param in model.parameters():
            param.requires_grad_(param in target_weights)

        # Pass 1: accumulate diagonal and covariance sketch Y = C @ Omega using all rowwise probes.
        model.zero_grad(set_to_none=True)
        for start in tqdm(
            range(0, max_samples, config.calibration_batch_size),
            desc=f"Collecting Fisher sketch L{chunk_start}-{chunk_start + len(layer_chunk) - 1}",
            dynamic_ncols=True,
            mininterval=5.0,
            maxinterval=30.0,
            ascii=True,
            leave=True,
        ):
            batch = tokens[start:start + config.calibration_batch_size].to(config.device)
            logits = model(input_ids=batch, use_cache=False).logits[:, :-1, :].float()
            probs = logits.softmax(dim=-1)
            pseudo = torch.multinomial(probs.reshape(-1, probs.shape[-1]), num_samples=1).reshape(probs.shape[:-1])
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), pseudo.reshape(-1), reduction="mean")
            loss.backward()

            num_predictions = logits.shape[0] * logits.shape[1]
            batch_scale = max(1, num_predictions) ** 0.5
            for local_idx, layer in enumerate(layer_chunk):
                layer_idx = chunk_start + local_idx
                for module_name, module in analyzer.get_modules(layer).items():
                    grad_probe = module.weight.grad.detach().float() * batch_scale
                    meta = group_meta[layer_idx][module_name]
                    for group_id, rows in enumerate(meta["groups"]):
                        P = grad_probe.index_select(0, rows) / (max(1, rows.numel()) ** 0.5)
                        meta["diag_total"][group_id].add_(P.square().sum(dim=0).cpu())
                        omega = meta["omega"][group_id]
                        Y_update = P.transpose(0, 1).matmul(P.matmul(omega))
                        meta["Y"][group_id].add_(Y_update.cpu())
            model.zero_grad(set_to_none=True)
            del logits, probs, pseudo, loss, batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        num_probe_batches = max(1, (max_samples + config.calibration_batch_size - 1) // config.calibration_batch_size)

        # Normalize streaming accumulators so they match the scale of the original probe_matrix / sqrt(num_batches).
        for local_idx, layer in enumerate(layer_chunk):
            layer_idx = chunk_start + local_idx
            for module_name, module in analyzer.get_modules(layer).items():
                meta = group_meta[layer_idx][module_name]
                for group_id, _rows in enumerate(meta["groups"]):
                    meta["diag_total"][group_id].div_(num_probe_batches)
                    meta["Y"][group_id].div_(num_probe_batches)

        # Build per-group bases Q from the first-pass sketch.
        for local_idx, layer in enumerate(layer_chunk):
            layer_idx = chunk_start + local_idx
            for module_name, module in analyzer.get_modules(layer).items():
                meta = group_meta[layer_idx][module_name]
                meta["Q"] = []
                meta["small_cov"] = []
                for group_id, _rows in enumerate(meta["groups"]):
                    diag_total = meta["diag_total"][group_id].to(config.device, non_blocking=True)
                    Y = meta["Y"][group_id].to(config.device, non_blocking=True)
                    _diag_total, Q = _fit_group_dlr_from_streaming_sketch(
                        diag_total=diag_total,
                        Y=Y,
                        rank=config.rank,
                        damping_ratio=config.damping_ratio,
                        eps=config.eps,
                    )
                    meta["diag_total"][group_id] = _diag_total.cpu()
                    meta["Q"].append(Q.cpu())
                    meta["small_cov"].append(torch.zeros((Q.shape[1], Q.shape[1]), dtype=torch.float32))
                    del diag_total, Y, Q, _diag_total
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        # Pass 2: accumulate exact covariance in the sketched subspace, still using full rowwise probes.
        model.zero_grad(set_to_none=True)
        for start in tqdm(
            range(0, max_samples, config.calibration_batch_size),
            desc=f"Refining Fisher subspace L{chunk_start}-{chunk_start + len(layer_chunk) - 1}",
            dynamic_ncols=True,
            mininterval=5.0,
            maxinterval=30.0,
            ascii=True,
            leave=True,
        ):
            batch = tokens[start:start + config.calibration_batch_size].to(config.device)
            logits = model(input_ids=batch, use_cache=False).logits[:, :-1, :].float()
            probs = logits.softmax(dim=-1)
            pseudo = torch.multinomial(probs.reshape(-1, probs.shape[-1]), num_samples=1).reshape(probs.shape[:-1])
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), pseudo.reshape(-1), reduction="mean")
            loss.backward()

            num_predictions = logits.shape[0] * logits.shape[1]
            batch_scale = max(1, num_predictions) ** 0.5
            for local_idx, layer in enumerate(layer_chunk):
                layer_idx = chunk_start + local_idx
                for module_name, module in analyzer.get_modules(layer).items():
                    grad_probe = module.weight.grad.detach().float() * batch_scale
                    meta = group_meta[layer_idx][module_name]
                    for group_id, rows in enumerate(meta["groups"]):
                        P = grad_probe.index_select(0, rows) / (max(1, rows.numel()) ** 0.5)
                        Q = meta["Q"][group_id].to(config.device, non_blocking=True)
                        Z = P.matmul(Q)
                        meta["small_cov"][group_id].add_(Z.transpose(0, 1).matmul(Z).cpu())
                        del Q, Z
            model.zero_grad(set_to_none=True)
            del logits, probs, pseudo, loss, batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        for local_idx, layer in enumerate(layer_chunk):
            layer_idx = chunk_start + local_idx
            for module_name, module in analyzer.get_modules(layer).items():
                meta = group_meta[layer_idx][module_name]
                for group_id, _rows in enumerate(meta["groups"]):
                    meta["small_cov"][group_id].div_(num_probe_batches)

        # Finalize D and U for each module/group.
        for local_idx, layer in enumerate(layer_chunk):
            layer_idx = chunk_start + local_idx
            for module_name, module in analyzer.get_modules(layer).items():
                meta = group_meta[layer_idx][module_name]
                diagonal_list = []
                lowrank_list = []
                for group_id, _rows in enumerate(meta["groups"]):
                    diag_total = meta["diag_total"][group_id].to(config.device, non_blocking=True)
                    Q = meta["Q"][group_id].to(config.device, non_blocking=True)
                    if config.rank <= 0 or Q.shape[1] == 0:
                        damping = config.damping_ratio * diag_total.mean().clamp_min(config.eps)
                        diagonal = diag_total + damping
                        lowrank = torch.zeros((diag_total.numel(), 0), device=config.device, dtype=torch.float32)
                    else:
                        small_cov = meta["small_cov"][group_id].to(config.device, non_blocking=True)
                        eigvals, eigvecs = torch.linalg.eigh(small_cov)
                        order = torch.argsort(eigvals, descending=True)
                        eigvals = eigvals[order]
                        eigvecs = eigvecs[:, order]
                        kept = min(config.rank, eigvals.numel())
                        if kept == 0:
                            lowrank = torch.zeros((diag_total.numel(), 0), device=config.device, dtype=torch.float32)
                        else:
                            kept_vals = eigvals[:kept].clamp_min(0.0)
                            kept_vecs = eigvecs[:, :kept]
                            lowrank = Q.matmul(kept_vecs * kept_vals.sqrt().unsqueeze(0))
                        residual_diag = (diag_total - lowrank.square().sum(dim=1)).clamp_min(0.0)
                        damping = config.damping_ratio * diag_total.mean().clamp_min(config.eps)
                        diagonal = residual_diag + damping
                        del small_cov, eigvals, eigvecs, order
                    diagonal_list.append(diagonal.cpu())
                    lowrank_list.append(lowrank.cpu())
                    del diag_total, Q, diagonal, lowrank
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                fisher_stats[layer_idx][module_name] = {
                    "group_d": torch.stack(diagonal_list, dim=0),
                    "group_U": torch.stack(lowrank_list, dim=0) if lowrank_list and lowrank_list[0].numel() else torch.zeros(
                        (len(diagonal_list), module.weight.shape[1], 0), dtype=torch.float32
                    ),
                }

        model.zero_grad(set_to_none=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for param in model.parameters():
        param.requires_grad_(original_requires_grad[id(param)])

    _restore_float_dtypes(model, original_param_dtypes, original_buffer_dtypes)
    _disable_checkpointing_for_stats(model)
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return fisher_stats



