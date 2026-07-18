from __future__ import annotations

import logging
import os
import shutil
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field

import torch
from tqdm.auto import tqdm

from ..datautils import get_tokens
from ..pack import pack
from ...analyzer import get_analyzer
from .config import EndLossDLRConfig
from .fisher_probe_cuda import collect_fisher_curvature
from .nll_gradient_cuda import collect_nll_gradients
from .row_objective_cuda import quantize_group_dlr_batched
from .serialization import packed_model_output_path, quantized_cache_path, save_layer_artifacts, save_metadata


@dataclass
class EndLossDLRModuleStats:
    nll_gradient: torch.Tensor
    group_d: torch.Tensor
    group_U: torch.Tensor
    row_to_group: torch.Tensor


@dataclass
class EndLossDLRModelStats:
    modules: dict[int, dict[str, EndLossDLRModuleStats]] = field(default_factory=dict)

    def to(self, device: str) -> "EndLossDLRModelStats":
        moved = EndLossDLRModelStats()
        for layer_idx, module_dict in self.modules.items():
            moved.modules[layer_idx] = {}
            for module_name, stats in module_dict.items():
                moved.modules[layer_idx][module_name] = EndLossDLRModuleStats(
                    nll_gradient=stats.nll_gradient.to(device),
                    group_d=stats.group_d.to(device),
                    group_U=stats.group_U.to(device),
                    row_to_group=stats.row_to_group.to(device),
                )
        return moved


def _make_row_to_group(out_features: int, num_output_groups: int, device: torch.device) -> torch.Tensor:
    groups = min(max(1, num_output_groups), out_features)
    chunk = (out_features + groups - 1) // groups
    row_to_group = torch.empty(out_features, device=device, dtype=torch.long)
    group_id = 0
    for start in range(0, out_features, chunk):
        row_to_group[start:start + chunk] = group_id
        group_id += 1
    return row_to_group


def collect_end_loss_statistics(
    model,
    calibration_loader,
    config: EndLossDLRConfig,
    analyzer=None,
    nll_cache_path: str | None = None,
) -> EndLossDLRModelStats:
    del model
    config.validate()
    if analyzer is None:
        raise ValueError("analyzer is required so the collector can stay compatible with the sample project structure")
    tokens = calibration_loader.to(config.device)
    if config.gradient_num_examples is not None:
        grad_count = min(tokens.shape[0], config.gradient_num_examples)
        grad_tokens = tokens[:grad_count]
        logging.info("Collecting NLL gradients from %d/%d calibration examples", grad_count, tokens.shape[0])
    else:
        grad_tokens = tokens
        logging.info("Collecting NLL gradients from all %d calibration examples", tokens.shape[0])

    if nll_cache_path is not None and os.path.exists(nll_cache_path):
        logging.info("Loading cached temporary NLL gradients from %s", nll_cache_path)
        nll_gradients = torch.load(nll_cache_path, map_location="cpu", weights_only=False)
    else:
        nll_gradients = collect_nll_gradients(
            analyzer,
            grad_tokens,
            config.calibration_batch_size,
            config.device,
            config.stats_layer_chunk_size,
        )
        if nll_cache_path is not None:
            os.makedirs(os.path.dirname(nll_cache_path), exist_ok=True)
            torch.save(nll_gradients, nll_cache_path)
            logging.info("Saved temporary NLL gradients to %s", nll_cache_path)
    fisher_curvature = collect_fisher_curvature(analyzer, tokens, config)
    stats = EndLossDLRModelStats()
    for layer_idx, layer in enumerate(analyzer.get_layers()):
        stats.modules[layer_idx] = {}
        for module_name, module in analyzer.get_modules(layer).items():
            row_to_group = _make_row_to_group(module.weight.shape[0], config.num_output_groups, torch.device(config.device))
            stats.modules[layer_idx][module_name] = EndLossDLRModuleStats(
                nll_gradient=nll_gradients[layer_idx][module_name].cpu(),
                group_d=fisher_curvature[layer_idx][module_name]["group_d"].cpu(),
                group_U=fisher_curvature[layer_idx][module_name]["group_U"].cpu(),
                row_to_group=row_to_group.cpu(),
            )
    return stats


def _quantize_row_chunk(
    weights: torch.Tensor,
    gradients: torch.Tensor,
    diagonal: torch.Tensor,
    lowrank: torch.Tensor,
    config: EndLossDLRConfig,
    log_prefix: str | None = None,
):
    batch = weights.shape[0]
    d_batch = diagonal.unsqueeze(0).expand(batch, -1)
    if lowrank.numel() == 0 or lowrank.shape[-1] == 0:
        U_batch = lowrank.new_zeros((batch, weights.shape[1], 0), dtype=torch.float32)
    else:
        U_batch = lowrank.unsqueeze(0).expand(batch, -1, -1)
    return quantize_group_dlr_batched(
        w=weights,
        g=gradients,
        d=d_batch,
        U=U_batch,
        K=config.num_levels,
        beta=config.beta,
        max_outer_iters=config.max_outer_iters,
        rel_tol=config.rel_tol,
        lambda_safety=config.lambda_safety,
        tie_tol=config.tie_tol,
        eps=config.eps,
        log_prefix=log_prefix,
    )


def quantize_model(model, stats: EndLossDLRModelStats, config: EndLossDLRConfig, analyzer=None):
    del model
    config.validate()
    if analyzer is None:
        raise ValueError("analyzer is required so model mutation follows the sample project structure")

    stats = stats.to(config.device)
    metadata = {
        "beta": config.beta,
        "rank": config.rank,
        "num_output_groups": config.num_output_groups,
        "max_outer_iters": config.max_outer_iters,
        "rel_tol": config.rel_tol,
        "lambda_safety": config.lambda_safety,
        "surrogate_losses": defaultdict(dict),
        "timings": {},
    }
    total_start = time.perf_counter()
    saved_layers = []

    layers = analyzer.get_layers()
    for layer_idx, layer in enumerate(tqdm(
        layers,
        desc="Quantizing layers",
        unit="layer",
        ascii=True,
        leave=False,
        dynamic_ncols=False,
        ncols=100,
        mininterval=5.0,
        maxinterval=30.0,
        file=sys.stdout,
    )):
        layer_start = time.perf_counter()
        layer_codebooks = {}
        layer_labels = {}
        modules = analyzer.get_modules(layer)
        for module_name, module in modules.items():
            module_stats = stats.modules[layer_idx][module_name]
            weight = module.weight.data.to(config.device).float()
            row_codebooks = []
            row_labels = []
            for group_id in range(module_stats.group_d.shape[0]):
                rows = torch.nonzero(module_stats.row_to_group == group_id, as_tuple=False).squeeze(1)
                if rows.numel() == 0:
                    continue
                diagonal = module_stats.group_d[group_id]
                lowrank = module_stats.group_U[group_id]
                for start in range(0, rows.numel(), config.row_batch_size):
                    row_chunk = rows[start:start + config.row_batch_size]
                    weights_chunk = weight.index_select(0, row_chunk)
                    grads_chunk = module_stats.nll_gradient.index_select(0, row_chunk)
                    chunk_start_row = int(row_chunk[0].item())
                    chunk_end_row = int(row_chunk[-1].item())
                    codebooks_chunk, labels_chunk, losses_chunk = _quantize_row_chunk(
                        weights=weights_chunk,
                        gradients=grads_chunk,
                        diagonal=diagonal,
                        lowrank=lowrank,
                        config=config,
                        log_prefix=f"[layer={layer_idx} module={module_name} group={group_id} rows={chunk_start_row}-{chunk_end_row}]",
                    )
                    quantized_chunk = torch.gather(codebooks_chunk, 1, labels_chunk.long())
                    weight.index_copy_(0, row_chunk, quantized_chunk.to(weight.dtype))
                    for local_idx, global_row in enumerate(row_chunk.tolist()):
                        row_codebooks.append((global_row, codebooks_chunk[local_idx]))
                        row_labels.append((global_row, labels_chunk[local_idx]))
                        metadata["surrogate_losses"][f"layer_{layer_idx}:{module_name}"][str(global_row)] = [float(x) for x in losses_chunk[local_idx].tolist()]
            row_codebooks.sort(key=lambda x: x[0])
            row_labels.sort(key=lambda x: x[0])
            module.weight.data.copy_(weight.to(module.weight.dtype))
            layer_codebooks[module_name] = torch.stack([item[1] for item in row_codebooks], dim=0).unsqueeze(1).cpu()
            layer_labels[module_name] = torch.stack([item[1] for item in row_labels], dim=0).unsqueeze(1).cpu()
            del row_codebooks, row_labels, weight
        metadata["timings"][f"layer_{layer_idx}"] = time.perf_counter() - layer_start
        saved_layers.append((layer_idx, layer_codebooks, layer_labels))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    metadata["timings"]["total_quantize"] = time.perf_counter() - total_start
    return saved_layers, metadata


def hybrid_end_loss_quantize(
    model,
    bits: int = 3,
    yaml_path: str | None = None,
    cache_dir: str = "./cache",
    dataset: str = "redpajama",
    seq_len: int = 4096,
    num_examples: int = 128,
    redpajama_source: str = "cache",
    overwrite_quantize: bool = False,
    overwrite_pack: bool = False,
    random_state: int | None = None,
    calibration_batch_size: int = 1,
    fisher_probes: int = 16,
    beta: float = 0.5,
    rank: int = 4,
    num_output_groups: int = 8,
    row_batch_size: int = 128,
    max_outer_iters: int = 8,
    rel_tol: float = 1e-7,
    lambda_safety: float = 1.01,
    tie_tol: float = 0.0,
):
    model_string = model if isinstance(model, str) else model.name_or_path
    model_name = model_string.split("/")[-1]
    analyzer = get_analyzer(model, yaml_path=yaml_path, include_tokenizer=True)
    config = EndLossDLRConfig(
        bits=bits,
        beta=beta,
        rank=rank,
        num_output_groups=num_output_groups,
        row_batch_size=row_batch_size,
        calibration_batch_size=calibration_batch_size,
        fisher_probes=fisher_probes,
        max_outer_iters=max_outer_iters,
        rel_tol=rel_tol,
        lambda_safety=lambda_safety,
        tie_tol=tie_tol,
        cache_dir=cache_dir,
        dataset=dataset,
        seq_len=seq_len,
        num_examples=num_examples,
    )
    config.validate()

    tokens_cache_path = os.path.join(cache_dir, "tokens", f"{model_name}-{dataset}_s{num_examples}_blk{seq_len}.pt")
    tokens = get_tokens(
        dataset,
        "train",
        analyzer.tokenizer,
        seq_len,
        num_examples,
        tokens_cache_path,
        random_state,
        redpajama_source=redpajama_source,
    )
    if isinstance(tokens, list):
        tokens = torch.stack([item.long().cpu() for item in tokens], dim=0)
    elif isinstance(tokens, torch.Tensor) and tokens.ndim == 3 and tokens.shape[1] == 1:
        tokens = tokens[:, 0, :]
    elif not isinstance(tokens, torch.Tensor):
        raise TypeError(f"Unsupported token cache type: {type(tokens).__name__}")

    output_dir = quantized_cache_path(cache_dir, model_name, bits, dataset, num_examples, seq_len)
    packed_dir = packed_model_output_path(cache_dir, model_name, bits, dataset, num_examples, seq_len)
    if overwrite_quantize and os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    stats = collect_end_loss_statistics(model=analyzer.model, calibration_loader=tokens, config=config, analyzer=analyzer)
    saved_layers, metadata = quantize_model(model=analyzer.model, stats=stats, config=config, analyzer=analyzer)
    metadata.update({
        "quantized_cache": output_dir,
        "packed_output": packed_dir,
    })
    for layer_idx, layer_codebooks, layer_labels in saved_layers:
        save_layer_artifacts(output_dir, layer_idx, layer_codebooks, layer_labels)
    save_metadata(output_dir, metadata)

    if overwrite_pack and os.path.isdir(packed_dir):
        shutil.rmtree(packed_dir)
    try:
        pack(
            analyzer=analyzer,
            lut_path=output_dir,
            output_model_path=packed_dir,
            seed_precision=bits,
            parent_precision=bits,
            cpu_count=cpu_count,
            dns=False,
        )
    except Exception as exc:
        logging.warning(f"Packing failed; quantized cache is still available at {output_dir}: {exc}")
    return output_dir, metadata



