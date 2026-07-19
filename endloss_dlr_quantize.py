#!/usr/bin/env python3
"""From-scratch EndLoss_DLR scalar quantization.

This path does not consume an existing SqueezeLLM cache. It collects end-loss
statistics from the original FP model, runs the DLR solver's own initialization
and alternating updates, then writes the standard AnyPrecision cache format:

    weights/l{layer}.pt
    lut_{bits}/l{layer}.pt

The existing pack/eval code can then be reused unchanged.
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from tqdm import tqdm

try:
    from env_utils import load_project_dotenv
except ModuleNotFoundError:
    def load_project_dotenv(*_args, **_kwargs):
        return None
from any_precision.analyzer import dispatch_model, get_analyzer
from any_precision.quantization.endloss_dlr import DLRConfig
from any_precision.quantization.endloss_dlr_batched import quantize_rows_dlr_batched
from any_precision.quantization.endloss_dlr_fast_nll import collect_nll_gradients
from any_precision.quantization.endloss_dlr_fast_fisher import collect_fisher_curvature
from any_precision.quantization.pack import pack


PIPELINE_STAGES = ("all", "tokens", "stats", "quantize", "pack")


@dataclass
class EndLossDLRQuantStats:
    rows_total: int = 0
    rows_quantized: int = 0
    rows_fallback: int = 0
    objective_sum: float = 0.0

    def add(self, other: "EndLossDLRQuantStats") -> None:
        for field in self.__dataclass_fields__:
            setattr(self, field, getattr(self, field) + getattr(other, field))


def _tag_value(value) -> str:
    return str(value).replace("-", "m").replace(".", "p").replace("+", "")

def build_fast_stats_config(
    rank: int,
    oversampling: int,
    n_calib: int,
    batch_size: int,
    device: str,
    fisher_probes: int,
    gradient_num_examples: int | None,
    stats_layer_chunk_size: int,
    num_output_groups: int,
    damping_ratio: float,
) -> dict:
    return {
        "stats_method": "fast_weight_gradient_fisher_v2",
        "rank": int(rank),
        "oversampling": int(oversampling),
        "n_calib": int(n_calib),
        "batch_size": int(batch_size),
        "device": str(device),
        "fisher_probes": int(fisher_probes),
        "gradient_num_examples": None if gradient_num_examples is None else int(gradient_num_examples),
        "stats_layer_chunk_size": int(stats_layer_chunk_size),
        "num_output_groups": int(num_output_groups),
        "damping_ratio": float(damping_ratio),
    }


def resolve_legacy_fast_stats_cache(
    args,
    analyzer,
    data_tag: str,
    default_stats_path: str,
) -> str:
    if args.overwrite_stats:
        return default_stats_path

    default_path = Path(default_stats_path)
    default_expected = [default_path / f"l{i}.pt" for i in range(analyzer.num_layers)]
    if default_expected and all(path.exists() for path in default_expected):
        return default_stats_path

    legacy_tag = (
        f"{data_tag}_r{args.rank}_os{args.oversampling}"
        f"_ncalib{args.n_calib}_fprobe{args.fisher_probes}_gex{args.gradient_num_examples or args.n_calib}"
        f"_lchunk{args.stats_layer_chunk_size}_og{args.num_output_groups}"
        f"_damp{_tag_value(args.damping_ratio)}_seed{args.random_state}"
    )
    legacy_path = Path(args.cache_dir) / "endloss_dlr_stats" / legacy_tag
    expected = [legacy_path / f"l{i}.pt" for i in range(analyzer.num_layers)]
    if not expected or not all(path.exists() for path in expected):
        return default_stats_path

    stats_config = build_fast_stats_config(
        rank=args.rank,
        oversampling=args.oversampling,
        n_calib=args.n_calib,
        batch_size=args.batch_size,
        device=args.device,
        fisher_probes=args.fisher_probes,
        gradient_num_examples=args.gradient_num_examples,
        stats_layer_chunk_size=args.stats_layer_chunk_size,
        num_output_groups=args.num_output_groups,
        damping_ratio=args.damping_ratio,
    )
    config_path = legacy_path / "_config.pt"
    if config_path.exists():
        try:
            if torch.load(config_path, map_location="cpu") != stats_config:
                return default_stats_path
        except Exception:
            return default_stats_path
    else:
        torch.save(stats_config, config_path)

    logging.info("Using legacy EndLoss_DLR stats cache: %s", legacy_path)
    return str(legacy_path)

def normalize_tokens(tokens, seq_len: int) -> torch.Tensor:
    if isinstance(tokens, torch.Tensor):
        if tokens.ndim == 2:
            return tokens.long()
        if tokens.ndim == 3 and tokens.shape[1] == 1:
            return tokens[:, 0, :].long()
        raise ValueError(f"Expected token tensor with shape [n, seq] or [n, 1, seq], got {tuple(tokens.shape)}")

    if isinstance(tokens, (list, tuple)):
        normalized = []
        for item in tokens:
            if not isinstance(item, torch.Tensor):
                raise TypeError(f"Expected tensor token item, got {type(item).__name__}")
            item = item.detach().cpu()
            if item.ndim == 2 and item.shape[0] == 1:
                item = item[0]
            if item.ndim != 1:
                raise ValueError(f"Expected token item with shape [seq] or [1, seq], got {tuple(item.shape)}")
            if item.numel() != seq_len:
                raise ValueError(f"Expected token length {seq_len}, got {item.numel()}")
            normalized.append(item.long())
        if not normalized:
            raise ValueError("Token list is empty")
        return torch.stack(normalized, dim=0)

    raise TypeError(f"Unsupported token cache type: {type(tokens).__name__}")



def _row_group_ranges(num_rows: int, num_groups: int) -> list[tuple[int, int]]:
    num_groups = max(1, min(int(num_groups), int(num_rows)))
    base = int(num_rows) // num_groups
    rem = int(num_rows) % num_groups
    ranges = []
    start = 0
    for group_idx in range(num_groups):
        size = base + (1 if group_idx < rem else 0)
        end = start + size
        ranges.append((start, end))
        start = end
    return ranges

def _module_name_candidates(module_name: str) -> list[str]:
    candidates = [module_name]
    if module_name.startswith("self_attn."):
        candidates.append(module_name.replace("self_attn.", "linear_attn.", 1))
    elif module_name.startswith("linear_attn."):
        candidates.append(module_name.replace("linear_attn.", "self_attn.", 1))

    leaf_name = module_name.split(".")[-1]
    if leaf_name == "o_proj":
        candidates.append(module_name[: -len("o_proj")] + "out_proj")
    elif leaf_name == "out_proj":
        candidates.append(module_name[: -len("out_proj")] + "o_proj")

    return list(dict.fromkeys(candidates))


def _resolve_layer_mapping_entry(mapping, layer_idx: int, module_name: str, mapping_name: str):
    for candidate in _module_name_candidates(module_name):
        if candidate in mapping:
            return mapping[candidate]

    if isinstance(mapping, dict):
        leaf_candidates = {name.split(".")[-1] for name in _module_name_candidates(module_name)}
        suffix_matches = [value for name, value in mapping.items() if name.split(".")[-1] in leaf_candidates]
        if len(suffix_matches) == 1:
            return suffix_matches[0]

    raise RuntimeError(
        f"Missing {mapping_name} for layer={layer_idx:02}, module={module_name}. "
        f"Tried aliases: {_module_name_candidates(module_name)}"
    )


def _batch_iter(tokens: torch.Tensor, batch_size: int, n_calib: int):
    limit = min(n_calib, tokens.shape[0])
    for start in range(0, limit, batch_size):
        yield tokens[start : start + batch_size]


def _prediction_slice(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 3 and tensor.shape[1] > 1:
        return tensor[:, :-1, :]
    return tensor



def _iter_activation_chunks(tensor: torch.Tensor, chunk_size: int):
    flat = tensor.reshape(-1, tensor.shape[-1])
    if flat.numel() == 0:
        return
    chunk_size = max(1, int(chunk_size))
    for start in range(0, flat.shape[0], chunk_size):
        yield flat[start : start + chunk_size].float()
def _prepare_model(analyzer):
    model = analyzer.model
    if torch.cuda.device_count() > 1:
        model = dispatch_model(model)
    model = model.bfloat16()
    model.eval()
    if model.device.type != "cuda" and torch.cuda.device_count() == 1:
        model.cuda()

    modules_by_layer = []
    target_weights = set()
    for layer in analyzer.get_layers():
        modules = analyzer.get_modules(layer)
        modules_by_layer.append(modules)
        for module in modules.values():
            target_weights.add(module.weight)

    original_requires_grad = {}
    for param in model.parameters():
        original_requires_grad[id(param)] = param.requires_grad
        param.requires_grad_(param in target_weights)

    return model, modules_by_layer, original_requires_grad


def _restore_model(model, original_requires_grad) -> None:
    for param in model.parameters():
        param.requires_grad_(original_requires_grad[id(param)])
    model.cpu()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _init_stat_buffers(modules_by_layer, rank: int, oversampling: int, seed: int):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    sketch_cols = rank + oversampling
    all_stats = []
    for modules in modules_by_layer:
        layer_stats = {}
        for module_name, module in modules.items():
            out_features, in_features = module.weight.shape
            layer_stats[module_name] = {
                "g_sum": torch.zeros(out_features, in_features, dtype=torch.float32),
                "diag_sum": torch.zeros(in_features, dtype=torch.float32),
                "a_count": 0,
                "omega": torch.randn(in_features, sketch_cols, generator=generator, dtype=torch.float32),
                "y_sum": torch.zeros(in_features, sketch_cols, dtype=torch.float32),
                "Q": None,
                "b_sum": None,
                "alpha_sum": torch.zeros(out_features, dtype=torch.float32),
                "alpha_count": 0,
            }
        all_stats.append(layer_stats)
    return all_stats


def _register_a_pass1_hooks(modules_by_layer, all_stats, stats_chunk_size: int):
    hooks = []

    def make_hook(stats):
        def hook(_module, inp, _out):
            x = inp[0] if isinstance(inp, tuple) else inp
            x_sliced = _prediction_slice(x.detach())
            omega = stats["omega"].to(x_sliced.device)
            for X in _iter_activation_chunks(x_sliced, stats_chunk_size):
                stats["diag_sum"] += X.square().sum(dim=0).cpu()
                stats["y_sum"] += (X.transpose(0, 1) @ (X @ omega)).cpu()
                stats["a_count"] += X.shape[0]

        return hook

    for layer_idx, modules in enumerate(modules_by_layer):
        for module_name, module in modules.items():
            hooks.append(module.register_forward_hook(make_hook(all_stats[layer_idx][module_name])))
    return hooks

def _register_a_pass2_hooks(modules_by_layer, all_stats, stats_chunk_size: int):
    hooks = []

    def make_hook(stats):
        def hook(_module, inp, _out):
            x = inp[0] if isinstance(inp, tuple) else inp
            x_sliced = _prediction_slice(x.detach())
            Q = stats["Q"].to(x_sliced.device)
            for X in _iter_activation_chunks(x_sliced, stats_chunk_size):
                Z = X @ Q
                stats["b_sum"] += (Z.transpose(0, 1) @ Z).cpu()

        return hook

    for layer_idx, modules in enumerate(modules_by_layer):
        for module_name, module in modules.items():
            hooks.append(module.register_forward_hook(make_hook(all_stats[layer_idx][module_name])))
    return hooks

def _register_alpha_hooks(modules_by_layer, all_stats, stats_chunk_size: int):
    hooks = []

    def make_hook(stats):
        def forward_hook(_module, _inp, out):
            out.retain_grad()

            def grad_hook(grad):
                delta_sliced = _prediction_slice(grad.detach())
                token_count = delta_sliced.reshape(-1, delta_sliced.shape[-1]).shape[0]
                if token_count == 0:
                    return
                for delta in _iter_activation_chunks(delta_sliced, stats_chunk_size):
                    stats["alpha_sum"] += delta.square().sum(dim=0).cpu()
                stats["alpha_count"] += token_count

            out.register_hook(grad_hook)

        return forward_hook

    for layer_idx, modules in enumerate(modules_by_layer):
        for module_name, module in modules.items():
            hooks.append(module.register_forward_hook(make_hook(all_stats[layer_idx][module_name])))
    return hooks

def _remove_hooks(hooks) -> None:
    for hook in hooks:
        hook.remove()


def _finish_a_pass1(all_stats, rank: int, oversampling: int) -> None:
    sketch_cols = rank + oversampling
    for layer_stats in all_stats:
        for stats in layer_stats.values():
            count = max(1, stats["a_count"])
            Y = stats["y_sum"] / count
            if torch.count_nonzero(Y) == 0:
                Q = torch.zeros(Y.shape[0], min(sketch_cols, Y.shape[0]), dtype=torch.float32)
            else:
                Q, _ = torch.linalg.qr(Y, mode="reduced")
            stats["Q"] = Q[:, : min(Q.shape[1], sketch_cols)].contiguous()
            stats["b_sum"] = torch.zeros(stats["Q"].shape[1], stats["Q"].shape[1], dtype=torch.float32)


def _finalize_stats(all_stats, total_pred_tokens: int, rank: int, d_min_scale: float):
    finalized = []
    for layer_stats in all_stats:
        layer_dict = {}
        for module_name, stats in layer_stats.items():
            a_count = max(1, stats["a_count"])
            diag_A = stats["diag_sum"] / a_count
            g = stats["g_sum"] / max(1, total_pred_tokens)
            Q = stats["Q"]
            B = stats["b_sum"] / a_count

            if Q.shape[1] == 0:
                U_A = torch.zeros(diag_A.shape[0], 0, dtype=torch.float32)
            else:
                eigvals, eigvecs = torch.linalg.eigh(B)
                order = torch.argsort(eigvals, descending=True)
                eigvals = eigvals[order][:rank].clamp_min(0.0)
                eigvecs = eigvecs[:, order][:, :rank]
                U_A = Q @ eigvecs
                if U_A.numel() > 0:
                    U_A = U_A * eigvals.sqrt().unsqueeze(0)

            lowrank_diag = U_A.square().sum(dim=-1)
            d_floor = d_min_scale * diag_A.mean().clamp_min(1e-12)
            d_A = (diag_A - lowrank_diag).clamp_min(d_floor)

            alpha = stats["alpha_sum"] / max(1, stats["alpha_count"])
            positive_alpha = alpha[alpha > 0]
            alpha_scale = positive_alpha.median() if positive_alpha.numel() else torch.tensor(1.0)
            alpha_min = float(1e-6 * alpha_scale)

            layer_dict[module_name] = {
                "g": g.cpu(),
                "diag_A": diag_A.cpu(),
                "d_A": d_A.cpu(),
                "U_A": U_A.cpu(),
                "alpha": alpha.cpu(),
                "alpha_min": alpha_min,
                "a_count": a_count,
                "alpha_count": stats["alpha_count"],
                "total_pred_tokens": total_pred_tokens,
            }
        finalized.append(layer_dict)
    return finalized


def collect_endloss_dlr_stats_fast(
    analyzer,
    tokens: torch.Tensor,
    output_folder: str,
    rank: int,
    oversampling: int,
    n_calib: int,
    batch_size: int,
    device: str,
    fisher_probes: int,
    gradient_num_examples: int | None,
    stats_layer_chunk_size: int,
    num_output_groups: int,
    damping_ratio: float,
    overwrite: bool,
) -> None:
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    expected = [output_path / f"l{i}.pt" for i in range(analyzer.num_layers)]
    stats_config = build_fast_stats_config(
        rank=rank,
        oversampling=oversampling,
        n_calib=n_calib,
        batch_size=batch_size,
        device=device,
        fisher_probes=fisher_probes,
        gradient_num_examples=gradient_num_examples,
        stats_layer_chunk_size=stats_layer_chunk_size,
        num_output_groups=num_output_groups,
        damping_ratio=damping_ratio,
    )
    config_path = output_path / "_config.pt"
    config_matches = False
    if config_path.exists():
        try:
            config_matches = torch.load(config_path, map_location="cpu") == stats_config
        except Exception:
            config_matches = False
    if not overwrite and expected and all(path.exists() for path in expected) and config_matches:
        logging.info("Cached fast EndLoss_DLR stats found in %s", output_folder)
        return
    if not overwrite and expected and all(path.exists() for path in expected) and not config_matches:
        logging.warning("Existing EndLoss_DLR stats cache config mismatch or missing metadata; recomputing: %s", output_folder)
    if overwrite or not config_matches:
        for path in expected:
            if path.exists():
                path.unlink()
        if config_path.exists():
            config_path.unlink()

    calib_tokens = tokens[: min(n_calib, tokens.shape[0])]
    grad_count = min(calib_tokens.shape[0], gradient_num_examples or calib_tokens.shape[0])
    grad_tokens = calib_tokens[:grad_count]

    logging.info(
        "Collecting fast EndLoss_DLR stats | grad_examples=%d fisher_probes=%d layer_chunk=%d output_groups=%d",
        grad_tokens.shape[0],
        min(fisher_probes, calib_tokens.shape[0]),
        stats_layer_chunk_size,
        num_output_groups,
    )
    nll_gradients = collect_nll_gradients(
        analyzer=analyzer,
        tokens=grad_tokens,
        batch_size=batch_size,
        device=device,
        layer_chunk_size=stats_layer_chunk_size,
    )
    fisher_config = SimpleNamespace(
        device=device,
        fisher_probes=fisher_probes,
        rank=rank,
        oversample=oversampling,
        num_output_groups=num_output_groups,
        damping_ratio=damping_ratio,
        eps=1e-12,
        calibration_batch_size=batch_size,
        stats_layer_chunk_size=stats_layer_chunk_size,
    )
    fisher_curvature = collect_fisher_curvature(analyzer, calib_tokens, fisher_config)

    for layer_idx in range(analyzer.num_layers):
        layer_dict = {}
        for module_name, grad in nll_gradients[layer_idx].items():
            curvature = fisher_curvature[layer_idx][module_name]
            layer_dict[module_name] = {
                "g": grad.cpu(),
                "group_d": curvature["group_d"].cpu(),
                "group_U": curvature["group_U"].cpu(),
                "num_output_groups": int(curvature["group_d"].shape[0]),
                "gradient_examples": int(grad_tokens.shape[0]),
                "fisher_probes": int(min(fisher_probes, calib_tokens.shape[0])),
            }
        torch.save(layer_dict, output_path / f"l{layer_idx}.pt")
    torch.save(stats_config, config_path)

def quantize_module_from_scratch(
    W_fp: torch.Tensor,
    module_stats: dict,
    bits: int,
    config: DLRConfig,
    row_batch_size: int,
    layer_idx: int | None = None,
    module_name: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, EndLossDLRQuantStats]:
    device = W_fp.device
    W_fp = W_fp.float()
    g = module_stats["g"].to(device).float()

    out_features, in_features = W_fp.shape
    K = 2 ** bits
    labels_out = torch.empty(out_features, 1, in_features, dtype=torch.uint8, device="cpu")
    lut_out = torch.empty(out_features, 1, K, dtype=torch.float16, device="cpu")
    stats = EndLossDLRQuantStats(rows_total=out_features)
    row_batch_size = max(1, int(row_batch_size))

    if "group_d" in module_stats:
        group_d = module_stats["group_d"].to(device).float()
        group_U = module_stats["group_U"].to(device).float()
        row_ranges = _row_group_ranges(out_features, int(module_stats.get("num_output_groups", group_d.shape[0])))
        group_iter = [(group_idx, start, end) for group_idx, (start, end) in enumerate(row_ranges)]
    else:
        d_A = module_stats["d_A"].to(device).float()
        U_A = module_stats["U_A"].to(device).float()
        alpha = module_stats["alpha"].to(device).float()
        alpha_min = float(module_stats["alpha_min"])
        alpha = alpha.clamp_min(alpha_min)
        group_iter = [(None, 0, out_features)]

    for group_idx, group_start, group_end in group_iter:
        for start in range(group_start, group_end, row_batch_size):
            end = min(start + row_batch_size, group_end)
            if group_idx is None:
                d_cur = d_A
                U_cur = U_A
                alpha_cur = alpha[start:end]
            else:
                d_cur = group_d[group_idx]
                U_cur = group_U[group_idx]
                alpha_cur = torch.ones(end - start, device=device, dtype=torch.float32)
            try:
                result = quantize_rows_dlr_batched(
                    w=W_fp[start:end],
                    g=g[start:end],
                    d_A=d_cur,
                    U_A=U_cur,
                    alpha=alpha_cur,
                    K=K,
                    config=config,
                )
            except RuntimeError as exc:
                retry_config = replace(config, lambda_safety=max(config.lambda_safety * 1.05, 1.05))
                location = f"layer={layer_idx}, module={module_name}, group={group_idx}, rows=[{start},{end})"
                logging.warning(
                    "EndLoss_DLR batched solve failed at %s with lambda_safety=%s; retrying with lambda_safety=%s. Error: %s",
                    location,
                    config.lambda_safety,
                    retry_config.lambda_safety,
                    exc,
                )
                try:
                    result = quantize_rows_dlr_batched(
                        w=W_fp[start:end],
                        g=g[start:end],
                        d_A=d_cur,
                        U_A=U_cur,
                        alpha=alpha_cur,
                        K=K,
                        config=retry_config,
                    )
                except RuntimeError as retry_exc:
                    raise RuntimeError(
                        f"EndLoss_DLR failed after retry at {location}; refusing to fall back to another quantizer."
                    ) from retry_exc

            if not torch.isfinite(result.codebooks).all() or not torch.isfinite(result.losses).all():
                location = f"layer={layer_idx}, module={module_name}, group={group_idx}, rows=[{start},{end})"
                max_abs = float(result.codebooks.detach().abs().max().item()) if result.codebooks.numel() else 0.0
                raise RuntimeError(f"EndLoss_DLR produced non-finite initialization/update at {location}; max_abs_codebook={max_abs:.6e}")

            labels_out[start:end, 0] = result.labels.detach().cpu().to(torch.uint8)
            lut_out[start:end, 0] = result.codebooks.detach().cpu().to(torch.float16)
            stats.rows_quantized += end - start
            stats.rows_fallback += result.fallback_rows
            stats.objective_sum += float(result.losses.sum().item())

    return labels_out, lut_out, stats

def quantize_endloss_dlr_cache(
    analyzer,
    stats_folder: str,
    output_folder: str,
    bits: int,
    config: DLRConfig,
    device: str,
    overwrite: bool,
    layer_range: tuple[int, int] | None = None,
    row_batch_size: int = 64,
) -> EndLossDLRQuantStats:
    output_path = Path(output_folder)
    if overwrite and output_path.exists() and layer_range is None:
        shutil.rmtree(output_path)
    (output_path / "weights").mkdir(parents=True, exist_ok=True)
    (output_path / f"lut_{bits}").mkdir(parents=True, exist_ok=True)

    selected_layers = range(analyzer.num_layers)
    if layer_range is not None:
        selected_layers = range(layer_range[0], min(layer_range[1], analyzer.num_layers))

    totals = EndLossDLRQuantStats()
    for layer_idx in tqdm(selected_layers, desc="EndLoss_DLR quantizing layers"):
        weights_path = output_path / "weights" / f"l{layer_idx}.pt"
        lut_path = output_path / f"lut_{bits}" / f"l{layer_idx}.pt"
        if weights_path.exists() and lut_path.exists() and not overwrite:
            logging.info("Skipping completed EndLoss_DLR layer cache: %s", weights_path)
            continue

        layer_stats = torch.load(Path(stats_folder) / f"l{layer_idx}.pt", map_location="cpu")
        fp_weights = analyzer.get_layer_weights(layer_idx)
        out_weights = {}
        out_luts = {}
        for module_name, module_stats in layer_stats.items():
            W_fp = _resolve_layer_mapping_entry(fp_weights, layer_idx, module_name, "FP weights").to(device)
            labels, luts, module_totals = quantize_module_from_scratch(
                W_fp=W_fp,
                module_stats=module_stats,
                bits=bits,
                config=config,
                row_batch_size=row_batch_size,
                layer_idx=layer_idx,
                module_name=module_name,
            )
            totals.add(module_totals)
            out_weights[module_name] = labels.numpy()
            out_luts[module_name] = luts
            del W_fp, labels, luts
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        torch.save(out_weights, weights_path)
        torch.save(out_luts, lut_path)

    return totals


def parse_args():
    parser = argparse.ArgumentParser(description="From-scratch EndLoss_DLR scalar quantization.")
    parser.add_argument("model", nargs="?", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--stage", choices=PIPELINE_STAGES, default="all")
    parser.add_argument("--bits", type=int, default=3)
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--dataset", default="redpajama")
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--num-examples", type=int, default=1024)
    parser.add_argument("--redpajama-source", choices=["cache", "raw"], default="cache")
    parser.add_argument("--redpajama-dataset-repo", default=None)
    parser.add_argument("--tokens-path", default="")
    parser.add_argument("--stats-path", default="")
    parser.add_argument("--quantized-path", default="")
    parser.add_argument("--output-packed-path", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-calib", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--stats-chunk-size", type=int, default=1024)
    parser.add_argument("--fisher-probes", type=int, default=16)
    parser.add_argument("--gradient-num-examples", type=int, default=None)
    parser.add_argument("--stats-layer-chunk-size", type=int, default=8)
    parser.add_argument("--num-output-groups", type=int, default=8)
    parser.add_argument("--damping-ratio", type=float, default=1e-4)
    parser.add_argument("--row-batch-size", type=int, default=64)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--oversampling", type=int, default=4)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--max-outer-iters", type=int, default=8)
    parser.add_argument("--rel-tol", type=float, default=1e-7)
    parser.add_argument("--lambda-safety", type=float, default=1.01)
    parser.add_argument("--stats-d-min-scale", type=float, default=1e-6)
    parser.add_argument("--solver-d-min", type=float, default=1e-8)
    parser.add_argument("--tie-tol", type=float, default=0.0)
    parser.add_argument("--layer-range", type=int, nargs=2, metavar=("START", "END"), default=None)
    parser.add_argument("--cpu-count", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--overwrite-tokens", action="store_true")
    parser.add_argument("--overwrite-stats", action="store_true")
    parser.add_argument("--overwrite-quantize", action="store_true")
    parser.add_argument("--overwrite-pack", action="store_true")
    return parser.parse_args()


def main():
    load_project_dotenv(Path(__file__).resolve().parent)
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s | %(levelname)s] %(message)s", datefmt="%H:%M:%S")
    args = parse_args()

    model_name = args.model.rstrip("/").split("/")[-1]
    data_tag = f"{model_name}-{args.dataset}_s{args.num_examples}_blk{args.seq_len}"
    stats_tag = (
        f"fastwgf_v2_{data_tag}_r{args.rank}_os{args.oversampling}"
        f"_ncalib{args.n_calib}_bs{args.batch_size}"
        f"_fprobe{args.fisher_probes}_gex{args.gradient_num_examples or args.n_calib}"
        f"_lchunk{args.stats_layer_chunk_size}_og{args.num_output_groups}"
        f"_damp{_tag_value(args.damping_ratio)}_seed{args.random_state}"
    )
    solver_tag = (
        f"{stats_tag}_beta{_tag_value(args.beta)}_iters{args.max_outer_iters}"
        f"_rtol{_tag_value(args.rel_tol)}_lambda{_tag_value(args.lambda_safety)}"
        f"_sdmin{_tag_value(args.solver_d_min)}"
    )
    run_tag = f"{model_name}-w{args.bits}-endloss-dlr-{solver_tag}"
    args.tokens_path = args.tokens_path or f"{args.cache_dir}/tokens/{data_tag}.pt"
    explicit_stats_path = bool(args.stats_path)
    default_stats_path = f"{args.cache_dir}/endloss_dlr_stats/{stats_tag}"
    args.stats_path = args.stats_path or default_stats_path
    args.quantized_path = args.quantized_path or f"{args.cache_dir}/endloss_dlr_quantized/{run_tag}"
    args.output_packed_path = args.output_packed_path or f"{args.cache_dir}/endloss_dlr_packed/anyprec-{run_tag}"

    logging.info("Loading model/analyzer: %s", args.model)
    analyzer = get_analyzer(args.model, include_tokenizer=True)
    if not explicit_stats_path:
        args.stats_path = resolve_legacy_fast_stats_cache(args, analyzer, data_tag, default_stats_path)

    if args.dataset == "redpajama" and args.redpajama_dataset_repo is not None:
        os.environ["REDPAJAMA_DATASET_REPO"] = args.redpajama_dataset_repo

    
    if args.stage == "pack":
        analyzer.drop_original_weights()
        output_packed = Path(args.output_packed_path)
        if output_packed.exists() and any(output_packed.iterdir()):
            if args.overwrite_pack:
                shutil.rmtree(output_packed)
            else:
                logging.info("Packed output already exists; use --overwrite-pack to rebuild: %s", output_packed)
                return
        pack(
            analyzer=analyzer,
            lut_path=args.quantized_path,
            output_model_path=args.output_packed_path,
            seed_precision=args.bits,
            parent_precision=args.bits,
            cpu_count=args.cpu_count,
        )
        logging.info("EndLoss_DLR packed model saved to %s", args.output_packed_path)
        return

    if args.overwrite_tokens:
        tokens_path = Path(args.tokens_path)
        if tokens_path.exists():
            tokens_path.unlink()
    from any_precision.quantization.datautils import get_tokens

    tokens = get_tokens(
        args.dataset,
        "train",
        analyzer.tokenizer,
        args.seq_len,
        args.num_examples,
        args.tokens_path,
        args.random_state,
        redpajama_source=args.redpajama_source,
    )
    tokens = normalize_tokens(tokens, args.seq_len)
    logging.info("Tokens ready: shape=%s", tuple(tokens.shape))
    if args.stage == "tokens":
        return

    collect_endloss_dlr_stats_fast(
        analyzer=analyzer,
        tokens=tokens,
        output_folder=args.stats_path,
        rank=args.rank,
        oversampling=args.oversampling,
        n_calib=args.n_calib,
        batch_size=args.batch_size,
        device=args.device,
        fisher_probes=args.fisher_probes,
        gradient_num_examples=args.gradient_num_examples,
        stats_layer_chunk_size=args.stats_layer_chunk_size,
        num_output_groups=args.num_output_groups,
        damping_ratio=args.damping_ratio,
        overwrite=args.overwrite_stats,
    )
    if args.stage == "stats":
        return

    config = DLRConfig(
        beta=args.beta,
        rank=args.rank,
        max_outer_iters=args.max_outer_iters,
        rel_tol=args.rel_tol,
        lambda_safety=args.lambda_safety,
        d_min=args.solver_d_min,
        tie_tol=args.tie_tol,
    )
    if args.max_outer_iters == 0:
        logging.info("EndLoss_DLR mode: initialization-only (max_outer_iters=0)")
    else:
        logging.info("EndLoss_DLR mode: full MM solver (max_outer_iters=%d)", args.max_outer_iters)
    totals = quantize_endloss_dlr_cache(
        analyzer=analyzer,
        stats_folder=args.stats_path,
        output_folder=args.quantized_path,
        bits=args.bits,
        config=config,
        device=args.device,
        overwrite=args.overwrite_quantize,
        layer_range=tuple(args.layer_range) if args.layer_range else None,
        row_batch_size=args.row_batch_size,
    )
    if args.max_outer_iters == 0:
        logging.info("Initial DLR objective | rows=%d objective_sum=%.6e", totals.rows_quantized, totals.objective_sum)
    logging.info(
        "EndLoss_DLR quantize summary | rows_total=%d rows_quantized=%d rows_fallback=%d objective_sum=%.6e",
        totals.rows_total,
        totals.rows_quantized,
        totals.rows_fallback,
        totals.objective_sum,
    )
    if args.stage == "quantize" or args.layer_range is not None:
        return

    analyzer.drop_original_weights()
    output_packed = Path(args.output_packed_path)
    if output_packed.exists() and any(output_packed.iterdir()):
        if args.overwrite_pack:
            shutil.rmtree(output_packed)
        else:
            logging.info("Packed output already exists; use --overwrite-pack to rebuild: %s", output_packed)
            return

    pack(
        analyzer=analyzer,
        lut_path=args.quantized_path,
        output_model_path=args.output_packed_path,
        seed_precision=args.bits,
        parent_precision=args.bits,
        cpu_count=args.cpu_count,
    )
    logging.info("EndLoss_DLR packed model saved to %s", args.output_packed_path)


if __name__ == "__main__":
    main()










