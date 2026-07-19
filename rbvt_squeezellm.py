#!/usr/bin/env python3
"""Apply CGC and/or RBVT to SqueezeLLM LUT assignments and repack."""

from __future__ import annotations

import argparse
import gc
import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import torch
from tqdm import tqdm

from env_utils import load_project_dotenv
from any_precision.analyzer import dispatch_model, get_analyzer
from any_precision.quantization.pack import pack


POSTPROCESS_MODES = ("cgc", "rbvt", "cgc_rbvt", "rbvt_cgc")
PIPELINE_STAGES = ("all", "stats", "apply", "pack")


@dataclass
class RBVTStats:
    flips: int = 0
    channels: int = 0
    candidates: int = 0
    boundary_kept: int = 0
    bias_before: float = 0.0
    bias_after: float = 0.0
    objective_before: float = 0.0
    objective_after: float = 0.0
    variance_delta: float = 0.0
    cgc_rows: int = 0

    def add(self, other: "RBVTStats"):
        for field in self.__dataclass_fields__:
            setattr(self, field, getattr(self, field) + getattr(other, field))


class ActivationStatsCollector:
    def __init__(self, analyzer, want_var: bool):
        self.analyzer = analyzer
        self.want_var = want_var
        self.sum: dict[str, torch.Tensor] = {}
        self.sumsq: dict[str, torch.Tensor] = {}
        self.count: dict[str, int] = {}
        self.hooks = []

    def _hook(self, name: str):
        def hook(_module, inp, _out):
            x = inp[0] if isinstance(inp, tuple) else inp
            x = x.detach().reshape(-1, x.shape[-1]).float()
            s = x.sum(dim=0).cpu()
            if name not in self.sum:
                self.sum[name] = s
                self.count[name] = x.shape[0]
                if self.want_var:
                    self.sumsq[name] = (x * x).sum(dim=0).cpu()
            else:
                self.sum[name] += s
                self.count[name] += x.shape[0]
                if self.want_var:
                    self.sumsq[name] += (x * x).sum(dim=0).cpu()
        return hook

    def register(self):
        for layer_idx, layer in enumerate(self.analyzer.get_layers()):
            for module_name, module in self.analyzer.get_modules(layer).items():
                name = f"{layer_idx:02}={module_name}"
                self.hooks.append(module.register_forward_hook(self._hook(name)))

    def remove(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def means_vars(self):
        means = {}
        variances = {}
        for name, total in self.sum.items():
            count = max(1, self.count[name])
            mean = total / count
            means[name] = mean
            if self.want_var and name in self.sumsq:
                ex2 = self.sumsq[name] / count
                variances[name] = (ex2 - mean * mean).clamp(min=0.0)
        return means, variances


def collect_activation_stats(analyzer, tokens: torch.Tensor, n_calib: int, batch_size: int, rbvt_lambda: float):
    model = analyzer.model
    if torch.cuda.device_count() > 1:
        model = dispatch_model(model)
    model = model.bfloat16()
    model.eval()
    if model.device.type != "cuda" and torch.cuda.device_count() == 1:
        model.cuda()

    collector = ActivationStatsCollector(analyzer, want_var=rbvt_lambda > 0.0)
    collector.register()
    try:
        limit = min(n_calib, tokens.shape[0])
        for start in tqdm(range(0, limit, batch_size), desc="Collecting RBVT activation stats"):
            batch = tokens[start:start + batch_size].to(model.device)
            with torch.inference_mode():
                model(input_ids=batch, use_cache=False)
            del batch
            if torch.cuda.is_available() and (start // batch_size + 1) % 16 == 0:
                torch.cuda.empty_cache()
    finally:
        collector.remove()

    model.cpu()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return collector.means_vars()


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


def _dequantize(indices: torch.Tensor, luts: torch.Tensor) -> torch.Tensor:
    # indices: [rows, 1, cols], luts: [rows, 1, levels]
    row_ids = torch.arange(indices.shape[0], device=indices.device).view(-1, 1)
    return luts[:, 0, :][row_ids, indices[:, 0, :].long()]


@torch.no_grad()
def apply_cgc_to_luts(
    W_fp: torch.Tensor,
    indices: torch.Tensor,
    luts: torch.Tensor,
    mu: torch.Tensor,
    sigma_ii: torch.Tensor | None,
    gap_floor: float,
    row_chunk: int,
) -> tuple[torch.Tensor, int]:
    device = W_fp.device
    levels = luts.to(device).float().clone()
    idx = indices.to(device).long()[:, 0, :].clone()
    W_fp = W_fp.to(device).float()
    mu = mu.to(device).float()
    sigma_ii = torch.zeros_like(mu) if sigma_ii is None else sigma_ii.to(device).float()

    out_features, _ = W_fp.shape
    num_levels = levels.shape[-1]
    updated_rows = 0

    if num_levels <= 1:
        return levels, 0

    for r0 in range(0, out_features, row_chunk):
        r1 = min(r0 + row_chunk, out_features)
        rc = r1 - r0

        idx_chunk = idx[r0:r1]
        W_fp_chunk = W_fp[r0:r1]
        codebook = levels[r0:r1, 0, :].clone()
        row_wq = torch.gather(codebook, 1, idx_chunk)
        e = row_wq - W_fp_chunk

        mu_expand = mu.unsqueeze(0).expand(rc, -1)
        sigma_expand = sigma_ii.unsqueeze(0).expand(rc, -1)
        b = (e * mu_expand).sum(dim=1)

        m = torch.zeros((rc, num_levels), device=device, dtype=torch.float32)
        h = torch.zeros((rc, num_levels), device=device, dtype=torch.float32)
        d = torch.zeros((rc, num_levels), device=device, dtype=torch.float32)
        m.scatter_add_(1, idx_chunk, mu_expand)
        h.scatter_add_(1, idx_chunk, sigma_expand * e)
        d.scatter_add_(1, idx_chunk, sigma_expand)

        d = d.clamp_min(gap_floor)
        A = torch.diag_embed(d) + m.unsqueeze(2) * m.unsqueeze(1)
        rhs = b.unsqueeze(1) * m + h
        try:
            delta = -torch.linalg.solve(A, rhs.unsqueeze(-1)).squeeze(-1)
        except RuntimeError:
            delta = -torch.linalg.lstsq(A, rhs.unsqueeze(-1)).solution.squeeze(-1)

        delta[:, 0] = 0.0
        if num_levels > 1:
            delta[:, -1] = 0.0
        if num_levels > 2:
            left_gap = codebook[:, 1:-1] - codebook[:, :-2]
            right_gap = codebook[:, 2:] - codebook[:, 1:-1]
            valid = (left_gap > gap_floor) & (right_gap > gap_floor)
            lower = -0.5 * left_gap
            upper = 0.5 * right_gap
            delta_mid = delta[:, 1:-1]
            delta_mid = torch.minimum(torch.maximum(delta_mid, lower), upper)
            delta[:, 1:-1] = torch.where(valid, delta_mid, torch.zeros_like(delta_mid))

        candidate_codebook = codebook + delta
        candidate_wq = torch.gather(candidate_codebook, 1, idx_chunk)
        candidate_e = candidate_wq - W_fp_chunk
        old_obj = b.square() + (sigma_expand * e.square()).sum(dim=1)
        candidate_b = (candidate_e * mu_expand).sum(dim=1)
        candidate_obj = candidate_b.square() + (sigma_expand * candidate_e.square()).sum(dim=1)
        changed = (delta.abs() > 1e-8).any(dim=1)
        accept = candidate_obj <= (old_obj + 1e-8)

        levels[r0:r1, 0, :] = torch.where(accept.unsqueeze(1), candidate_codebook, codebook)
        updated_rows += int((accept & changed).sum().item())

    return levels, updated_rows


@torch.no_grad()
def apply_rbvt_indices(
    W_fp: torch.Tensor,
    indices: torch.Tensor,
    luts: torch.Tensor,
    mu: torch.Tensor,
    sigma_ii: torch.Tensor | None,
    rbvt_lambda: float,
    rbvt_topk: int,
    row_chunk: int,
    gap_floor: float,
    strict_descent: bool,
) -> tuple[torch.Tensor, RBVTStats]:
    device = W_fp.device
    out_features, in_features = W_fp.shape
    levels = luts.to(device).float()
    idx_full = indices.to(device).long()[:, 0, :].clone()
    Wq_full = _dequantize(indices.to(device), levels).float()
    mu = mu.to(device).float()
    sigma_ii = torch.zeros_like(mu) if sigma_ii is None else sigma_ii.to(device).float()

    stats = RBVTStats(channels=out_features)
    num_levels = levels.shape[-1]
    relax_eps = 1e-12

    for r0 in range(0, out_features, row_chunk):
        r1 = min(r0 + row_chunk, out_features)
        rc = r1 - r0
        Wr = W_fp[r0:r1].float()
        Wq = Wq_full[r0:r1]
        idx = idx_full[r0:r1]
        row_ids = torch.arange(rc, device=device).unsqueeze(1)

        e = Wq - Wr
        e_sign = torch.sign(e)
        b = e @ mu

        left_idx = (idx - 1).clamp(min=0)
        right_idx = (idx + 1).clamp(max=num_levels - 1)
        cur = levels[r0:r1, 0, :][row_ids, idx]
        left = levels[r0:r1, 0, :][row_ids, left_idx]
        right = levels[r0:r1, 0, :][row_ids, right_idx]

        move_down = e_sign > 0
        gap = torch.where(move_down, (cur - left).abs(), (right - cur).abs())
        target_idx = torch.where(move_down, left_idx, right_idx)
        feasible = torch.where(move_down, idx > 0, idx < (num_levels - 1))
        gap_ok = gap > gap_floor

        v = mu.unsqueeze(0) * e_sign * gap
        r = v.abs()
        # Use the raw diagonal variance change after CGC. Since CGC keeps the
        # previous assignments fixed while shifting codewords, the corrected
        # assignments are not guaranteed to remain nearest-neighbor with
        # respect to the updated codebook, so q can legitimately be negative.
        # Negative q is beneficial here: it means the RBVT move reduces both
        # the first-moment error and the diagonal reconstruction term.
        q = sigma_ii.unsqueeze(0) * (gap.square() - 2.0 * gap * e.abs())
        sign_aligned = (b.unsqueeze(1) * v) > 0
        admissible = feasible & gap_ok & sign_aligned & (r > relax_eps)
        rho = q / (r + relax_eps)

        for rr in range(rc):
            T = float(abs(b[rr].item()))
            base_obj = T * T
            stats.bias_before += base_obj
            stats.objective_before += base_obj
            if T <= relax_eps:
                stats.bias_after += base_obj
                stats.objective_after += base_obj
                continue

            cand = torch.nonzero(admissible[rr], as_tuple=False).squeeze(1)
            stats.candidates += int(cand.numel())
            if cand.numel() == 0:
                stats.bias_after += base_obj
                stats.objective_after += base_obj
                continue

            cand_rho = rho[rr, cand]
            if rbvt_topk > 0 and cand.numel() > rbvt_topk:
                _, topk_idx = torch.topk(cand_rho, k=rbvt_topk, largest=False, sorted=False)
                cand = cand[topk_idx]
                cand_rho = cand_rho[topk_idx]
            cand = cand[torch.argsort(cand_rho, descending=False)]

            r_cand = r[rr, cand]
            q_cand = q[rr, cand]
            limit = T if strict_descent else 2.0 * T
            cum_r = torch.cumsum(r_cand, dim=0)
            cum_q = torch.cumsum(q_cand, dim=0)
            zero = torch.zeros(1, device=device, dtype=r_cand.dtype)
            s_prev = torch.cat([zero, cum_r[:-1]], dim=0)
            q_prev = torch.cat([zero, cum_q[:-1]], dim=0)

            upper = ((limit - s_prev) / (r_cand + relax_eps)).clamp(min=0.0, max=1.0)
            gamma_star = (T - s_prev - rbvt_lambda * q_cand / (2.0 * (r_cand + relax_eps))) / (r_cand + relax_eps)
            gamma = torch.minimum(torch.maximum(gamma_star, torch.zeros_like(gamma_star)), upper)
            relaxed_obj = (T - s_prev - gamma * r_cand).square() + rbvt_lambda * (q_prev + gamma * q_cand)
            relaxed_obj = torch.where(upper > 0.0, relaxed_obj, torch.full_like(relaxed_obj, float("inf")))

            best_val, best_pos = relaxed_obj.min(dim=0)
            if float(best_val.item()) >= base_obj:
                stats.bias_after += base_obj
                stats.objective_after += base_obj
                continue

            best_pos_i = int(best_pos.item())
            best_gamma = float(gamma[best_pos_i].item())
            prefix_count = best_pos_i
            prefix_r = float(s_prev[best_pos_i].item())
            prefix_q = float(q_prev[best_pos_i].item())
            drop_obj = (T - prefix_r) ** 2 + rbvt_lambda * prefix_q

            keep_obj = float("inf")
            keep_count = prefix_count
            keep_r = prefix_r
            keep_q = prefix_q
            if best_gamma > 0.0:
                keep_r = float((prefix_r + r_cand[best_pos_i]).item())
                if keep_r <= limit + 1e-8:
                    keep_q = float((prefix_q + q_cand[best_pos_i]).item())
                    keep_obj = (T - keep_r) ** 2 + rbvt_lambda * keep_q
                    keep_count = prefix_count + 1

            if keep_obj < drop_obj:
                chosen_count = keep_count
                stats.boundary_kept += int(best_gamma < 1.0)
                final_r = keep_r
                final_q = keep_q
            else:
                chosen_count = prefix_count
                final_r = prefix_r
                final_q = prefix_q

            if chosen_count > 0:
                chosen = cand[:chosen_count]
                idx_full[r0 + rr, chosen] = target_idx[rr, chosen]
                stats.flips += chosen_count

            stats.bias_after += (T - final_r) ** 2
            stats.objective_after += (T - final_r) ** 2 + rbvt_lambda * final_q
            stats.variance_delta += final_q

    return idx_full.cpu().to(torch.uint8).unsqueeze(1), stats


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


def _resolve_stat_tensor(stats_mapping, layer_idx: int, module_name: str, stats_name: str):
    for candidate in _module_name_candidates(module_name):
        stat_key = f"{layer_idx:02}={candidate}"
        if stat_key in stats_mapping:
            return stats_mapping[stat_key]

    leaf_candidates = {name.split(".")[-1] for name in _module_name_candidates(module_name)}
    suffix_matches = [
        value
        for name, value in stats_mapping.items()
        if name.startswith(f"{layer_idx:02}=") and name.split("=")[-1].split(".")[-1] in leaf_candidates
    ]
    if len(suffix_matches) == 1:
        return suffix_matches[0]

    raise RuntimeError(
        f"Missing {stats_name} for {layer_idx:02}={module_name}. "
        f"Tried aliases: {_module_name_candidates(module_name)}"
    )


def apply_postprocess_to_sqllm_cache(args, analyzer, means, variances):
    src = Path(args.input_quantized_path)
    dst = Path(args.output_quantized_path)
    if args.overwrite and args.layer_range is None and dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)

    if args.mode != "rbvt_cgc":
        for subdir in src.iterdir():
            if subdir.is_dir() and subdir.name.startswith("lut_"):
                shutil.copytree(subdir, dst / subdir.name, dirs_exist_ok=True)
    else:
        (dst / f"lut_{args.bits}").mkdir(parents=True, exist_ok=True)
    if (src / "misc_weights").exists():
        shutil.copytree(src / "misc_weights", dst / "misc_weights", dirs_exist_ok=True)
    (dst / "weights").mkdir(parents=True, exist_ok=True)

    run_cgc = args.mode in {"cgc", "cgc_rbvt", "rbvt_cgc"}
    run_rbvt = args.mode in {"rbvt", "cgc_rbvt", "rbvt_cgc"}

    total = RBVTStats()
    desc = {
        "cgc": "Applying CGC to SqueezeLLM cache",
        "rbvt": "Applying RBVT to SqueezeLLM cache",
        "cgc_rbvt": "Applying CGC+RBVT to SqueezeLLM cache",
        "rbvt_cgc": "Applying RBVT then CGC to SqueezeLLM cache",
    }[args.mode]

    selected_layers = range(analyzer.num_layers)
    if args.layer_range:
        start, end = args.layer_range
        selected_layers = range(start, min(end, analyzer.num_layers))

    if args.mode == "rbvt_cgc":
        for layer_idx in tqdm(selected_layers, desc=f"{desc} [RBVT pass]"):
            output_weight_path = dst / "weights" / f"l{layer_idx}.pt"
            if output_weight_path.exists() and not args.overwrite:
                logging.info("Skipping completed RBVT layer cache: %s", output_weight_path)
                continue

            layer_weights = torch.load(src / "weights" / f"l{layer_idx}.pt", map_location="cpu")
            layer_luts = torch.load(src / f"lut_{args.bits}" / f"l{layer_idx}.pt", map_location="cpu")
            fp_weights = analyzer.get_layer_weights(layer_idx)
            out_layer = {}
            for module_name in layer_weights.keys():
                W_fp = _resolve_layer_mapping_entry(fp_weights, layer_idx, module_name, "FP weights").to(args.device)
                indices = torch.as_tensor(layer_weights[module_name], device=args.device)
                luts = torch.as_tensor(_resolve_layer_mapping_entry(layer_luts, layer_idx, module_name, "layer LUTs"), device=args.device)
                mu = _resolve_stat_tensor(means, layer_idx, module_name, "RBVT activation stats")
                sigma_ii = _resolve_stat_tensor(variances, layer_idx, module_name, "RBVT activation variances") if variances else None

                new_indices, stats = apply_rbvt_indices(
                    W_fp=W_fp,
                    indices=indices,
                    luts=luts,
                    mu=mu,
                    sigma_ii=sigma_ii,
                    rbvt_lambda=args.rbvt_lambda,
                    rbvt_topk=args.rbvt_topk,
                    row_chunk=args.row_chunk,
                    gap_floor=args.gap_floor,
                    strict_descent=not args.allow_overshoot,
                )
                total.add(stats)

                out_layer[module_name] = new_indices.numpy()
                del W_fp, indices, luts, new_indices
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            torch.save(out_layer, output_weight_path)

        for layer_idx in tqdm(selected_layers, desc=f"{desc} [CGC pass]"):
            output_lut_path = dst / f"lut_{args.bits}" / f"l{layer_idx}.pt"
            layer_weights = torch.load(dst / "weights" / f"l{layer_idx}.pt", map_location="cpu")
            layer_luts = torch.load(src / f"lut_{args.bits}" / f"l{layer_idx}.pt", map_location="cpu")
            fp_weights = analyzer.get_layer_weights(layer_idx)
            out_luts = {}
            for module_name in layer_weights.keys():
                W_fp = _resolve_layer_mapping_entry(fp_weights, layer_idx, module_name, "FP weights").to(args.device)
                indices = torch.as_tensor(layer_weights[module_name], device=args.device)
                luts = torch.as_tensor(_resolve_layer_mapping_entry(layer_luts, layer_idx, module_name, "layer LUTs"), device=args.device)
                mu = _resolve_stat_tensor(means, layer_idx, module_name, "RBVT activation stats")
                sigma_ii = _resolve_stat_tensor(variances, layer_idx, module_name, "RBVT activation variances") if variances else None

                luts, cgc_rows = apply_cgc_to_luts(
                    W_fp=W_fp,
                    indices=indices,
                    luts=luts,
                    mu=mu,
                    sigma_ii=sigma_ii,
                    gap_floor=args.gap_floor,
                    row_chunk=args.row_chunk,
                )
                total.cgc_rows += cgc_rows

                out_luts[module_name] = luts.detach().cpu()
                del W_fp, indices, luts
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            torch.save(out_luts, output_lut_path)

        return total

    for layer_idx in tqdm(selected_layers, desc=desc):
        output_weight_path = dst / "weights" / f"l{layer_idx}.pt"
        if output_weight_path.exists() and not args.overwrite:
            logging.info("Skipping completed post-processed layer cache: %s", output_weight_path)
            continue

        layer_weights = torch.load(src / "weights" / f"l{layer_idx}.pt", map_location="cpu")
        layer_luts = torch.load(src / f"lut_{args.bits}" / f"l{layer_idx}.pt", map_location="cpu")
        fp_weights = analyzer.get_layer_weights(layer_idx)
        out_layer = {}
        out_luts = {}
        for module_name in layer_weights.keys():
            W_fp = _resolve_layer_mapping_entry(fp_weights, layer_idx, module_name, "FP weights").to(args.device)
            indices = torch.as_tensor(layer_weights[module_name], device=args.device)
            luts = torch.as_tensor(_resolve_layer_mapping_entry(layer_luts, layer_idx, module_name, "layer LUTs"), device=args.device)
            mu = _resolve_stat_tensor(means, layer_idx, module_name, "RBVT activation stats")
            sigma_ii = _resolve_stat_tensor(variances, layer_idx, module_name, "RBVT activation variances") if variances else None

            if run_cgc:
                luts, cgc_rows = apply_cgc_to_luts(
                    W_fp=W_fp,
                    indices=indices,
                    luts=luts,
                    mu=mu,
                    sigma_ii=sigma_ii,
                    gap_floor=args.gap_floor,
                    row_chunk=args.row_chunk,
                )
                total.cgc_rows += cgc_rows

            if run_rbvt:
                new_indices, stats = apply_rbvt_indices(
                    W_fp=W_fp,
                    indices=indices,
                    luts=luts,
                    mu=mu,
                    sigma_ii=sigma_ii,
                    rbvt_lambda=args.rbvt_lambda,
                    rbvt_topk=args.rbvt_topk,
                    row_chunk=args.row_chunk,
                    gap_floor=args.gap_floor,
                    strict_descent=not args.allow_overshoot,
                )
                total.add(stats)
            else:
                new_indices = indices.detach().cpu().to(torch.uint8)

            out_layer[module_name] = new_indices.numpy()
            out_luts[module_name] = luts.detach().cpu()
            del W_fp, indices, luts, new_indices
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        torch.save(out_layer, output_weight_path)
        torch.save(out_luts, dst / f"lut_{args.bits}" / f"l{layer_idx}.pt")

    return total


def parse_args():
    parser = argparse.ArgumentParser(description="CGC/RBVT post-processing for SqueezeLLM caches.")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--mode", choices=POSTPROCESS_MODES, default="cgc_rbvt")
    parser.add_argument("--stage", choices=PIPELINE_STAGES, default="all")
    parser.add_argument("--bits", type=int, default=3)
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--dataset", default="redpajama")
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--num-examples", type=int, default=1024)
    parser.add_argument("--tokens-path", default="")
    parser.add_argument("--input-quantized-path", default="")
    parser.add_argument("--output-quantized-path", default="")
    parser.add_argument("--output-packed-path", default="")
    parser.add_argument("--stats-path", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-calib", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--rbvt-lambda", type=float, default=1.0)
    parser.add_argument("--rbvt-topk", type=int, default=0)
    parser.add_argument("--row-chunk", type=int, default=1024)
    parser.add_argument("--gap-floor", type=float, default=1e-8)
    parser.add_argument("--allow-overshoot", action="store_true")
    parser.add_argument("--layer-range", type=int, nargs=2, metavar=("START", "END"), default=None,
                        help="Only apply post-processing on layers in [START, END). Useful for sharding large RBVT runs.")
    parser.add_argument("--cpu-count", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--overwrite-stats", action="store_true")
    return parser.parse_args()


def mode_tag(mode: str) -> str:
    return {
        "cgc": "sqllm-cgc",
        "rbvt": "sqllm-rbvt",
        "cgc_rbvt": "sqllm-cgc-rbvt",
        "rbvt_cgc": "sqllm-rbvt-cgc",
    }[mode]


def main():
    load_project_dotenv(Path(__file__).resolve().parent)
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s | %(levelname)s] %(message)s", datefmt="%H:%M:%S")
    args = parse_args()
    model_basename = args.model.rstrip("/").split("/")[-1]
    tag = mode_tag(args.mode)
    run_name = f"{model_basename}-w{args.bits}-{tag}-{args.dataset}_s{args.num_examples}_blk{args.seq_len}_lambda{args.rbvt_lambda:g}"
    args.tokens_path = args.tokens_path or f"{args.cache_dir}/tokens/{model_basename}-{args.dataset}_s{args.num_examples}_blk{args.seq_len}.pt"
    args.input_quantized_path = args.input_quantized_path or f"{args.cache_dir}/quantized/{model_basename}-w{args.bits}_orig{args.bits}-{args.dataset}_s{args.num_examples}_blk{args.seq_len}"
    args.output_quantized_path = args.output_quantized_path or f"{args.cache_dir}/post_sqllm_quantized/{run_name}"
    args.output_packed_path = args.output_packed_path or f"{args.cache_dir}/post_sqllm_packed/anyprec-{tag}-{model_basename}-w{args.bits}-{args.dataset}_s{args.num_examples}_blk{args.seq_len}_lambda{args.rbvt_lambda:g}"
    if not Path(args.input_quantized_path).exists():
        raise FileNotFoundError(f"Missing SqueezeLLM quantized cache: {args.input_quantized_path}")
    if (
        args.stage in {"all", "pack"}
        and Path(args.output_packed_path).exists()
        and any(Path(args.output_packed_path).iterdir())
        and not args.overwrite
    ):
        logging.info("Packed output already exists; use --overwrite to rebuild: %s", args.output_packed_path)
        return

    logging.info("Loading model/analyzer: %s", args.model)
    analyzer = get_analyzer(args.model, include_tokenizer=True)

    if args.stage == "pack":
        analyzer.drop_original_weights()
        if Path(args.output_packed_path).exists() and args.overwrite:
            shutil.rmtree(args.output_packed_path)
        pack(
            analyzer=analyzer,
            lut_path=args.output_quantized_path,
            output_model_path=args.output_packed_path,
            seed_precision=args.bits,
            parent_precision=args.bits,
            cpu_count=args.cpu_count,
        )
        logging.info("Post-processed SqueezeLLM packed model saved to %s", args.output_packed_path)
        return

    if not Path(args.tokens_path).exists():
        raise FileNotFoundError(f"Missing calibration tokens: {args.tokens_path}")

    tokens = normalize_tokens(torch.load(args.tokens_path, map_location="cpu"), args.seq_len)
    logging.info("Loaded calibration tokens: shape=%s", tuple(tokens.shape))
    means, variances = collect_activation_stats(
        analyzer=analyzer,
        tokens=tokens,
        n_calib=args.n_calib,
        batch_size=args.batch_size,
        rbvt_lambda=args.rbvt_lambda,
    )
    logging.info("Activation stats computed fresh for this run; no stats cache will be reused or saved.")
    logging.info("Activation stats ready: means=%d variances=%d", len(means), len(variances))

    if args.stage == "stats":
        logging.info("Stage=stats complete. Skipping LUT post-process and pack.")
        return

    totals = apply_postprocess_to_sqllm_cache(args, analyzer, means, variances)
    logging.info(
        "Post-process summary | mode=%s flips=%d candidates=%d boundary_kept=%d cgc_rows=%d bias %.6e -> %.6e objective %.6e -> %.6e var_delta=%.6e",
        args.mode,
        totals.flips,
        totals.candidates,
        totals.boundary_kept,
        totals.cgc_rows,
        totals.bias_before,
        totals.bias_after,
        totals.objective_before,
        totals.objective_after,
        totals.variance_delta,
    )

    if args.layer_range is not None:
        logging.info(
            "Applied only layer range [%d, %d); skipping pack because cache is partial.",
            args.layer_range[0],
            args.layer_range[1],
        )
        return

    analyzer.drop_original_weights()
    if Path(args.output_packed_path).exists() and args.overwrite:
        shutil.rmtree(args.output_packed_path)
    pack(
        analyzer=analyzer,
        lut_path=args.output_quantized_path,
        output_model_path=args.output_packed_path,
        seed_precision=args.bits,
        parent_precision=args.bits,
        cpu_count=args.cpu_count,
    )
    logging.info("Post-processed SqueezeLLM packed model saved to %s", args.output_packed_path)


if __name__ == "__main__":
    main()
