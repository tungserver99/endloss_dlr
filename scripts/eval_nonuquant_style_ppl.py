#!/usr/bin/env python3
"""RBVT-squeeze style sliding-window perplexity evaluation for Squeeze caches."""

from __future__ import annotations

import argparse
import gc
import json
import pickle
import sys
import warnings
from pathlib import Path

import torch
import torch.nn as nn
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from any_precision.analyzer import get_analyzer
from env_utils import load_project_dotenv


def _hf_device_map(device: str):
    return {"": device} if device != "auto" else "auto"


def _canonical_dataset_name(name: str) -> str:
    lowered = name.lower()
    if lowered in {"wikitext2", "wikitext-2", "wiki"}:
        return "WikiText-2"
    if lowered in {"c4", "c4_new"}:
        return "C4"
    raise ValueError(f"Unsupported dataset: {name}")


def _resolve_quantized_cache_path(path_str: str) -> Path:
    path = Path(path_str)
    if (path / "weights").exists():
        return path

    parent = path.parent
    name = path.name
    candidates = [path]

    if "_lambda" in name:
        prefix, lambda_value = name.rsplit("_lambda", 1)
        try:
            lambda_float = float(lambda_value)
            lambda_compact = f"{lambda_float:g}"
            candidates.append(parent / f"{prefix}_lambda{lambda_compact}")
            candidates.append(parent / f"{prefix}_lambda{lambda_float}")
        except ValueError:
            pass

    for candidate in candidates:
        if (candidate / "weights").exists():
            return candidate

    return path


def _load_tokenizer(model_path: str, token: str | None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=True,
            token=token,
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _preferred_eval_dtype(model_path: str) -> str:
    lowered = model_path.lower()
    if "qwen" in lowered or "gemma" in lowered:
        return "bfloat16"
    return "float16"


def _find_linear_modules(model) -> dict[str, nn.Linear]:
    return {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
    }


def _resolve_linear_module(linear_modules: dict[str, nn.Linear], layer_idx: int, module_name: str) -> nn.Linear:
    candidates = [module_name]
    if module_name.startswith("self_attn."):
        candidates.append(module_name.replace("self_attn.", "linear_attn.", 1))
    elif module_name.startswith("linear_attn."):
        candidates.append(module_name.replace("linear_attn.", "self_attn.", 1))

    leaf_name = module_name.split(".")[-1]
    if leaf_name == "out_proj":
        candidates.append(module_name[: -len("out_proj")] + "o_proj")
    elif leaf_name == "o_proj":
        candidates.append(module_name[: -len("o_proj")] + "out_proj")

    matches = []
    for candidate in dict.fromkeys(candidates):
        suffix = f".{layer_idx}.{candidate}"
        matches.extend(module for name, module in linear_modules.items() if name.endswith(suffix))

    if len(matches) != 1:
        alias_leaf_matches = []
        for alias_leaf_name in dict.fromkeys([leaf_name, "o_proj" if leaf_name == "out_proj" else leaf_name, "out_proj" if leaf_name == "o_proj" else leaf_name]):
            alias_leaf_matches.extend(
                module for name, module in linear_modules.items()
                if name.endswith(f".{layer_idx}.{alias_leaf_name}") or name.endswith(f".{layer_idx}.linear_attn.{alias_leaf_name}") or name.endswith(f".{layer_idx}.self_attn.{alias_leaf_name}")
            )
        matches = list(dict.fromkeys(matches + alias_leaf_matches))

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one Linear module matching layer={layer_idx}, module={module_name}; "
            f"found {len(matches)}"
        )
    return matches[0]


def _module_name_candidates(module_name: str) -> list[str]:
    candidates = [module_name]
    if module_name.startswith("self_attn."):
        candidates.append(module_name.replace("self_attn.", "linear_attn.", 1))
    elif module_name.startswith("linear_attn."):
        candidates.append(module_name.replace("linear_attn.", "self_attn.", 1))

    leaf_name = module_name.split(".")[-1]
    if leaf_name == "out_proj":
        candidates.append(module_name[: -len("out_proj")] + "o_proj")
    elif leaf_name == "o_proj":
        candidates.append(module_name[: -len("o_proj")] + "out_proj")

    return list(dict.fromkeys(candidates))


def _resolve_layer_linear_module(analyzer, layer_idx: int, module_name: str) -> nn.Linear:
    layer = analyzer.get_layers()[layer_idx]
    layer_module_paths = analyzer.get_layer_module_paths(layer_idx)

    actual_path = None
    for candidate in _module_name_candidates(module_name):
        if candidate in layer_module_paths:
            actual_path = layer_module_paths[candidate]
            break

    if actual_path is None:
        leaf_candidates = {name.split(".")[-1] for name in _module_name_candidates(module_name)}
        suffix_matches = [path for name, path in layer_module_paths.items() if name.split(".")[-1] in leaf_candidates]
        if len(suffix_matches) == 1:
            actual_path = suffix_matches[0]

    if actual_path is None:
        raise RuntimeError(
            f"Could not resolve layer={layer_idx}, module={module_name} from analyzer paths {list(layer_module_paths.keys())[:12]}"
        )

    module = layer
    for attrib_name in actual_path.split("."):
        module = getattr(module, attrib_name)
    if not isinstance(module, nn.Linear):
        raise RuntimeError(f"Resolved module is not nn.Linear: layer={layer_idx}, module={module_name}, path={actual_path}")
    return module


def _dequantize_module(indices, lut, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    idx = torch.as_tensor(indices, device=device, dtype=torch.long)
    levels = torch.as_tensor(lut, device=device, dtype=dtype)
    rows, groups, _ = idx.shape
    row_ids = torch.arange(rows, device=device).view(-1, 1, 1)
    group_ids = torch.arange(groups, device=device).view(1, -1, 1)
    return levels[row_ids, group_ids, idx].reshape(rows, -1)


def _nan_inf_summary(tensor: torch.Tensor) -> str:
    finite_mask = torch.isfinite(tensor)
    nan_count = torch.isnan(tensor).sum().item()
    inf_count = torch.isinf(tensor).sum().item()
    finite_count = finite_mask.sum().item()
    if finite_count > 0:
        finite_values = tensor[finite_mask]
        min_value = finite_values.min().item()
        max_value = finite_values.max().item()
        return (
            f"shape={tuple(tensor.shape)} dtype={tensor.dtype} "
            f"nan={nan_count} inf={inf_count} finite={finite_count} "
            f"min={min_value:.6g} max={max_value:.6g}"
        )
    return (
        f"shape={tuple(tensor.shape)} dtype={tensor.dtype} "
        f"nan={nan_count} inf={inf_count} finite=0"
    )


def _report_nonfinite(name: str, tensor: torch.Tensor):
    if not torch.isfinite(tensor).all():
        print(f"[nonfinite] {name}: {_nan_inf_summary(tensor)}")


@torch.no_grad()
def _load_sqllm_quantized_model(
    base_model_path: str,
    quantized_path: str,
    device: str,
    dtype: torch.dtype,
    token: str | None,
):
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=dtype,
        device_map=_hf_device_map(device),
        trust_remote_code=True,
        token=token,
    )
    model.eval()

    quantized_dir = Path(quantized_path)
    weight_dir = quantized_dir / "weights"
    lut_dirs = sorted(
        [path for path in quantized_dir.iterdir() if path.is_dir() and path.name.startswith("lut_")],
        key=lambda p: int(p.name.split("_", 1)[1]),
    )
    if not weight_dir.exists() or not lut_dirs:
        raise FileNotFoundError(f"{quantized_path} is not a valid Squeeze quantized cache")

    lut_dir = lut_dirs[-1]
    analyzer = get_analyzer(model)
    weight_files = sorted(weight_dir.glob("l*.pt"), key=lambda p: int(p.stem[1:]))

    for weight_file in tqdm(weight_files, desc="Materializing quantized weights", unit="layer"):
        layer_idx = int(weight_file.stem[1:])
        layer_weights = torch.load(weight_file, map_location="cpu", weights_only=False)
        layer_luts = torch.load(lut_dir / weight_file.name, map_location="cpu", weights_only=False)

        for module_name, indices in layer_weights.items():
            target_module = _resolve_layer_linear_module(analyzer, layer_idx, module_name)
            lut_tensor = layer_luts[module_name]
            _report_nonfinite(f"lut layer={layer_idx} module={module_name}", lut_tensor)
            dequantized = _dequantize_module(
                indices=indices,
                lut=lut_tensor,
                device=target_module.weight.device,
                dtype=target_module.weight.dtype,
            )
            _report_nonfinite(f"dequantized layer={layer_idx} module={module_name}", dequantized)
            target_module.weight.data.copy_(dequantized.to(target_module.weight.dtype))
            _report_nonfinite(f"model_weight layer={layer_idx} module={module_name}", target_module.weight.data)
            del dequantized

        del layer_weights, layer_luts
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return model


def _load_wikitext2(cache_dir: Path, seed: int, token: str | None):
    cache_file = cache_dir / f"wikitext2_test_seed{seed}.pkl"
    if cache_file.exists():
        with open(cache_file, "rb") as handle:
            return pickle.load(handle)

    dataset = load_dataset(
        "Salesforce/wikitext",
        "wikitext-2-raw-v1",
        split="test",
        token=token,
    )
    full_text = "\n".join([x for x in dataset["text"] if x])
    result = [full_text]
    with open(cache_file, "wb") as handle:
        pickle.dump(result, handle)
    return result


def _load_c4(cache_dir: Path, seed: int, n_samples: int, token: str | None):
    cache_file = cache_dir / f"c4_validation_n{n_samples}_seed{seed}.pkl"
    if cache_file.exists():
        with open(cache_file, "rb") as handle:
            return pickle.load(handle)

    dataset = load_dataset(
        "allenai/c4",
        "en",
        split="validation",
        streaming=True,
        token=token,
    )
    texts = []
    for item in tqdm(dataset, total=n_samples, desc="Collecting C4"):
        if len(texts) >= n_samples:
            break
        text = item["text"].strip()
        if len(text) > 500:
            texts.append(text)

    result = ["\n\n".join(texts)]
    with open(cache_file, "wb") as handle:
        pickle.dump(result, handle)
    return result


@torch.no_grad()
def evaluate_sliding_window(model, tokenizer, texts, device: str, max_length: int, stride: int, limit_tokens: int | None):
    model.eval()
    nlls = []
    total_tokens = 0

    for text in texts:
        input_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids
        if tokenizer.bos_token_id is not None:
            if input_ids.shape[1] == 0 or input_ids[0, 0].item() != tokenizer.bos_token_id:
                bos = torch.tensor([[tokenizer.bos_token_id]], device=input_ids.device)
                input_ids = torch.cat([bos, input_ids], dim=1)

        if limit_tokens is not None and input_ids.size(1) > limit_tokens:
            input_ids = input_ids[:, :limit_tokens]

        input_ids = input_ids.to(device)
        seq_len = input_ids.size(1)
        if seq_len < 2:
            continue

        prev_end_loc = 0
        window_range = list(range(0, seq_len, stride))
        pbar = tqdm(window_range, desc=f"Windows ({seq_len:,} toks)", unit="win", leave=False)

        for begin_loc in pbar:
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc

            input_chunk = input_ids[:, begin_loc:end_loc]
            target_chunk = input_chunk.clone()
            if begin_loc > 0:
                target_chunk[:, :-trg_len] = -100

            outputs = model(input_chunk, labels=target_chunk)
            if not torch.isfinite(outputs.logits).all():
                print(
                    f"[nonfinite] logits dataset_window begin={begin_loc} end={end_loc}: "
                    f"{_nan_inf_summary(outputs.logits)}"
                )
            if not torch.isfinite(outputs.loss):
                print(
                    f"[nonfinite] loss dataset_window begin={begin_loc} end={end_loc}: "
                    f"value={outputs.loss.item()}"
                )
            neg_log_likelihood = outputs.loss * trg_len
            nlls.append(neg_log_likelihood)
            prev_end_loc = end_loc

            current_nll = torch.stack(nlls).sum()
            current_ppl = torch.exp(current_nll / (total_tokens + prev_end_loc)).item()
            pbar.set_postfix({"PPL": f"{current_ppl:.4f}", "tokens": f"{total_tokens + prev_end_loc:,}"})

            if end_loc == seq_len:
                break

        total_tokens += seq_len

    if not nlls:
        return None

    total_nll = torch.stack(nlls).sum()
    perplexity = torch.exp(total_nll / total_tokens).item()
    return {"perplexity": perplexity, "total_tokens": total_tokens}


def parse_args():
    parser = argparse.ArgumentParser(description="RBVT-squeeze style PPL evaluation for Squeeze caches")
    parser.add_argument("--model-path", required=True, help="Base HF model repo/path used to rebuild dense weights")
    parser.add_argument("--quantized-path", default="", help="Squeeze quantized cache path containing weights/ + lut_*")
    parser.add_argument("--model-name", default="", help="Display name for result output")
    parser.add_argument("--tokenizer-path", default="", help="Optional tokenizer source; defaults to --model-path")
    parser.add_argument("--datasets", nargs="+", default=["wikitext2", "c4"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--precision", type=int, default=None, help="Ignored; kept for compatibility with older scripts")
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--limit-tokens", type=int, default=None)
    parser.add_argument("--c4-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", default="./dataset_cache")
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--output-file", default="")
    return parser.parse_args()


def main():
    load_project_dotenv(verbose=True)
    args = parse_args()

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if args.dtype not in dtype_map:
        raise ValueError(f"Unsupported dtype {args.dtype}; choose from {sorted(dtype_map)}")

    preferred_dtype = _preferred_eval_dtype(args.model_path)
    if args.dtype == "float16" and preferred_dtype == "bfloat16":
        print(
            f"[eval_nonuquant_style_ppl] Switching eval dtype from float16 to bfloat16 for {args.model_path} "
            "to avoid Gemma/Qwen NaN instability."
        )
        args.dtype = preferred_dtype

    quantized_path = args.quantized_path or args.model_path
    quantized_dir = _resolve_quantized_cache_path(quantized_path)
    if not (quantized_dir / "weights").exists():
        raise ValueError(
            "Expected --quantized-path to point at a Squeeze quantized cache containing weights/ and lut_*/. "
            f"Got: {quantized_path}"
        )

    tokenizer_source = args.tokenizer_path or args.model_path
    tokenizer = _load_tokenizer(tokenizer_source, args.hf_token)
    model = _load_sqllm_quantized_model(
        base_model_path=args.model_path,
        quantized_path=str(quantized_dir),
        device=args.device,
        dtype=dtype_map[args.dtype],
        token=args.hf_token,
    )

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    dataset_loaders = {
        "WikiText-2": lambda: _load_wikitext2(cache_dir, args.seed, args.hf_token),
        "C4": lambda: _load_c4(cache_dir, args.seed, args.c4_samples, args.hf_token),
    }

    results = {}
    ordered_dataset_names = [_canonical_dataset_name(name) for name in args.datasets]
    for dataset_name in ordered_dataset_names:
        print("\n" + "=" * 80)
        print(f"Evaluating model on {dataset_name} | name={args.model_name or quantized_dir.name}")
        print("=" * 80)
        texts = dataset_loaders[dataset_name]()
        dataset_result = evaluate_sliding_window(
            model=model,
            tokenizer=tokenizer,
            texts=texts,
            device=args.device,
            max_length=args.max_length,
            stride=args.stride,
            limit_tokens=args.limit_tokens,
        )
        if dataset_result is None:
            print("  Evaluation failed (no results)")
            continue
        print(
            f"  Perplexity: {dataset_result['perplexity']:.4f} | "
            f"tokens={dataset_result['total_tokens']:,}"
        )
        results[dataset_name] = dataset_result

    print("\n" + "=" * 80)
    print("PERPLEXITY SUMMARY")
    print("=" * 80)
    for dataset_name, data in results.items():
        print(
            f"{dataset_name:<15} "
            f"name={args.model_name or quantized_dir.name:<32} "
            f"ppl={data['perplexity']:.4f} "
            f"tokens={data['total_tokens']:,}"
        )

    if args.output_file:
        payload = {
            "model_name": args.model_name or quantized_dir.name,
            "base_model_path": args.model_path,
            "quantized_path": str(quantized_dir),
            "evaluation": {
                "stride": args.stride,
                "max_length": args.max_length,
                "c4_samples": args.c4_samples,
                "datasets": results,
            },
        }
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        print(f"\nSaved results to {output_path}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()


