import json
import os
import pickle
from pathlib import Path

import lm_eval
import torch
import torch.nn as nn
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .helpers.utils import get_tokenizer_type, logprint, vprint


def _hf_device_map(device: str):
    return {"": device} if device != "auto" else "auto"


def _infer_base_model_repo(model_name: str) -> str:
    name = os.path.basename(model_name.rstrip("/"))
    prefixes = [
        "anyprec-sqllm-rbvt-cgc-",
        "anyprec-sqllm-cgc-rbvt-",
        "anyprec-sqllm-cgc-",
        "anyprec-sqllm-rbvt-",
        "anyprec-",
    ]
    for prefix in prefixes:
        if name.startswith(prefix):
            name = name[len(prefix):]
            break

    suffix_markers = [
        "-w2_",
        "-w3_",
        "-w4_",
        "-w2-",
        "-w3-",
        "-w4-",
    ]
    cut = len(name)
    for marker in suffix_markers:
        idx = name.find(marker)
        if idx != -1:
            cut = min(cut, idx)
    name = name[:cut]

    if name.startswith(("Meta-Llama-", "Llama-")):
        return f"meta-llama/{name}"
    if name.startswith("Mistral-"):
        return f"mistralai/{name}"
    if name.startswith(("Qwen", "Qwen2", "Qwen2.5", "Qwen3")):
        return f"Qwen/{name}"
    if name.startswith(("Gemma-", "gemma-")):
        return f"google/{name}"
    raise ValueError(f"Cannot infer base model repo from path/name: {model_name}")


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

    matches = []
    for candidate in candidates:
        suffix = f".{layer_idx}.{candidate}"
        matches.extend(module for name, module in linear_modules.items() if name.endswith(suffix))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one Linear module matching layer={layer_idx}, module={module_name}; "
            f"found {len(matches)}"
        )
    return matches[0]


def _dequantize_module(indices, lut, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    idx = torch.as_tensor(indices, device=device, dtype=torch.long)
    levels = torch.as_tensor(lut, device=device, dtype=dtype)
    rows, groups, _ = idx.shape
    row_ids = torch.arange(rows, device=device).view(-1, 1, 1)
    group_ids = torch.arange(groups, device=device).view(1, -1, 1)
    return levels[row_ids, group_ids, idx].reshape(rows, -1)


def _resolve_quantized_cache_path(model_path: str) -> str:
    path = Path(model_path)
    if (path / "weights").exists():
        return str(path)

    name = path.name
    cache_root = path.parent.parent if path.parent.name in {"packed", "post_sqllm_packed", "rbvt_sqllm_packed"} else Path("./cache")
    if name.startswith("anyprec-sqllm-rbvt-cgc-"):
        mapped = name.replace("anyprec-sqllm-rbvt-cgc-", "", 1)
        mapped = mapped.replace("-redpajama_", "-sqllm-rbvt-cgc-redpajama_", 1) if "-redpajama_" in mapped and "-sqllm-" not in mapped else mapped
        mapped = mapped.replace("-c4_", "-sqllm-rbvt-cgc-c4_", 1) if "-c4_" in mapped and "-sqllm-" not in mapped else mapped
        candidate = cache_root / "post_sqllm_quantized" / mapped
        if candidate.exists():
            return str(candidate)
    if name.startswith("anyprec-sqllm-cgc-rbvt-"):
        mapped = name.replace("anyprec-sqllm-cgc-rbvt-", "", 1)
        mapped = mapped.replace(f"-w", "-w", 1)
        mapped = mapped.replace("-redpajama_", "-sqllm-cgc-rbvt-redpajama_", 1) if "-redpajama_" in mapped and "-sqllm-" not in mapped else mapped
        mapped = mapped.replace("-c4_", "-sqllm-cgc-rbvt-c4_", 1) if "-c4_" in mapped and "-sqllm-" not in mapped else mapped
        candidate = cache_root / "post_sqllm_quantized" / mapped
        if candidate.exists():
            return str(candidate)
    if name.startswith("anyprec-sqllm-cgc-"):
        mapped = name.replace("anyprec-sqllm-cgc-", "", 1)
        mapped = mapped.replace("-redpajama_", "-sqllm-cgc-redpajama_", 1) if "-redpajama_" in mapped and "-sqllm-" not in mapped else mapped
        mapped = mapped.replace("-c4_", "-sqllm-cgc-c4_", 1) if "-c4_" in mapped and "-sqllm-" not in mapped else mapped
        candidate = cache_root / "post_sqllm_quantized" / mapped
        if candidate.exists():
            return str(candidate)
    if name.startswith("anyprec-sqllm-rbvt-"):
        mapped = name.replace("anyprec-sqllm-rbvt-", "", 1)
        mapped = mapped.replace("-redpajama_", "-sqllm-rbvt-redpajama_", 1) if "-redpajama_" in mapped and "-sqllm-" not in mapped else mapped
        mapped = mapped.replace("-c4_", "-sqllm-rbvt-c4_", 1) if "-c4_" in mapped and "-sqllm-" not in mapped else mapped
        candidate = cache_root / "post_sqllm_quantized" / mapped
        if candidate.exists():
            return str(candidate)
    if name.startswith("anyprec-"):
        mapped = name.replace("anyprec-", "", 1)
        candidate = cache_root / "quantized" / mapped
        if candidate.exists():
            return str(candidate)

    return model_path


@torch.no_grad()
def _load_sqllm_quantized_model(base_model_path: str, quantized_path: str, device: str, dtype: torch.dtype, verbose: bool):
    logprint(verbose, f"Loading base HF model from {base_model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=dtype,
        device_map=_hf_device_map(device),
        trust_remote_code=True,
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
    linear_modules = _find_linear_modules(model)
    weight_files = sorted(weight_dir.glob("l*.pt"), key=lambda p: int(p.stem[1:]))

    for weight_file in tqdm(weight_files, desc="Materializing quantized weights", disable=not verbose):
        layer_idx = int(weight_file.stem[1:])
        layer_weights = torch.load(weight_file, map_location="cpu")
        layer_luts = torch.load(lut_dir / weight_file.name, map_location="cpu")

        for module_name, indices in layer_weights.items():
            target_module = _resolve_linear_module(linear_modules, layer_idx, module_name)
            dequantized = _dequantize_module(
                indices=indices,
                lut=layer_luts[module_name],
                device=target_module.weight.device,
                dtype=target_module.weight.dtype,
            )
            target_module.weight.data.copy_(dequantized.to(target_module.weight.dtype))
            del dequantized

        del layer_weights, layer_luts
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return model


@torch.no_grad()
def auto_model_load(model_path, device='cuda', dtype=torch.float16, verbose=True):
    logprint(verbose, "Loading tokenizer and model...")

    resolved_quantized_path = _resolve_quantized_cache_path(model_path)
    if Path(resolved_quantized_path).exists() and (Path(resolved_quantized_path) / "weights").exists():
        base_model_repo = _infer_base_model_repo(model_path)
        tokenizer = AutoTokenizer.from_pretrained(base_model_repo, trust_remote_code=True, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = _load_sqllm_quantized_model(
            base_model_path=base_model_repo,
            quantized_path=resolved_quantized_path,
            device=device,
            dtype=dtype,
            verbose=verbose,
        )
        tokenizer_type = get_tokenizer_type(base_model_repo)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=_hf_device_map(device),
            trust_remote_code=True,
        )
        model.eval()
        tokenizer_type = get_tokenizer_type(model_path)

    logprint(verbose, f"{model.__class__.__name__} model loaded.")
    return tokenizer_type, tokenizer, model


def _load_wikitext2(cache_dir: Path, token: str | None = None):
    cache_file = cache_dir / "wikitext2_test.pkl"
    if cache_file.exists():
        with open(cache_file, "rb") as handle:
            return pickle.load(handle)

    dataset = load_dataset(
        "Salesforce/wikitext",
        "wikitext-2-raw-v1",
        split="test",
        token=token,
    )
    result = ["\n".join([x for x in dataset["text"] if x])]
    with open(cache_file, "wb") as handle:
        pickle.dump(result, handle)
    return result


def _load_c4(cache_dir: Path, n_samples: int = 2000, token: str | None = None):
    cache_file = cache_dir / f"c4_validation_n{n_samples}.pkl"
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


def _load_eval_texts(testcase_name: str, cache_dir: Path, token: str | None = None):
    lowered = testcase_name.lower()
    if lowered == "wikitext2":
        return _load_wikitext2(cache_dir, token=token)
    if lowered == "c4":
        return _load_c4(cache_dir, token=token)
    raise ValueError(f"Unsupported testcase {testcase_name}")


@torch.no_grad()
def evaluate_ppl(model, tokenizer, testcases, verbose=True, chunk_size=2048, tokenizer_type=None):
    del tokenizer_type
    model.eval()
    results = {}
    cache_dir = Path("./dataset_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device

    for testcase_name in testcases:
        vprint(verbose, f"---------------------- {testcase_name} ----------------------")
        texts = _load_eval_texts(testcase_name, cache_dir)
        nlls = []
        total_tokens = 0

        for text in texts:
            input_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids
            if tokenizer.bos_token_id is not None:
                if input_ids.shape[1] == 0 or input_ids[0, 0].item() != tokenizer.bos_token_id:
                    bos = torch.tensor([[tokenizer.bos_token_id]], device=input_ids.device)
                    input_ids = torch.cat([bos, input_ids], dim=1)

            input_ids = input_ids.to(device)
            seq_len = input_ids.size(1)
            if seq_len < 2:
                continue

            prev_end_loc = 0
            for begin_loc in tqdm(range(0, seq_len, 512), disable=not verbose, desc=f"{testcase_name} windows"):
                end_loc = min(begin_loc + chunk_size, seq_len)
                trg_len = end_loc - prev_end_loc
                input_chunk = input_ids[:, begin_loc:end_loc]
                target_chunk = input_chunk.clone()
                if begin_loc > 0:
                    target_chunk[:, :-trg_len] = -100

                outputs = model(input_chunk, labels=target_chunk)
                nlls.append(outputs.loss * trg_len)
                prev_end_loc = end_loc
                if end_loc == seq_len:
                    break

            total_tokens += seq_len

        if not nlls:
            continue

        ppl = torch.exp(torch.stack(nlls).sum() / total_tokens).item()
        logprint(verbose, f"Perplexity ({testcase_name}): {ppl}")
        results[testcase_name] = ppl

    return results


@torch.no_grad()
def run_lm_eval(tokenizer, model, tasks, verbose=True):
    model.eval()
    model_lm = lm_eval.models.huggingface.HFLM(pretrained=model, tokenizer=tokenizer)
    eval_results = lm_eval.simple_evaluate(model=model_lm, tasks=tasks)

    if verbose:
        logprint(verbose, json.dumps(eval_results['results'], indent=4))

    return {task: eval_results['results'][task] for task in tasks}
