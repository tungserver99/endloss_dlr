#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.eval_nonuquant_style_ppl import _load_sqllm_quantized_model, _load_wikitext2
from env_utils import load_project_dotenv


def _hf_device_map(device: str):
    return {"": device} if device != "auto" else "auto"


def _stats(fp: torch.Tensor, q: torch.Tensor) -> dict[str, float]:
    fp32 = fp.detach().float().cpu().flatten()
    q32 = q.detach().float().cpu().flatten()
    delta = q32 - fp32
    fp_norm = torch.linalg.vector_norm(fp32).clamp_min(1e-12)
    q_norm = torch.linalg.vector_norm(q32).clamp_min(1e-12)
    return {
        "fp_max": float(fp32.abs().max().item()),
        "q_max": float(q32.abs().max().item()),
        "delta_max": float(delta.abs().max().item()),
        "delta_l2": float(torch.linalg.vector_norm(delta).item()),
        "rel_delta_l2": float((torch.linalg.vector_norm(delta) / fp_norm).item()),
        "cosine": float(F.cosine_similarity(fp32, q32, dim=0).item()),
        "fp_norm": float(fp_norm.item()),
        "q_norm": float(q_norm.item()),
    }


def _print_stats(prefix: str, layer_idx: int, token_idx: int, fp: torch.Tensor, q: torch.Tensor) -> None:
    item = _stats(fp, q)
    print(
        f"{prefix}\t{layer_idx}\t{token_idx}\t"
        f"{item['fp_max']:.6e}\t{item['q_max']:.6e}\t{item['delta_max']:.6e}\t"
        f"{item['delta_l2']:.6e}\t{item['rel_delta_l2']:.6e}\t{item['cosine']:.6e}\t"
        f"{item['fp_norm']:.6e}\t{item['q_norm']:.6e}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare FP vs quantized hidden states layer by layer.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--quantized-path", required=True)
    parser.add_argument("--tokenizer-path", default="")
    parser.add_argument("--window", type=int, default=55)
    parser.add_argument("--token", type=int, default=642)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hf-token", default=None)
    args = parser.parse_args()

    load_project_dotenv(ROOT_DIR)
    token = args.hf_token or None
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path or args.model_path, token=token, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    fp_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map=_hf_device_map(args.device),
        trust_remote_code=True,
        token=token,
    )
    fp_model.eval()
    q_model = _load_sqllm_quantized_model(args.model_path, args.quantized_path, args.device, dtype, token)
    q_model.eval()

    texts = _load_wikitext2(ROOT_DIR / "dataset_cache", seed=42, token=token)
    input_ids_all = tokenizer("\n\n".join(texts), return_tensors="pt").input_ids
    begin = args.window * args.stride
    end = min(begin + args.max_length, int(input_ids_all.shape[1]))
    input_ids = input_ids_all[:, begin:end].to(args.device)
    tok = args.token
    print(f"window={args.window} token={tok} absolute_token={begin + tok} token_range=[{begin},{end}) dtype={dtype}")

    fp_in = {}
    fp_out = {}
    q_in = {}
    q_out = {}

    handles = []
    def pre_store(store, idx):
        def hook(_module, inputs):
            store[idx] = inputs[0].detach()[:, tok, :].cpu()
        return hook
    def post_store(store, idx):
        def hook(_module, _inputs, output):
            tensor = output[0] if isinstance(output, (tuple, list)) else output
            store[idx] = tensor.detach()[:, tok, :].cpu()
        return hook

    for idx, layer in enumerate(fp_model.model.layers):
        handles.append(layer.register_forward_pre_hook(pre_store(fp_in, idx)))
        handles.append(layer.register_forward_hook(post_store(fp_out, idx)))
    for idx, layer in enumerate(q_model.model.layers):
        handles.append(layer.register_forward_pre_hook(pre_store(q_in, idx)))
        handles.append(layer.register_forward_hook(post_store(q_out, idx)))

    with torch.no_grad():
        fp_logits = fp_model(input_ids).logits[:, tok, :].detach().cpu()
        q_logits = q_model(input_ids).logits[:, tok, :].detach().cpu()

    for handle in handles:
        handle.remove()

    print("stage\tlayer\ttoken\tfp_max\tq_max\tdelta_max\tdelta_l2\trel_delta_l2\tcosine\tfp_norm\tq_norm")
    for idx in range(len(fp_model.model.layers)):
        _print_stats("layer_input", idx, tok, fp_in[idx], q_in[idx])
        _print_stats("layer_output", idx, tok, fp_out[idx], q_out[idx])
    _print_stats("logits", -1, tok, fp_logits, q_logits)


if __name__ == "__main__":
    main()