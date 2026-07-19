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


def _stats(fp: torch.Tensor, q: torch.Tensor):
    fp32 = fp.detach().float().cpu().flatten()
    q32 = q.detach().float().cpu().flatten()
    delta = q32 - fp32
    fp_norm = torch.linalg.vector_norm(fp32).clamp_min(1e-12)
    return {
        "fp_max": float(fp32.abs().max().item()),
        "q_max": float(q32.abs().max().item()),
        "delta_max": float(delta.abs().max().item()),
        "delta_l2": float(torch.linalg.vector_norm(delta).item()),
        "rel_delta_l2": float((torch.linalg.vector_norm(delta) / fp_norm).item()),
        "cosine": float(F.cosine_similarity(fp32, q32, dim=0).item()),
    }


def _print_stage(stage: str, fp: torch.Tensor, q: torch.Tensor, topk: int) -> None:
    s = _stats(fp, q)
    print(
        f"STAGE\t{stage}\tfp_max={s['fp_max']:.6e}\tq_max={s['q_max']:.6e}\t"
        f"delta_max={s['delta_max']:.6e}\tdelta_l2={s['delta_l2']:.6e}\t"
        f"rel_delta_l2={s['rel_delta_l2']:.6e}\tcosine={s['cosine']:.6e}"
    )
    fp_vec = fp.detach().float().cpu().flatten()
    q_vec = q.detach().float().cpu().flatten()
    delta = q_vec - fp_vec
    vals, idxs = torch.topk(delta.abs(), k=min(topk, delta.numel()))
    print(f"TOP_DELTA stage={stage} index\tfp\tq\tdelta\tabs_delta")
    for val, idx in zip(vals.tolist(), idxs.tolist()):
        print(f"{idx}\t{float(fp_vec[idx].item()):.6e}\t{float(q_vec[idx].item()):.6e}\t{float(delta[idx].item()):.6e}\t{float(val):.6e}")


def _causal_mask(hidden: torch.Tensor) -> torch.Tensor:
    seq_len = hidden.shape[1]
    min_value = torch.finfo(hidden.dtype).min
    mask = torch.full((seq_len, seq_len), min_value, dtype=hidden.dtype, device=hidden.device)
    mask = torch.triu(mask, diagonal=1)
    return mask[None, None, :, :]


def _run_manual_layer(layer, hidden: torch.Tensor, position_ids: torch.Tensor, position_embeddings):
    layer_input = hidden
    residual = hidden
    normed_attn = layer.input_layernorm(hidden)
    attn_out = layer.self_attn(
        normed_attn,
        attention_mask=_causal_mask(hidden),
        position_ids=position_ids,
        position_embeddings=position_embeddings,
        use_cache=False,
    )[0]
    after_attn = residual + attn_out
    residual = after_attn
    normed_mlp = layer.post_attention_layernorm(after_attn)
    gate = layer.mlp.gate_proj(normed_mlp)
    up = layer.mlp.up_proj(normed_mlp)
    product = F.silu(gate.float()).to(gate.dtype) * up
    mlp_out = layer.mlp.down_proj(product)
    layer_output = residual + mlp_out
    return {
        "layer_input": layer_input,
        "attn_normed_input": normed_attn,
        "attn_output": attn_out,
        "after_attn_residual": after_attn,
        "mlp_normed_input": normed_mlp,
        "mlp_gate": gate,
        "mlp_up": up,
        "mlp_product": product,
        "mlp_down_output": mlp_out,
        "layer_output": layer_output,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare one decoder layer stages between FP and quantized model.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--quantized-path", required=True)
    parser.add_argument("--tokenizer-path", default="")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--window", type=int, default=55)
    parser.add_argument("--token", type=int, default=642)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--hf-token", default=None)
    args = parser.parse_args()

    load_project_dotenv(ROOT_DIR)
    token = args.hf_token or None
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path or args.model_path, token=token, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    fp_model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=dtype, device_map=_hf_device_map(args.device), trust_remote_code=True, token=token)
    fp_model.eval()
    q_model = _load_sqllm_quantized_model(args.model_path, args.quantized_path, args.device, dtype, token)
    q_model.eval()

    texts = _load_wikitext2(ROOT_DIR / "dataset_cache", seed=42, token=token)
    input_ids_all = tokenizer("\n\n".join(texts), return_tensors="pt").input_ids
    begin = args.window * args.stride
    end = min(begin + args.max_length, int(input_ids_all.shape[1]))
    input_ids = input_ids_all[:, begin:end].to(args.device)

    with torch.no_grad():
        embeds = fp_model.model.embed_tokens(input_ids)
        fp_hidden = embeds
        q_hidden = q_model.model.embed_tokens(input_ids)
        position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        fp_position_embeddings = fp_model.model.rotary_emb(fp_hidden, position_ids)
        q_position_embeddings = q_model.model.rotary_emb(q_hidden, position_ids)
        fp_states = _run_manual_layer(fp_model.model.layers[args.layer], fp_hidden, position_ids, fp_position_embeddings)
        q_states = _run_manual_layer(q_model.model.layers[args.layer], q_hidden, position_ids, q_position_embeddings)

    tok = args.token
    print(f"window={args.window} token={tok} absolute_token={begin + tok} layer={args.layer} dtype={dtype}")
    for stage in [
        "layer_input",
        "attn_normed_input",
        "attn_output",
        "after_attn_residual",
        "mlp_normed_input",
        "mlp_gate",
        "mlp_up",
        "mlp_product",
        "mlp_down_output",
        "layer_output",
    ]:
        _print_stage(stage, fp_states[stage][0, tok], q_states[stage][0, tok], args.topk)


if __name__ == "__main__":
    main()