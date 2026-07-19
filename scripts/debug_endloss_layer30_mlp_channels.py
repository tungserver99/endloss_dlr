#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.eval_nonuquant_style_ppl import _load_sqllm_quantized_model, _load_wikitext2
from env_utils import load_project_dotenv


def _finite_max_abs(tensor: torch.Tensor) -> float:
    finite = torch.isfinite(tensor)
    if not finite.any():
        return float("nan")
    return float(tensor[finite].detach().abs().max().item())


def _top_abs(vec: torch.Tensor, k: int = 20):
    vals, idx = torch.topk(vec.detach().float().abs().cpu(), k=min(k, vec.numel()))
    return [(int(i), float(vec.detach().float().cpu()[i].item()), float(v.item())) for v, i in zip(vals, idx)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe layer MLP gate/up/product channels before down_proj overflow.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--quantized-path", required=True)
    parser.add_argument("--tokenizer-path", default="")
    parser.add_argument("--layer", type=int, default=30)
    parser.add_argument("--window", type=int, default=55)
    parser.add_argument("--token", type=int, default=642, help="Token index within the selected window")
    parser.add_argument("--channels", type=int, nargs="+", default=[3721, 7006])
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

    model = _load_sqllm_quantized_model(args.model_path, args.quantized_path, args.device, dtype, token)
    model.eval()

    texts = _load_wikitext2(ROOT_DIR / "dataset_cache", seed=42, token=token)
    input_ids_all = tokenizer("\n\n".join(texts), return_tensors="pt").input_ids
    begin = args.window * args.stride
    end = min(begin + args.max_length, int(input_ids_all.shape[1]))
    input_ids = input_ids_all[:, begin:end].to(args.device)

    layer = model.model.layers[args.layer]
    mlp = layer.mlp
    captured = {}

    def mlp_pre_hook(_module, inputs):
        captured["mlp_input"] = inputs[0].detach()

    handle = mlp.register_forward_pre_hook(mlp_pre_hook)
    with torch.no_grad():
        _ = model(input_ids)
    handle.remove()

    h = captured["mlp_input"]
    tok = args.token
    if tok >= h.shape[1]:
        raise ValueError(f"token {tok} out of window length {h.shape[1]}")
    h_tok = h[0, tok]

    with torch.no_grad():
        gate = mlp.gate_proj(h[0:1, tok:tok + 1]).squeeze(0).squeeze(0)
        up = mlp.up_proj(h[0:1, tok:tok + 1]).squeeze(0).squeeze(0)
        silu_gate = F.silu(gate.float()).to(gate.dtype)
        product = silu_gate * up
        down = mlp.down_proj(product.view(1, 1, -1)).squeeze(0).squeeze(0)

    print(f"window={args.window} token_range=[{begin},{end}) token_in_window={tok} absolute_token={begin + tok} layer={args.layer}")
    print(f"mlp_input finite={bool(torch.isfinite(h_tok).all().item())} max_abs={_finite_max_abs(h_tok):.6e}")
    print(f"gate finite={bool(torch.isfinite(gate).all().item())} max_abs={_finite_max_abs(gate):.6e}")
    print(f"silu_gate finite={bool(torch.isfinite(silu_gate).all().item())} max_abs={_finite_max_abs(silu_gate):.6e}")
    print(f"up finite={bool(torch.isfinite(up).all().item())} max_abs={_finite_max_abs(up):.6e}")
    print(f"product finite={bool(torch.isfinite(product).all().item())} max_abs={_finite_max_abs(product):.6e}")
    print(f"down finite={bool(torch.isfinite(down).all().item())} max_abs_finite={_finite_max_abs(down):.6e}")

    print("CHANNELS channel\tmlp_input\tgate\tsilu_gate\tup\tproduct")
    for channel in args.channels:
        mlp_input_value = "N/A"
        if channel < h_tok.numel():
            mlp_input_value = f"{float(h_tok[channel].float().item()):.6e}"
        print(
            f"{channel}\t{mlp_input_value}\t"
            f"{float(gate[channel].float().item()):.6e}\t"
            f"{float(silu_gate[channel].float().item()):.6e}\t"
            f"{float(up[channel].float().item()):.6e}\t"
            f"{float(product[channel].float().item()):.6e}"
        )

    print("TOP_PRODUCT_CHANNELS channel\tvalue\tabs")
    for channel, value, abs_value in _top_abs(product, k=30):
        print(f"{channel}\t{value:.6e}\t{abs_value:.6e}")
    print("TOP_GATE_CHANNELS channel\tvalue\tabs")
    for channel, value, abs_value in _top_abs(gate, k=20):
        print(f"{channel}\t{value:.6e}\t{abs_value:.6e}")
    print("TOP_UP_CHANNELS channel\tvalue\tabs")
    for channel, value, abs_value in _top_abs(up, k=20):
        print(f"{channel}\t{value:.6e}\t{abs_value:.6e}")


if __name__ == "__main__":
    main()