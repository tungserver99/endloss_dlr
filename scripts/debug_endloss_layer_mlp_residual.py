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

from scripts.eval_nonuquant_style_ppl import _load_sqllm_quantized_model, _load_wikitext2, _dequantize_module
from env_utils import load_project_dotenv


def _finite_max_abs(tensor: torch.Tensor) -> float:
    finite = torch.isfinite(tensor)
    if not finite.any():
        return float("nan")
    return float(tensor[finite].detach().float().abs().max().item())


def _top_abs(vec: torch.Tensor, k: int = 20):
    vals, idx = torch.topk(vec.detach().float().abs().cpu(), k=min(k, vec.numel()))
    return [(int(i), float(vec.detach().float().cpu()[i].item()), float(v.item())) for v, i in zip(vals, idx)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug one decoder layer MLP/residual and one down_proj output row.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--quantized-path", required=True)
    parser.add_argument("--tokenizer-path", default="")
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--window", type=int, default=55)
    parser.add_argument("--token", type=int, default=642)
    parser.add_argument("--row", type=int, default=2533)
    parser.add_argument("--col", type=int, default=2549)
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
    captured = {}

    def layer_pre(_module, inputs):
        captured["layer_input"] = inputs[0].detach()

    def attn_post(_module, _inputs, output):
        tensor = output[0] if isinstance(output, (tuple, list)) else output
        captured["attn_output"] = tensor.detach()

    def mlp_pre(_module, inputs):
        captured["mlp_input"] = inputs[0].detach()

    def down_post(_module, inputs, output):
        captured["down_input"] = inputs[0].detach()
        captured["down_output"] = output.detach() if torch.is_tensor(output) else output[0].detach()

    def layer_post(_module, _inputs, output):
        tensor = output[0] if isinstance(output, (tuple, list)) else output
        captured["layer_output"] = tensor.detach()

    handles = [
        layer.register_forward_pre_hook(layer_pre),
        layer.self_attn.register_forward_hook(attn_post),
        layer.mlp.register_forward_pre_hook(mlp_pre),
        layer.mlp.down_proj.register_forward_hook(down_post),
        layer.register_forward_hook(layer_post),
    ]
    with torch.no_grad():
        _ = model(input_ids)
    for handle in handles:
        handle.remove()

    tok = args.token
    row = args.row
    col = args.col
    layer_input = captured["layer_input"][0, tok].float().cpu()
    attn_output = captured["attn_output"][0, tok].float().cpu()
    mlp_input = captured["mlp_input"][0, tok].float().cpu()
    down_input = captured["down_input"][0, tok].float().cpu()
    down_output = captured["down_output"][0, tok].float().cpu()
    layer_output = captured["layer_output"][0, tok].float().cpu()

    down_weight = layer.mlp.down_proj.weight.detach().float().cpu()
    contrib = down_weight[row] * down_input
    fp32_dot = float(torch.dot(down_weight[row], down_input).item())

    print(f"window={args.window} token={tok} absolute_token={begin + tok} layer={args.layer} row={row} col={col}")
    print(f"layer_input max_abs={float(layer_input.abs().max().item()):.6e}")
    print(f"attn_output max_abs={float(attn_output.abs().max().item()):.6e}")
    print(f"mlp_input max_abs={float(mlp_input.abs().max().item()):.6e}")
    print(f"down_input/product max_abs={float(down_input.abs().max().item()):.6e}")
    print(f"down_output max_abs={float(down_output.abs().max().item()):.6e} row_value={float(down_output[row].item()):.6e} fp32_dot_row={fp32_dot:.6e}")
    print(f"layer_output max_abs={float(layer_output.abs().max().item()):.6e} row_value={float(layer_output[row].item()):.6e}")
    if col < down_input.numel():
        print(
            "TARGET_CONTRIB "
            f"col={col} input={float(down_input[col].item()):.6e} "
            f"weight={float(down_weight[row, col].item()):.6e} "
            f"product={float(contrib[col].item()):.6e}"
        )

    print("TOP_DOWN_ROW_CONTRIBS col\tinput\tweight\tproduct_abs\tproduct")
    for c, _value, abs_value in _top_abs(contrib, k=30):
        print(f"{c}\t{float(down_input[c].item()):.6e}\t{float(down_weight[row, c].item()):.6e}\t{abs_value:.6e}\t{float(contrib[c].item()):.6e}")

    weight_path = Path(args.quantized_path) / "weights" / f"l{args.layer}.pt"
    lut_dir = sorted([p for p in Path(args.quantized_path).iterdir() if p.is_dir() and p.name.startswith("lut_")], key=lambda p: int(p.name.split("_", 1)[1]))[-1]
    layer_weights = torch.load(weight_path, map_location="cpu")
    layer_luts = torch.load(lut_dir / f"l{args.layer}.pt", map_location="cpu")
    if "mlp.down_proj" in layer_weights:
        materialized = _dequantize_module(layer_weights["mlp.down_proj"], layer_luts["mlp.down_proj"], torch.device("cpu"), torch.float32)
        print(f"CACHE row={row} col={col} value={float(materialized[row, col].item()):.6e} row_max_abs={float(materialized[row].abs().max().item()):.6e} module_max_abs={float(materialized.abs().max().item()):.6e}")
        idx = torch.as_tensor(layer_weights["mlp.down_proj"])
        lut = torch.as_tensor(layer_luts["mlp.down_proj"])
        if idx.ndim == 3:
            code = int(idx[row, 0, col].item())
            usage = int((idx[row, 0] == code).sum().item())
            print(f"CACHE_CODE row={row} col={col} codeword={code} usage_count_in_row={usage} lut_value={float(lut[row, 0, code].item()):.6e}")


if __name__ == "__main__":
    main()