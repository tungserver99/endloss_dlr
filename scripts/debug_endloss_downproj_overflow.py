#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
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
    return float(tensor[finite].detach().abs().max().item())


def _first_nonfinite_coord(tensor: torch.Tensor):
    bad = ~torch.isfinite(tensor)
    if not bad.any():
        return None
    flat = int(bad.reshape(-1).nonzero(as_tuple=False)[0].item())
    coord = []
    shape = list(tensor.shape)
    for size in reversed(shape):
        coord.append(flat % size)
        flat //= size
    return tuple(reversed(coord))


def _get_module(model, path: str):
    module = model
    for part in path.split("."):
        module = getattr(module, part)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug FP16 overflow in one down_proj module.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--quantized-path", required=True)
    parser.add_argument("--tokenizer-path", default="")
    parser.add_argument("--layer", type=int, default=30)
    parser.add_argument("--module-path", default="mlp.down_proj")
    parser.add_argument("--window", type=int, required=True)
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

    layer_module = _get_module(model, f"model.layers.{args.layer}.{args.module_path}")
    captured = {}

    def hook(_module, inputs, output):
        captured["input"] = inputs[0].detach()
        captured["output"] = output.detach() if torch.is_tensor(output) else output[0].detach()

    handle = layer_module.register_forward_hook(hook)
    with torch.no_grad():
        outputs = model(input_ids)
    handle.remove()

    inp = captured["input"]
    out = captured["output"]
    logits = outputs.logits
    print(f"window={args.window} token_range=[{begin},{end}) module=model.layers.{args.layer}.{args.module_path}")
    print(f"input shape={tuple(inp.shape)} dtype={inp.dtype} finite={bool(torch.isfinite(inp).all().item())} max_abs_finite={_finite_max_abs(inp):.6e}")
    print(f"output shape={tuple(out.shape)} dtype={out.dtype} finite={bool(torch.isfinite(out).all().item())} max_abs_finite={_finite_max_abs(out):.6e}")
    print(f"logits finite={bool(torch.isfinite(logits).all().item())} max_abs_finite={_finite_max_abs(logits):.6e}")

    out_bad = _first_nonfinite_coord(out)
    if out_bad is None:
        print("NO_NONFINITE_IN_TARGET_MODULE_OUTPUT")
    else:
        batch_idx, token_idx, row_idx = out_bad[-3], out_bad[-2], out_bad[-1]
        x = inp[batch_idx, token_idx].float().cpu()
        w = layer_module.weight.detach().float().cpu()
        contrib = w[row_idx] * x
        abs_contrib = contrib.abs()
        top_vals, top_cols = torch.topk(abs_contrib, k=min(20, abs_contrib.numel()))
        print(f"FIRST_NONFINITE_OUTPUT coord={out_bad} output_row={row_idx}")
        print(f"input_token_max_abs={float(x.abs().max().item()):.6e} weight_row_max_abs={float(w[row_idx].abs().max().item()):.6e} fp32_dot={float(torch.dot(w[row_idx], x).item()):.6e}")
        print("TOP_CONTRIBS col\tinput\tweight\tproduct_abs\tproduct")
        for val, col in zip(top_vals.tolist(), top_cols.tolist()):
            prod = float(contrib[col].item())
            print(f"{col}\t{float(x[col].item()):.6e}\t{float(w[row_idx, col].item()):.6e}\t{float(val):.6e}\t{prod:.6e}")

    weight_path = Path(args.quantized_path) / "weights" / f"l{args.layer}.pt"
    lut_dir = sorted([p for p in Path(args.quantized_path).iterdir() if p.is_dir() and p.name.startswith("lut_")], key=lambda p: int(p.name.split("_", 1)[1]))[-1]
    layer_weights = torch.load(weight_path, map_location="cpu")
    layer_luts = torch.load(lut_dir / f"l{args.layer}.pt", map_location="cpu")
    module_key = args.module_path
    if module_key in layer_weights:
        materialized = _dequantize_module(layer_weights[module_key], layer_luts[module_key], torch.device("cpu"), torch.float32).cpu()
        flat = int(materialized.abs().reshape(-1).argmax().item())
        row = flat // materialized.shape[1]
        col = flat % materialized.shape[1]
        print(f"CACHE_MODULE_MAX row={row} col={col} value={float(materialized[row, col].item()):.6e} max_abs={float(materialized.abs().max().item()):.6e}")
        if out_bad is not None:
            print(f"CACHE_WEIGHT_AT_NONFINITE_ROW_MAX row={row_idx} max_abs={float(materialized[row_idx].abs().max().item()):.6e} argmax_col={int(materialized[row_idx].abs().argmax().item())}")


if __name__ == "__main__":
    main()