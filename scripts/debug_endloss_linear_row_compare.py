#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from any_precision.analyzer import get_analyzer
from scripts.eval_nonuquant_style_ppl import _load_sqllm_quantized_model, _load_wikitext2, _dequantize_module
from env_utils import load_project_dotenv


def _hf_device_map(device: str):
    return {"": device} if device != "auto" else "auto"


def _get_module(model, path: str):
    module = model
    for part in path.split("."):
        module = getattr(module, part)
    return module


def _row_metrics(fp_row: torch.Tensor, q_row: torch.Tensor, x: torch.Tensor):
    fp = fp_row.float().cpu()
    q = q_row.float().cpu()
    x_cpu = x.float().cpu()
    err = q - fp
    fp_dot = float(torch.dot(fp, x_cpu).item())
    q_dot = float(torch.dot(q, x_cpu).item())
    cos = float(torch.nn.functional.cosine_similarity(fp, q, dim=0).item())
    return {
        "max_abs_fp_weight": float(fp.abs().max().item()),
        "max_abs_q_weight": float(q.abs().max().item()),
        "max_abs_err": float(err.abs().max().item()),
        "l1_err": float(err.abs().sum().item()),
        "l2_err": float(torch.linalg.vector_norm(err).item()),
        "rel_l2_err": float((torch.linalg.vector_norm(err) / torch.linalg.vector_norm(fp).clamp_min(1e-12)).item()),
        "cosine": cos,
        "fp_dot": fp_dot,
        "q_dot": q_dot,
        "dot_delta": q_dot - fp_dot,
        "abs_dot_delta": abs(q_dot - fp_dot),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare original vs quantized linear row outputs for one layer/token.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--quantized-path", required=True)
    parser.add_argument("--tokenizer-path", default="")
    parser.add_argument("--layer", type=int, default=30)
    parser.add_argument("--window", type=int, default=55)
    parser.add_argument("--token", type=int, default=642)
    parser.add_argument("--rows", type=int, nargs="+", default=[3721, 7006])
    parser.add_argument("--modules", nargs="+", default=["mlp.gate_proj", "mlp.up_proj"])
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

    captured = {}
    def capture_input(name):
        def hook(_module, inputs):
            captured[name] = inputs[0].detach()
        return hook

    fp_mlp = _get_module(fp_model, f"model.layers.{args.layer}.mlp")
    q_mlp = _get_module(q_model, f"model.layers.{args.layer}.mlp")
    fp_handle = fp_mlp.register_forward_pre_hook(capture_input("fp"))
    q_handle = q_mlp.register_forward_pre_hook(capture_input("q"))
    with torch.no_grad():
        _ = fp_model(input_ids)
        _ = q_model(input_ids)
    fp_handle.remove()
    q_handle.remove()

    fp_x = captured["fp"][0, args.token].detach()
    q_x = captured["q"][0, args.token].detach()
    print(f"window={args.window} token={args.token} absolute_token={begin + args.token} layer={args.layer}")
    print(f"mlp_input fp_max_abs={float(fp_x.float().abs().max().item()):.6e} q_max_abs={float(q_x.float().abs().max().item()):.6e} input_delta_l2={float(torch.linalg.vector_norm((q_x-fp_x).float()).item()):.6e}")

    weight_dir = Path(args.quantized_path) / "weights"
    lut_dir = sorted([p for p in Path(args.quantized_path).iterdir() if p.is_dir() and p.name.startswith("lut_")], key=lambda p: int(p.name.split("_", 1)[1]))[-1]
    layer_weights = torch.load(weight_dir / f"l{args.layer}.pt", map_location="cpu")
    layer_luts = torch.load(lut_dir / f"l{args.layer}.pt", map_location="cpu")

    print("module\trow\tfp_out_fp_input\tq_out_fp_input\tfp_out_q_input\tq_out_q_input\tdelta_on_q_input\tmax_abs_fp_weight\tmax_abs_q_weight\tmax_abs_err\tl1_err\tl2_err\trel_l2_err\tcosine")
    for module_name in args.modules:
        fp_module = _get_module(fp_model, f"model.layers.{args.layer}.{module_name}")
        q_weight = _dequantize_module(layer_weights[module_name], layer_luts[module_name], torch.device("cpu"), torch.float32)
        fp_weight = fp_module.weight.detach().float().cpu()
        for row in args.rows:
            metrics_fp_input = _row_metrics(fp_weight[row], q_weight[row], fp_x)
            metrics_q_input = _row_metrics(fp_weight[row], q_weight[row], q_x)
            print(
                f"{module_name}\t{row}\t"
                f"{metrics_fp_input['fp_dot']:.6e}\t{metrics_fp_input['q_dot']:.6e}\t"
                f"{metrics_q_input['fp_dot']:.6e}\t{metrics_q_input['q_dot']:.6e}\t{metrics_q_input['dot_delta']:.6e}\t"
                f"{metrics_q_input['max_abs_fp_weight']:.6e}\t{metrics_q_input['max_abs_q_weight']:.6e}\t"
                f"{metrics_q_input['max_abs_err']:.6e}\t{metrics_q_input['l1_err']:.6e}\t{metrics_q_input['l2_err']:.6e}\t"
                f"{metrics_q_input['rel_l2_err']:.6e}\t{metrics_q_input['cosine']:.6e}"
            )


if __name__ == "__main__":
    main()