#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.eval_nonuquant_style_ppl import _load_sqllm_quantized_model
from env_utils import load_project_dotenv


def _finite_max_abs(tensor: torch.Tensor) -> float:
    finite = torch.isfinite(tensor)
    if not finite.any():
        return float("nan")
    return float(tensor[finite].detach().abs().max().item())


def _interesting_module(name: str) -> bool:
    if name == "lm_head" or name.endswith("model.norm"):
        return True
    if ".layers." in name:
        return True
    if name.startswith("model.layers."):
        return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan WikiText2 windows and report first non-finite activation/logit.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--quantized-path", required=True)
    parser.add_argument("--tokenizer-path", default="")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--max-windows", type=int, default=260)
    parser.add_argument("--start-window", type=int, default=0)
    parser.add_argument("--hf-token", default=None)
    args = parser.parse_args()

    load_project_dotenv(ROOT_DIR)
    token = args.hf_token or None
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path or args.model_path,
        token=token,
        use_fast=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = _load_sqllm_quantized_model(args.model_path, args.quantized_path, args.device, dtype, token)
    model.eval()

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", token=token)
    text = "\n\n".join(dataset["text"])
    input_ids_all = tokenizer(text, return_tensors="pt").input_ids
    seq_len = int(input_ids_all.shape[1])
    total_windows = (seq_len + args.stride - 1) // args.stride
    stop_window = min(total_windows, args.start_window + args.max_windows)
    print(
        f"scan quantized_path={args.quantized_path} dtype={dtype} seq_len={seq_len} "
        f"windows=[{args.start_window},{stop_window})/{total_windows} max_length={args.max_length} stride={args.stride}"
    )

    state = {"bad": None, "window": None, "begin": None, "end": None}

    def make_hook(name: str):
        def hook(_module, _inputs, output):
            if state["bad"] is not None:
                return
            tensors = output if isinstance(output, (tuple, list)) else (output,)
            for tensor in tensors:
                if torch.is_tensor(tensor) and torch.is_floating_point(tensor) and not torch.isfinite(tensor).all():
                    state["bad"] = (
                        name,
                        tuple(tensor.shape),
                        str(tensor.dtype),
                        _finite_max_abs(tensor),
                    )
                    break
        return hook

    hooks = []
    for name, module in model.named_modules():
        if _interesting_module(name):
            hooks.append(module.register_forward_hook(make_hook(name)))

    try:
        for window_idx in tqdm(range(args.start_window, stop_window), desc="Forward scan", unit="win"):
            begin = window_idx * args.stride
            end = min(begin + args.max_length, seq_len)
            if end <= begin + 1:
                break
            state["bad"] = None
            state["window"] = window_idx
            state["begin"] = begin
            state["end"] = end
            input_ids = input_ids_all[:, begin:end].to(args.device)
            with torch.no_grad():
                outputs = model(input_ids)
                logits = outputs.logits
            if state["bad"] is not None:
                name, shape, dtype_name, max_abs = state["bad"]
                print(
                    "FIRST_NONFINITE_ACTIVATION "
                    f"window={window_idx} token_range=[{begin},{end}) name={name} "
                    f"shape={shape} dtype={dtype_name} max_abs_finite={max_abs:.6e}"
                )
                return
            if not torch.isfinite(logits).all():
                print(
                    "FIRST_NONFINITE_LOGITS "
                    f"window={window_idx} token_range=[{begin},{end}) "
                    f"shape={tuple(logits.shape)} dtype={logits.dtype} max_abs_finite={_finite_max_abs(logits):.6e}"
                )
                return
            if window_idx % 25 == 0:
                print(
                    f"window={window_idx} token_range=[{begin},{end}) logits_max_abs={_finite_max_abs(logits):.6e}"
                )
            del input_ids, outputs, logits
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        print(f"OK: no non-finite activation/logits in scanned windows [{args.start_window},{stop_window})")
    finally:
        for hook in hooks:
            hook.remove()


if __name__ == "__main__":
    main()