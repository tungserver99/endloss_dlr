#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.eval_nonuquant_style_ppl import _load_sqllm_quantized_model
from env_utils import load_project_dotenv


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug first non-finite activation/logit after materializing EndLoss_DLR cache.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--quantized-path", required=True)
    parser.add_argument("--tokenizer-path", default="")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--hf-token", default=None)
    args = parser.parse_args()

    load_project_dotenv(Path(__file__).resolve().parent.parent)
    token = args.hf_token or None
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path or args.model_path, token=token, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = _load_sqllm_quantized_model(args.model_path, args.quantized_path, args.device, dtype, token)
    model.eval()

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", token=token)
    text = "\n\n".join(dataset["text"])
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc.input_ids[:, args.offset:args.offset + args.max_length].to(args.device)
    print(f"input shape={tuple(input_ids.shape)} dtype={dtype} quantized_path={args.quantized_path}")

    hooks = []
    first_bad = {"name": None}

    def make_hook(name):
        def hook(_module, _inputs, output):
            if first_bad["name"] is not None:
                return
            tensors = output if isinstance(output, (tuple, list)) else (output,)
            for tensor in tensors:
                if torch.is_tensor(tensor) and torch.is_floating_point(tensor) and not torch.isfinite(tensor).all():
                    finite = torch.isfinite(tensor)
                    max_abs = float(tensor[finite].detach().abs().max().item()) if finite.any() else float("nan")
                    first_bad["name"] = name
                    print(f"FIRST_NONFINITE_ACTIVATION name={name} shape={tuple(tensor.shape)} dtype={tensor.dtype} max_abs_finite={max_abs:.6e}")
                    break
        return hook

    for name, module in model.named_modules():
        if any(s in name for s in ("layers.", "model.norm", "lm_head")):
            hooks.append(module.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits
        finite = torch.isfinite(logits)
        print(f"logits finite={bool(finite.all().item())} dtype={logits.dtype}")
        if finite.any():
            print(f"logits max_abs_finite={float(logits[finite].abs().max().item()):.6e}")
        if first_bad["name"] is None and not finite.all():
            print("FIRST_NONFINITE_ACTIVATION not caught by hooks; logits are non-finite")
        if first_bad["name"] is None and finite.all():
            print("OK: forward logits finite on this window")

    for hook in hooks:
        hook.remove()


if __name__ == "__main__":
    main()