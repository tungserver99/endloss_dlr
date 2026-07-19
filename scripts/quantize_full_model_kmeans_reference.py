#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numba
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from any_precision.analyzer import get_analyzer
from any_precision.quantization.quantize import _seed_and_upscale_layer, _save_results
from any_precision.quantization.end_loss_dlr.serialization import save_metadata
from env_utils import load_project_dotenv


def _setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s | %(levelname)s] %(message)s',
        datefmt='%H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def _default_output_dir(cache_dir: str, model_name: str, bits: int, dataset: str, num_examples: int, seq_len: int) -> str:
    return os.path.join(
        cache_dir,
        "quantized",
        f"{model_name}-w{bits}_kmeansref-{dataset}_s{num_examples}_blk{seq_len}",
    )


def _configure_numba_threads(cpu_count: int | None):
    if cpu_count is None:
        try:
            cpu_count = int(os.popen("nproc").read().strip())
        except Exception:
            cpu_count = os.cpu_count() or 1
    cpu_count = max(1, int(cpu_count))
    numba.set_num_threads(cpu_count)
    logging.info("Using %d CPU threads for flash1dkmeans reference quantization", cpu_count)
    return cpu_count


def _make_uniform_gradients(layer_weights: list[np.ndarray]) -> list[np.ndarray]:
    gradients = []
    for weight in layer_weights:
        gradients.append(np.ones_like(weight, dtype=np.float32))
    return gradients


def quantize_reference_kmeans(
    model: str,
    bits: int,
    cache_dir: str,
    dataset: str,
    seq_len: int,
    num_examples: int,
    output_dir: str,
    cpu_count: int | None,
    overwrite_quantize: bool,
):
    analyzer = get_analyzer(model, include_tokenizer=False)
    _configure_numba_threads(cpu_count)

    if overwrite_quantize and os.path.isdir(output_dir):
        logging.info("Removing existing quantized cache at %s", output_dir)
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    logging.info("Beginning full-model flash1dkmeans reference quantization")
    logging.info("Output cache: %s", output_dir)

    for layer_idx in range(analyzer.num_layers):
        layer_module_paths = analyzer.get_layer_module_paths(layer_idx)
        module_names = list(layer_module_paths.keys())
        layer_weights = [analyzer.get_layer_weights(layer_idx)[name].float().numpy() for name in module_names]
        layer_gradients = _make_uniform_gradients(layer_weights)

        luts_by_bit_by_module, parent_weights = _seed_and_upscale_layer(
            layer_gradients,
            layer_weights,
            bits,
            bits,
            1,
            random_state=None,
        )
        _save_results(output_dir, bits, bits, module_names, luts_by_bit_by_module, parent_weights, layer_idx)
        logging.info("Quantized layer %d/%d", layer_idx + 1, analyzer.num_layers)

    metadata = {
        "method": "flash1dkmeans_reference",
        "model": model,
        "bits": bits,
        "dataset": dataset,
        "seq_len": seq_len,
        "num_examples": num_examples,
        "output_dir": output_dir,
        "group_count": 1,
        "sample_weight": "uniform_ones",
    }
    save_metadata(output_dir, metadata)
    logging.info("Reference K-means quantization complete")
    return output_dir


def maybe_run_eval(model: str, quantized_path: str, datasets: list[str], dtype: str, stride: int, max_length: int, c4_samples: int, output_file: str | None):
    cmd = [
        sys.executable,
        str(ROOT_DIR / "scripts" / "eval_nonuquant_style_ppl.py"),
        "--model-path", model,
        "--quantized-path", quantized_path,
        "--model-name", Path(quantized_path).name,
        "--tokenizer-path", model,
        "--dtype", dtype,
        "--stride", str(stride),
        "--max-length", str(max_length),
        "--c4-samples", str(c4_samples),
    ]
    if output_file:
        cmd.extend(["--output-file", output_file])
    if datasets:
        cmd.extend(["--datasets", *datasets])
    logging.info("Running evaluator: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    load_project_dotenv(verbose=True)
    _setup_logging()

    parser = argparse.ArgumentParser(description="Full-model flash1dkmeans reference quantization with optional PPL eval")
    parser.add_argument("--model", default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--bits", type=int, default=3)
    parser.add_argument("--cache-dir", default="./cache")
    parser.add_argument("--dataset", default="redpajama")
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--num-examples", type=int, default=1024)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--cpu-count", type=int, default=None)
    parser.add_argument("--overwrite-quantize", action="store_true")
    parser.add_argument("--run-eval", action="store_true")
    parser.add_argument("--eval-datasets", nargs="+", default=["wikitext2", "c4"])
    parser.add_argument("--eval-dtype", default="float32")
    parser.add_argument("--eval-stride", type=int, default=512)
    parser.add_argument("--eval-max-length", type=int, default=2048)
    parser.add_argument("--eval-c4-samples", type=int, default=2000)
    parser.add_argument("--eval-output-file", default="")
    args = parser.parse_args()

    model_name = args.model.split("/")[-1]
    output_dir = args.output_dir or _default_output_dir(
        args.cache_dir,
        model_name,
        args.bits,
        args.dataset,
        args.num_examples,
        args.seq_len,
    )

    quantized_path = quantize_reference_kmeans(
        model=args.model,
        bits=args.bits,
        cache_dir=args.cache_dir,
        dataset=args.dataset,
        seq_len=args.seq_len,
        num_examples=args.num_examples,
        output_dir=output_dir,
        cpu_count=args.cpu_count,
        overwrite_quantize=args.overwrite_quantize,
    )

    summary = {"quantized_path": quantized_path}
    print(json.dumps(summary, indent=2))

    if args.run_eval:
        maybe_run_eval(
            model=args.model,
            quantized_path=quantized_path,
            datasets=args.eval_datasets,
            dtype=args.eval_dtype,
            stride=args.eval_stride,
            max_length=args.eval_max_length,
            c4_samples=args.eval_c4_samples,
            output_file=args.eval_output_file or None,
        )


if __name__ == "__main__":
    main()
