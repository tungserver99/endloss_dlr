from __future__ import annotations

import gc
import logging
import os
from pathlib import Path

import numba
import torch
from tqdm.auto import tqdm

from .method_a_gradient import (
    _disable_checkpointing_for_stats,
    _enable_checkpointing_for_stats,
    _iter_layer_chunks,
    _progress_kwargs,
    _restore_float_dtypes,
    _snapshot_float_dtypes,
    model_identity,
    tensor_fingerprint,
)
from .quantize import _save_results, _seed_and_upscale_layer


def _q0_complete(output_folder: str | Path, bits: int, num_layers: int) -> bool:
    output = Path(output_folder)
    return all(
        (output / "weights" / f"l{layer_idx}.pt").exists()
        and (output / f"lut_{bits}" / f"l{layer_idx}.pt").exists()
        for layer_idx in range(num_layers)
    )


def collect_sqllm_squared_gradients_chunked(
    analyzer,
    tokens: torch.Tensor,
    output_folder: str,
    device: str,
    layer_chunk_size: int = 8,
    overwrite: bool = False,
) -> None:
    """Collect SqueezeLLM squared weight gradients in DLR-like layer chunks.

    This keeps SqueezeLLM's original squared-weight-gradient signal and lets
    PyTorch compute weight gradients normally, but only for a chunk of layers
    at a time so GPU memory does not hold gradients for the whole model.
    """
    root = Path(output_folder)
    layer_chunk_size = max(1, int(layer_chunk_size))
    config = {
        "schema": 1,
        "source": "sqllm_squared_weight_gradient_layer_chunked",
        "num_examples": int(tokens.shape[0]),
        "seq_len": int(tokens.shape[1]),
        "actual_batch_size": 1,
        "layer_chunk_size": int(layer_chunk_size),
        "model": model_identity(analyzer),
        "tokens_sha256": tensor_fingerprint(tokens),
    }
    config_path = root / "_config.pt"
    expected = [root / f"l{layer_idx}.pt" for layer_idx in range(analyzer.num_layers)]
    cache_matches = (
        not overwrite
        and config_path.exists()
        and torch.load(config_path, map_location="cpu") == config
    )
    if cache_matches and all(path.exists() for path in expected):
        logging.info("Reusing layer-chunked SqueezeLLM squared-gradient cache at %s", root)
        return

    root.mkdir(parents=True, exist_ok=True)
    torch.save(config, config_path)

    model = analyzer.model
    model.to(device).eval()
    original_param_dtypes, original_buffer_dtypes = _snapshot_float_dtypes(model)
    model.bfloat16()
    _enable_checkpointing_for_stats(model)
    if model.device.type != "cuda" and torch.cuda.device_count() == 1:
        model.cuda()

    layers = analyzer.get_layers()
    original_requires_grad = {id(param): param.requires_grad for param in model.parameters()}

    try:
        for chunk_start, layer_chunk in _iter_layer_chunks(layers, layer_chunk_size):
            chunk_paths = [root / f"l{chunk_start + idx}.pt" for idx in range(len(layer_chunk))]
            if cache_matches and all(path.exists() for path in chunk_paths):
                continue

            target_weights = {
                module.weight
                for layer in layer_chunk
                for module in analyzer.get_modules(layer).values()
            }
            for param in model.parameters():
                param.requires_grad_(param in target_weights)

            hooks = [weight.register_hook(lambda grad: grad.square()) for weight in target_weights]
            model.zero_grad(set_to_none=True)
            for start in tqdm(
                range(tokens.shape[0]),
                desc=f"SqueezeLLM gradients L{chunk_start}-{chunk_start + len(layer_chunk) - 1}",
                **_progress_kwargs(),
            ):
                batch = tokens[start:start + 1].to(model.device)
                model(input_ids=batch, labels=batch, use_cache=False).loss.backward()
                if start == 0:
                    missing = [
                        name
                        for layer in layer_chunk
                        for name, module in analyzer.get_modules(layer).items()
                        if module.weight.grad is None
                    ]
                    if missing:
                        raise RuntimeError(
                            "Missing SqueezeLLM squared-gradient hooks after first backward: "
                            f"{missing[:8]}"
                        )
                del batch

            for hook in hooks:
                hook.remove()

            for local_idx, layer in enumerate(layer_chunk):
                layer_idx = chunk_start + local_idx
                gradients = {}
                for module_name, module in analyzer.get_modules(layer).items():
                    grad = module.weight.grad
                    if grad is None:
                        raise RuntimeError(
                            f"Missing SqueezeLLM gradient for layer={layer_idx}, module={module_name}"
                        )
                    gradients[module_name] = grad.detach().float().cpu()
                torch.save(gradients, root / f"l{layer_idx}.pt")

            model.zero_grad(set_to_none=True)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        for param in model.parameters():
            param.requires_grad_(original_requires_grad[id(param)])
        _restore_float_dtypes(model, original_param_dtypes, original_buffer_dtypes)
        _disable_checkpointing_for_stats(model)
        model.cpu().eval()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def build_sqllm_q0_from_chunked_gradients(
    analyzer,
    gradients_folder: str,
    output_folder: str,
    bits: int,
    cpu_count: int | None = None,
    overwrite: bool = False,
) -> None:
    """Run the original SqueezeLLM seed/upscale quantizer from chunked gradients."""
    output = Path(output_folder)
    if not overwrite and _q0_complete(output, bits, analyzer.num_layers):
        logging.info("Reusing layer-chunked SqueezeLLM q0 cache at %s", output)
        return

    cpu_count = int(cpu_count or os.cpu_count() or 1)
    numba.set_num_threads(max(1, cpu_count))
    logging.info("Building q0 with original SqueezeLLM quantizer from layer-chunked gradients")
    for layer_idx in tqdm(range(analyzer.num_layers), desc="Building SqueezeLLM q0"):
        weights_path = output / "weights" / f"l{layer_idx}.pt"
        lut_path = output / f"lut_{bits}" / f"l{layer_idx}.pt"
        if not overwrite and weights_path.exists() and lut_path.exists():
            continue

        gradients = torch.load(Path(gradients_folder) / f"l{layer_idx}.pt", map_location="cpu")
        fp_weights = analyzer.get_layer_weights(layer_idx)
        module_names = list(analyzer.get_layer_module_paths(layer_idx).keys())
        layer_gradients = [gradients[name].float().numpy() for name in module_names]
        layer_modules = [fp_weights[name].float().numpy() for name in module_names]
        luts_by_bit_by_module, parent_weights = _seed_and_upscale_layer(
            layer_gradients,
            layer_modules,
            int(bits),
            int(bits),
            1,
        )
        _save_results(
            str(output),
            int(bits),
            int(bits),
            module_names,
            luts_by_bit_by_module,
            parent_weights,
            layer_idx,
        )
        del gradients, fp_weights, layer_gradients, layer_modules
        gc.collect()


def ensure_chunked_sqllm_initialization(
    analyzer,
    tokens: torch.Tensor,
    gradients_folder: str,
    output_folder: str,
    bits: int,
    device: str,
    layer_chunk_size: int = 8,
    cpu_count: int | None = None,
    overwrite: bool = False,
) -> None:
    if not overwrite and _q0_complete(output_folder, bits, analyzer.num_layers):
        logging.info("Reusing layer-chunked SqueezeLLM q0 cache at %s", output_folder)
        return
    collect_sqllm_squared_gradients_chunked(
        analyzer=analyzer,
        tokens=tokens,
        output_folder=gradients_folder,
        device=device,
        layer_chunk_size=layer_chunk_size,
        overwrite=overwrite,
    )
    build_sqllm_q0_from_chunked_gradients(
        analyzer=analyzer,
        gradients_folder=gradients_folder,
        output_folder=output_folder,
        bits=bits,
        cpu_count=cpu_count,
        overwrite=overwrite,
    )
