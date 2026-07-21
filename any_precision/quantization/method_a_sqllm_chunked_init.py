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
    layer_chunk_size: int = 1,
    overwrite: bool = False,
) -> None:
    """Collect SqueezeLLM squared weight gradients with DLR-style all-layer hooks.

    Each calibration sample contributes (d loss / d W)^2, matching SqueezeLLM's
    Fisher/importance signal. The memory change is that weight gradients are
    formed in module output-gradient hooks and accumulated on CPU, so PyTorch
    does not keep full-model param.grad tensors on GPU.
    """
    del layer_chunk_size  # kept in the public signature for Method A wiring compatibility
    root = Path(output_folder)
    config = {
        "schema": 1,
        "source": "sqllm_squared_weight_gradient_all_layers_cpu_accum",
        "num_examples": int(tokens.shape[0]),
        "seq_len": int(tokens.shape[1]),
        "actual_batch_size": 1,
        "model": model_identity(analyzer),
        "tokens_sha256": tensor_fingerprint(tokens),
    }
    config_path = root / "_config.pt"
    expected = [root / f"l{layer_idx}.pt" for layer_idx in range(analyzer.num_layers)]
    if (
        not overwrite
        and all(path.exists() for path in expected)
        and config_path.exists()
        and torch.load(config_path, map_location="cpu") == config
    ):
        logging.info("Reusing DLR-style SqueezeLLM squared-gradient cache at %s", root)
        return

    root.mkdir(parents=True, exist_ok=True)
    model = analyzer.model
    model.to(device).eval()
    original_param_dtypes, original_buffer_dtypes = _snapshot_float_dtypes(model)
    model.bfloat16()
    _enable_checkpointing_for_stats(model)
    if model.device.type != "cuda" and torch.cuda.device_count() == 1:
        model.cuda()

    layers = analyzer.get_layers()
    original_requires_grad = {id(param): param.requires_grad for param in model.parameters()}
    for param in model.parameters():
        param.requires_grad_(False)

    accumulators: dict[tuple[int, str], torch.Tensor] = {}
    update_counts: dict[tuple[int, str], int] = {}
    hooks = []
    for layer_idx, layer in enumerate(layers):
        for module_name, module in analyzer.get_modules(layer).items():
            key = (layer_idx, module_name)
            accumulators[key] = torch.zeros_like(module.weight, dtype=torch.float32, device="cpu")
            update_counts[key] = 0

            def make_hook(acc_key):
                def forward_hook(_module, inputs, output):
                    if not inputs or not isinstance(inputs[0], torch.Tensor):
                        return
                    if not isinstance(output, torch.Tensor) or not output.requires_grad:
                        return
                    module_inputs = inputs[0].detach()

                    def output_gradient_hook(gradient):
                        grad_2d = gradient.reshape(-1, gradient.shape[-1]).float()
                        input_2d = module_inputs.reshape(-1, module_inputs.shape[-1]).float()
                        weight_grad = grad_2d.transpose(0, 1).matmul(input_2d)
                        accumulators[acc_key].add_(weight_grad.square().cpu())
                        update_counts[acc_key] += 1
                        del grad_2d, input_2d, weight_grad

                    output.register_hook(output_gradient_hook)
                return forward_hook

            hooks.append(module.register_forward_hook(make_hook(key)))

    try:
        for start in tqdm(
            range(tokens.shape[0]),
            desc="SqueezeLLM gradients (DLR-style all layers)",
            **_progress_kwargs(),
        ):
            batch = tokens[start:start + 1].to(model.device)
            model(input_ids=batch, labels=batch, use_cache=False).loss.backward()
            model.zero_grad(set_to_none=True)
            del batch
    finally:
        for hook in hooks:
            hook.remove()

    expected_updates = int(tokens.shape[0])
    for layer_idx, layer in enumerate(layers):
        gradients = {}
        for module_name in analyzer.get_modules(layer):
            key = (layer_idx, module_name)
            if update_counts[key] != expected_updates:
                raise RuntimeError(
                    f"Expected {expected_updates} SqueezeLLM updates at "
                    f"layer={layer_idx} module={module_name}; got {update_counts[key]}"
                )
            gradients[module_name] = accumulators[key]
        torch.save(gradients, root / f"l{layer_idx}.pt")

    for param in model.parameters():
        param.requires_grad_(original_requires_grad[id(param)])
    _restore_float_dtypes(model, original_param_dtypes, original_buffer_dtypes)
    _disable_checkpointing_for_stats(model)
    model.cpu().eval()
    torch.save(config, config_path)
    del accumulators
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
        logging.info("Reusing DLR-style SqueezeLLM q0 cache at %s", output)
        return

    cpu_count = int(cpu_count or os.cpu_count() or 1)
    numba.set_num_threads(max(1, cpu_count))
    logging.info("Building q0 with original SqueezeLLM quantizer from DLR-style gradients")
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
    layer_chunk_size: int = 1,
    cpu_count: int | None = None,
    overwrite: bool = False,
) -> None:
    if not overwrite and _q0_complete(output_folder, bits, analyzer.num_layers):
        logging.info("Reusing DLR-style SqueezeLLM q0 cache at %s", output_folder)
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
