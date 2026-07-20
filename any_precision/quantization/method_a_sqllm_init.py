from __future__ import annotations

import gc
import logging
import os
from pathlib import Path

import numba
import numpy as np
import torch
from tqdm.auto import tqdm

from .method_a_gradient import (
    _disable_checkpointing_for_stats,
    _enable_checkpointing_for_stats,
    _iter_layer_chunks,
    _restore_float_dtypes,
    _snapshot_float_dtypes,
    model_identity,
    tensor_fingerprint,
)


def collect_sqllm_importance(
    analyzer,
    tokens: torch.Tensor,
    output_folder: str,
    device: str,
    layer_chunk_size: int = 1,
    overwrite: bool = False,
) -> None:
    """Collect SqueezeLLM's squared weight-gradient importance layer-wise."""
    root = Path(output_folder)
    config = {
        "schema": 1,
        "source": "sqllm_squared_weight_gradient",
        "num_examples": int(tokens.shape[0]),
        "seq_len": int(tokens.shape[1]),
        "actual_batch_size": 1,
        "model": model_identity(analyzer),
        "tokens_sha256": tensor_fingerprint(tokens),
    }
    expected = [root / f"l{i}.pt" for i in range(analyzer.num_layers)]
    config_path = root / "_config.pt"
    if not overwrite and expected and all(path.exists() for path in expected):
        if config_path.exists() and torch.load(config_path, map_location="cpu") == config:
            logging.info("Reusing SqueezeLLM importance cache at %s", root)
            return
    root.mkdir(parents=True, exist_ok=True)

    model = analyzer.model
    model.to(device).eval()
    original_param_dtypes, original_buffer_dtypes = _snapshot_float_dtypes(model)
    model.bfloat16()
    _enable_checkpointing_for_stats(model)
    layers = analyzer.get_layers()
    original_requires_grad = {id(param): param.requires_grad for param in model.parameters()}

    for chunk_start, layer_chunk in _iter_layer_chunks(layers, max(1, layer_chunk_size)):
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
            range(0, tokens.shape[0]),
            desc=f"SqueezeLLM init importance L{chunk_start}-{chunk_start + len(layer_chunk) - 1}",
            ascii=True,
            leave=False,
            mininterval=5.0,
        ):
            batch = tokens[start:start + 1].to(device)
            model(input_ids=batch, labels=batch, use_cache=False).loss.backward()
            del batch
        for hook in hooks:
            hook.remove()
        for local_idx, layer in enumerate(layer_chunk):
            layer_idx = chunk_start + local_idx
            layer_importance = {}
            for module_name, module in analyzer.get_modules(layer).items():
                if module.weight.grad is None:
                    raise RuntimeError(
                        f"Missing SqueezeLLM importance for layer={layer_idx}, module={module_name}"
                    )
                layer_importance[module_name] = module.weight.grad.detach().float().cpu()
            torch.save(layer_importance, root / f"l{layer_idx}.pt")
        model.zero_grad(set_to_none=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for param in model.parameters():
        param.requires_grad_(original_requires_grad[id(param)])
    _restore_float_dtypes(model, original_param_dtypes, original_buffer_dtypes)
    _disable_checkpointing_for_stats(model)
    model.cpu().eval()
    torch.save(config, config_path)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_sqllm_initialization(
    analyzer,
    importance_folder: str,
    output_folder: str,
    bits: int,
    cpu_count: int | None = None,
    overwrite: bool = False,
) -> None:
    """Call the repository's SqueezeLLM scalar quantizer to create q0."""
    from .quantize import _seed_and_upscale_layer
    
    output = Path(output_folder)
    importance_config = torch.load(
        Path(importance_folder) / "_config.pt", map_location="cpu"
    )
    init_config = {
        "schema": 1,
        "source": "sqllm_initialization",
        "bits": int(bits),
        "importance": importance_config,
    }
    config_path = output / "_config.pt"
    cache_matches = (
        not overwrite
        and config_path.exists()
        and torch.load(config_path, map_location="cpu") == init_config
    )
    weights_dir = output / "weights"
    lut_dir = output / f"lut_{bits}"
    weights_dir.mkdir(parents=True, exist_ok=True)
    lut_dir.mkdir(parents=True, exist_ok=True)
    cpu_count = int(cpu_count or os.cpu_count() or 1)
    numba.set_num_threads(max(1, cpu_count))

    for layer_idx in tqdm(range(analyzer.num_layers), desc="Building SqueezeLLM q0"):
        weights_path = weights_dir / f"l{layer_idx}.pt"
        lut_path = lut_dir / f"l{layer_idx}.pt"
        if cache_matches and weights_path.exists() and lut_path.exists():
            continue
        importance = torch.load(
            Path(importance_folder) / f"l{layer_idx}.pt", map_location="cpu"
        )
        fp_weights = analyzer.get_layer_weights(layer_idx)
        module_names = list(analyzer.get_layer_module_paths(layer_idx).keys())
        importance_arrays = [importance[name].float().numpy() for name in module_names]
        weight_arrays = [fp_weights[name].float().numpy() for name in module_names]
        luts_by_module, labels_by_module = _seed_and_upscale_layer(
            importance_arrays,
            weight_arrays,
            int(bits),
            int(bits),
            1,
            random_state=0,
        )
        torch.save(
            {name: labels_by_module[idx].astype(np.uint8) for idx, name in enumerate(module_names)},
            weights_path,
        )
        torch.save(
            {name: luts_by_module[idx][0].astype(np.float16) for idx, name in enumerate(module_names)},
            lut_path,
        )
        del importance, fp_weights, importance_arrays, weight_arrays, luts_by_module, labels_by_module
        gc.collect()
    torch.save(init_config, config_path)
