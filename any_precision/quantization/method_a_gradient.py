from __future__ import annotations

from collections import defaultdict
import hashlib
import sys

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


def _enable_checkpointing_for_stats(model):
    model._method_a_checkpointing_was_enabled = bool(
        getattr(model, "is_gradient_checkpointing", False)
    )
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model._method_a_original_use_cache = model.config.use_cache
        model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    model._method_a_input_grads_were_enabled = hasattr(model, "_require_grads_hook")
    if (
        hasattr(model, "enable_input_require_grads")
        and not model._method_a_input_grads_were_enabled
    ):
        model.enable_input_require_grads()


def _disable_checkpointing_for_stats(model):
    if (
        hasattr(model, "gradient_checkpointing_disable")
        and not getattr(model, "_method_a_checkpointing_was_enabled", False)
    ):
        model.gradient_checkpointing_disable()
    if hasattr(model, "_method_a_checkpointing_was_enabled"):
        del model._method_a_checkpointing_was_enabled
    if (
        hasattr(model, "disable_input_require_grads")
        and not getattr(model, "_method_a_input_grads_were_enabled", False)
    ):
        model.disable_input_require_grads()
    if hasattr(model, "_method_a_input_grads_were_enabled"):
        del model._method_a_input_grads_were_enabled
    if hasattr(model, "config") and hasattr(model, "_method_a_original_use_cache"):
        model.config.use_cache = model._method_a_original_use_cache
        del model._method_a_original_use_cache


def _snapshot_float_dtypes(model):
    param_dtypes = {}
    buffer_dtypes = {}
    for name, param in model.named_parameters():
        if torch.is_floating_point(param):
            param_dtypes[name] = param.dtype
    for name, buffer in model.named_buffers():
        if torch.is_floating_point(buffer):
            buffer_dtypes[name] = buffer.dtype
    return param_dtypes, buffer_dtypes


def _restore_float_dtypes(model, param_dtypes, buffer_dtypes):
    for name, param in model.named_parameters():
        target_dtype = param_dtypes.get(name)
        if target_dtype is not None and param.dtype != target_dtype:
            param.data = param.data.to(dtype=target_dtype)
            if param.grad is not None and torch.is_floating_point(param.grad):
                param.grad.data = param.grad.data.to(dtype=target_dtype)
    for name, buffer in model.named_buffers():
        target_dtype = buffer_dtypes.get(name)
        if target_dtype is not None and buffer.dtype != target_dtype:
            module = model
            parts = name.split('.')
            for part in parts[:-1]:
                module = getattr(module, part)
            module._buffers[parts[-1]] = buffer.to(dtype=target_dtype)


def _iter_layer_chunks(layers, chunk_size: int):
    for start in range(0, len(layers), chunk_size):
        yield start, layers[start:start + chunk_size]


def tensor_fingerprint(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def model_identity(analyzer) -> str:
    config = getattr(analyzer.model, "config", None)
    return str(getattr(config, "_name_or_path", analyzer.model.__class__.__name__))


def _progress_kwargs() -> dict:
    return {
        "ascii": True,
        "leave": False,
        "dynamic_ncols": False,
        "ncols": 100,
        "mininterval": 5.0,
        "maxinterval": 30.0,
        "file": sys.stdout,
    }


def collect_nll_gradients(analyzer, tokens: torch.Tensor, batch_size: int, device: str, layer_chunk_size: int = 8) -> dict[int, dict[str, torch.Tensor]]:
    model = analyzer.model
    model.to(device)
    model.eval()
    original_param_dtypes, original_buffer_dtypes = _snapshot_float_dtypes(model)
    model = model.bfloat16()
    _enable_checkpointing_for_stats(model)

    if model.device.type != "cuda" and torch.cuda.device_count() == 1:
        model.cuda()

    layers = analyzer.get_layers()
    grads: dict[int, dict[str, torch.Tensor]] = defaultdict(dict)

    original_requires_grad = {id(param): param.requires_grad for param in model.parameters()}

    for chunk_start, layer_chunk in _iter_layer_chunks(layers, layer_chunk_size):
        target_weights = set()
        for layer in layer_chunk:
            for module in analyzer.get_modules(layer).values():
                target_weights.add(module.weight)

        for param in model.parameters():
            param.requires_grad_(param in target_weights)

        model.zero_grad(set_to_none=True)
        total_pred_tokens = 0
        for start in tqdm(
            range(0, tokens.shape[0], batch_size),
            desc=f"Collecting NLL gradients L{chunk_start}-{chunk_start + len(layer_chunk) - 1}",
            **_progress_kwargs(),
        ):
            batch = tokens[start:start + batch_size].to(model.device)
            logits = model(input_ids=batch, use_cache=False).logits[:, :-1, :].float()
            labels = batch[:, 1:]
            loss_sum = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="sum")
            loss_sum.backward()
            total_pred_tokens += labels.numel()
            del logits, labels, loss_sum, batch

        scale = 1.0 / max(1, total_pred_tokens)
        for local_idx, layer in enumerate(layer_chunk):
            layer_idx = chunk_start + local_idx
            for module_name, module in analyzer.get_modules(layer).items():
                grad = module.weight.grad
                if grad is None:
                    raise RuntimeError(f"Missing gradient for layer {layer_idx} module {module_name}")
                grads[layer_idx][module_name] = grad.detach().float().mul_(scale).cpu()

        model.zero_grad(set_to_none=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for param in model.parameters():
        param.requires_grad_(original_requires_grad[id(param)])

    _restore_float_dtypes(model, original_param_dtypes, original_buffer_dtypes)
    _disable_checkpointing_for_stats(model)
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return grads



