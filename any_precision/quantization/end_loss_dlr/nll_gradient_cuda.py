from __future__ import annotations

from collections import defaultdict

import torch
from tqdm.auto import tqdm


def _enable_checkpointing_for_stats(model):
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()


def _disable_checkpointing_for_stats(model):
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = True


def _iter_layer_chunks(layers, chunk_size: int):
    for start in range(0, len(layers), chunk_size):
        yield start, layers[start:start + chunk_size]


def collect_nll_gradients(analyzer, tokens: torch.Tensor, batch_size: int, device: str, layer_chunk_size: int = 8) -> dict[int, dict[str, torch.Tensor]]:
    model = analyzer.model
    model.to(device)
    model.eval()
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
        steps = 0
        for start in tqdm(
            range(0, tokens.shape[0], batch_size),
            desc=f"Collecting NLL gradients L{chunk_start}-{chunk_start + len(layer_chunk) - 1}",
            dynamic_ncols=True,
            mininterval=5.0,
            maxinterval=30.0,
            ascii=True,
            leave=True,
        ):
            batch = tokens[start:start + batch_size].to(model.device)
            outputs = model(input_ids=batch, labels=batch, use_cache=False)
            outputs.loss.backward()
            steps += 1
            del outputs, batch

        scale = 1.0 / max(1, steps)
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

    _disable_checkpointing_for_stats(model)
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return grads
