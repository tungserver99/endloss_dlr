def __getattr__(name):
    if name == "any_precision_quantize":
        from .main import any_precision_quantize

        return any_precision_quantize
    if name == "layerwise_nuq":
        from .layerwise_main import layerwise_nuq

        return layerwise_nuq
    if name == "full_nuq":
        from .full_main import full_nuq

        return full_nuq
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["any_precision_quantize", "layerwise_nuq", "full_nuq"]
