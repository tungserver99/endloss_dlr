__all__ = ["modules", "quantization", "AnyPrecisionForCausalLM"]


def __getattr__(name):
    if name == "modules":
        from . import modules
        return modules
    if name == "quantization":
        from . import quantization
        return quantization
    if name == "AnyPrecisionForCausalLM":
        from .modules import AnyPrecisionForCausalLM
        return AnyPrecisionForCausalLM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
