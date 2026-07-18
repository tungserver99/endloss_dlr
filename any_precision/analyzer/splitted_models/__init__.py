from .llama import SplittedLlamaModel

__all__ = [
    "SplittedLlamaModel",
    "SplittedQwen3Model",
    "SplittedGemma3TextModel",
]


def __getattr__(name):
    if name == "SplittedQwen3Model":
        from .qwen3 import SplittedQwen3Model
        return SplittedQwen3Model
    if name == "SplittedGemma3TextModel":
        from .gemma3 import SplittedGemma3TextModel
        return SplittedGemma3TextModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
