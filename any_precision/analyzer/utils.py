import torch
from transformers import AutoModelForCausalLM, PreTrainedModel, AutoTokenizer, PreTrainedTokenizerBase
from .splitted_models import SplittedLlamaModel


def _is_qwen35_name(name):
    if not isinstance(name, str):
        return False
    lowered = name.lower()
    return "qwen3_5" in lowered or "qwen3.5" in lowered


def _is_gemma2_name(name):
    if not isinstance(name, str):
        return False
    lowered = name.lower()
    return "gemma2" in lowered or "gemma-2" in lowered


def get_architecture_name(model):
    architectures = getattr(model.config, "architectures", None)
    if isinstance(architectures, (list, tuple)) and len(architectures) == 1 and architectures[0]:
        architecture = architectures[0]
        if _is_qwen35_name(architecture):
            return "Qwen3_5ForCausalLM"
        if _is_gemma2_name(architecture):
            return "Gemma2ForCausalLM"
        if "Qwen3" in architecture:
            return "Qwen3ForCausalLM"
        return architecture
    if isinstance(architectures, str) and architectures:
        if _is_qwen35_name(architectures):
            return "Qwen3_5ForCausalLM"
        if _is_gemma2_name(architectures):
            return "Gemma2ForCausalLM"
        if "Qwen3" in architectures:
            return "Qwen3ForCausalLM"
        return architectures

    model_type = getattr(model.config, "model_type", None)
    if _is_qwen35_name(model_type):
        return "Qwen3_5ForCausalLM"
    if _is_gemma2_name(model_type) or model_type == "gemma2":
        return "Gemma2ForCausalLM"
    if isinstance(model_type, str) and model_type.startswith("qwen3"):
        return "Qwen3ForCausalLM"
    if model_type == "llama":
        return "LlamaForCausalLM"
    if model_type == "gemma3":
        return "Gemma3ForConditionalGeneration"

    class_name = model.__class__.__name__
    if _is_qwen35_name(class_name):
        return "Qwen3_5ForCausalLM"
    if _is_gemma2_name(class_name):
        return "Gemma2ForCausalLM"
    if "Qwen3" in class_name:
        return "Qwen3ForCausalLM"
    if class_name and class_name != "AutoModelForCausalLM":
        return class_name

    inner_model = getattr(model, "model", None)
    if inner_model is not None:
        inner_class_name = inner_model.__class__.__name__
        if inner_class_name:
            if _is_qwen35_name(inner_class_name):
                return "Qwen3_5ForCausalLM"
            if _is_gemma2_name(inner_class_name):
                return "Gemma2ForCausalLM"
            if "Qwen3" in inner_class_name:
                return "Qwen3ForCausalLM"
            if "Llama" in inner_class_name:
                return "LlamaForCausalLM"
            if "Gemma3" in inner_class_name:
                return "Gemma3ForConditionalGeneration"

    raise ValueError("Unable to infer model architecture from config/model class")


def load_model(model_str_or_model, dtype=torch.float16):
    """Returns a model from a string or a model object. If a string is passed, it will be loaded from the HuggingFace"""
    if isinstance(model_str_or_model, str):
        # Qwen / Gemma models are more stable in bfloat16
        if not "llama" in model_str_or_model.lower():
            dtype = torch.bfloat16

        model = AutoModelForCausalLM.from_pretrained(
            model_str_or_model,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map='cpu',
        )
    else:
        assert isinstance(model_str_or_model, PreTrainedModel), "model must be a string or a PreTrainedModel"
        model = model_str_or_model
    return model


def dispatch_model(model):
    architecture = get_architecture_name(model)
    if architecture == 'LlamaForCausalLM':
        model.model.__class__ = SplittedLlamaModel
        model.model.config.use_cache = False
        model.model.set_devices()
        model.lm_head.to("cuda:0")
        return model
    elif architecture == 'Qwen3ForCausalLM':
        from .splitted_models import SplittedQwen3Model
        model.model.__class__ = SplittedQwen3Model
        model.model.config.use_cache = False
        model.model.set_devices()
        model.lm_head.to("cuda:0")
        return model
    elif architecture == 'Gemma3ForConditionalGeneration':
        from .splitted_models import SplittedGemma3TextModel
        model.to("cuda:0")
        model.model.language_model.__class__ = SplittedGemma3TextModel
        model.model.language_model.config.use_cache = False
        model.model.language_model.set_devices()
        model.lm_head.to("cuda:0")
        return model
    elif architecture == 'Gemma2ForCausalLM':
        raise NotImplementedError(
            "Gemma 2 is supported through the single-GPU analyzer path only for now. "
            "Please run with one visible GPU when extracting gradients/calibration data."
        )
    elif architecture == 'Qwen3_5ForCausalLM':
        raise NotImplementedError(
            "Qwen 3.5 is supported through the single-GPU analyzer path only for now. "
            "Please run with one visible GPU when extracting gradients/calibration data."
        )
    else:
        raise NotImplementedError(f"Model {architecture} is not supported")


def load_tokenizer(model_str_or_model_or_tokenizer):
    """Returns a tokenizer from the model string or model object or tokenizer object"""
    if isinstance(model_str_or_model_or_tokenizer, str):
        model_str = model_str_or_model_or_tokenizer
        return AutoTokenizer.from_pretrained(model_str, trust_remote_code=True)
    elif isinstance(model_str_or_model_or_tokenizer, PreTrainedModel):
        model_str = model_str_or_model_or_tokenizer.name_or_path
        return AutoTokenizer.from_pretrained(model_str, trust_remote_code=True)
    else:
        assert isinstance(model_str_or_model_or_tokenizer, PreTrainedTokenizerBase), \
            f"Unsupported type for model_str_or_model_or_tokenizer: {type(model_str_or_model_or_tokenizer)}"
        return model_str_or_model_or_tokenizer
