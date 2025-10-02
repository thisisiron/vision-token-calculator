from .analyst import (
    Qwen2VLAnalyst,
    Qwen2_5_VLAnalyst,
    InternVLAnalyst,
    LLaVAAnalyst,
    LLaVANextAnalyst,
    LlavaOnevisionAnalyst,
)
from transformers import AutoProcessor, AutoConfig
from typing import Callable, Dict, Tuple


SUPPORTED_MODELS: set[str] = {
    "llava",
    "llava-next",
    "llava-onevision",
    "qwen2-vl",
    "qwen2.5-vl",
    "internvl3",
}

# Mapping from short model name to Hugging Face repository id
MODEL_TO_HF_ID: dict[str, str] = {
    "qwen2.5-vl": "Qwen/Qwen2.5-VL-3B-Instruct",
    "qwen2-vl": "Qwen/Qwen2-VL-2B-Instruct",
    "internvl3": "OpenGVLab/InternVL3-1B-hf",
    "llava": "llava-hf/llava-1.5-7b-hf",
    "llava-next": "llava-hf/llava-v1.6-mistral-7b-hf",
    "llava-onevision": "llava-hf/llava-onevision-qwen2-7b-ov-hf",
}

# Default short model name used across the app when none is provided
DEFAULT_MODEL: str = "qwen2.5-vl"


def map_model_id(model_name: str) -> str:
    """Map a supported short model name to its HF id.

    Args:
        model_name (str): Short model name such as "qwen2.5-vl" or "internvl3".

    Returns:
        str: Hugging Face repository id for the given model name.
    """
    key = model_name.strip().lower()
    if key not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model: {model_name}")
    return MODEL_TO_HF_ID[key]


def load_analyst(model_name: str = DEFAULT_MODEL):
    """Factory that builds the correct analyst for a given short model name.

    Selection is handled via a small registry mapping so it is easy to extend
    with additional models without modifying conditional logic.
    """
    model_id = map_model_id(model_name)

    key = model_name.strip().lower()

    # Analyst builders receive (processor, config) and must return an instance
    # of a VLMAnalyst subclass. The config argument may be ignored when not
    # needed by the analyst.
    ANALYST_REGISTRY: Dict[str, Tuple[Callable, bool]] = {
        "qwen2.5-vl": (lambda proc, cfg: Qwen2_5_VLAnalyst(proc), False),
        "qwen2-vl": (lambda proc, cfg: Qwen2VLAnalyst(proc), False),
        "internvl3": (lambda proc, cfg: InternVLAnalyst(proc, cfg), True),
        "llava": (lambda proc, cfg: LLaVAAnalyst(proc), False),
        "llava-next": (lambda proc, cfg: LLaVANextAnalyst(proc), False),
        "llava-onevision": (lambda proc, cfg: LlavaOnevisionAnalyst(proc), False),
    }

    if key not in ANALYST_REGISTRY:
        raise ValueError(f"No analyst registered for model: {model_name}")

    builder, needs_config = ANALYST_REGISTRY[key]

    processor = AutoProcessor.from_pretrained(model_id)
    config = AutoConfig.from_pretrained(model_id) if needs_config else None

    return builder(processor, config)


__all__ = [
    "Qwen2VLAnalyst",
    "Qwen2_5_VLAnalyst",
    "InternVLAnalyst",
    "LLaVAAnalyst",
    "LLaVANextAnalyst",
    "LlavaOnevisionAnalyst",
    "load_analyst",
    "map_model_id",
    "SUPPORTED_MODELS",
    "MODEL_TO_HF_ID",
    "DEFAULT_MODEL",
]
