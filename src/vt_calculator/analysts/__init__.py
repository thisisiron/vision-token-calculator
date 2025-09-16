from .analyst import Qwen2VLAnalyst, Qwen2_5_VLAnalyst, InternVLAnalyst
from transformers import AutoProcessor, AutoConfig


SUPPORTED_MODELS: set[str] = {
    "qwen2.5-vl",
    "internvl3",
}

# Mapping from short model name to Hugging Face repository id
MODEL_TO_HF_ID: dict[str, str] = {
    "qwen2.5-vl": "Qwen/Qwen2.5-VL-3B-Instruct",
    "internvl3": "OpenGVLab/InternVL3-1B-hf",
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
    """Factory that builds the correct analyst for a given short model name."""
    model_id = map_model_id(model_name)

    # Choose analyst class by family
    key = model_name.strip().lower()
    is_internvl = key.startswith("internvl3")

    processor = AutoProcessor.from_pretrained(model_id)
    if is_internvl:
        config = AutoConfig.from_pretrained(model_id)
        analyst = InternVLAnalyst(processor, config)
    else:
        analyst = Qwen2_5_VLAnalyst(processor)

    return analyst


__all__ = [
    "Qwen2VLAnalyst",
    "Qwen2_5_VLAnalyst",
    "InternVLAnalyst",
    "load_analyst",
    "map_model_id",
    "SUPPORTED_MODELS",
    "MODEL_TO_HF_ID",
    "DEFAULT_MODEL",
]
