from .analyst import (
    Qwen2VLAnalyst,
    Qwen2_5_VLAnalyst,
    Qwen3VLAnalyst,
    InternVLAnalyst,
    LLaVAAnalyst,
    LLaVANextAnalyst,
    LlavaOnevisionAnalyst,
    DeepSeekOCRAnalyst,
    Phi4MultimodalAnalyst,
)
from typing import Optional
from transformers import AutoProcessor, AutoConfig
from typing import Callable, Dict, Optional, Tuple


MODEL_TO_HF_ID: dict[str, Optional[str]] = {
    "qwen2.5-vl": "Qwen/Qwen2.5-VL-3B-Instruct",
    "qwen2-vl": "Qwen/Qwen2-VL-2B-Instruct",
    "qwen3-vl": "Qwen/Qwen3-VL-2B-Instruct",
    "internvl3": "OpenGVLab/InternVL3-1B-hf",
    "llava": "llava-hf/llava-1.5-7b-hf",
    "llava-next": "llava-hf/llava-v1.6-mistral-7b-hf",
    "llava-onevision": "llava-hf/llava-onevision-qwen2-7b-ov-hf",
    "deepseek-ocr-tiny": None,
    "deepseek-ocr-small": None,
    "deepseek-ocr-base": None,
    "deepseek-ocr-large": None,
    "deepseek-ocr-gundam": None,
    "phi4-multimodal": None,
}

SUPPORTED_MODELS: set[str] = set(MODEL_TO_HF_ID.keys())

# Default short model name used across the app when none is provided
DEFAULT_MODEL: str = "qwen2.5-vl"


def map_model_id(model_name: str) -> Optional[str]:
    key = model_name.strip().lower()
    if key not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model: {model_name}")
    return MODEL_TO_HF_ID[key]


def load_analyst(model_name: str = DEFAULT_MODEL):
    """Factory that builds the correct analyst for a given short model name."""
    key = model_name.strip().lower()

    if key not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model: {model_name}")

    ANALYST_REGISTRY: Dict[str, Tuple[Callable, bool]] = {
        "qwen2.5-vl": (lambda proc, cfg: Qwen2_5_VLAnalyst(proc), False),
        "qwen2-vl": (lambda proc, cfg: Qwen2VLAnalyst(proc), False),
        "qwen3-vl": (lambda proc, cfg: Qwen3VLAnalyst(proc), False),
        "internvl3": (lambda proc, cfg: InternVLAnalyst(proc, cfg), True),
        "llava": (lambda proc, cfg: LLaVAAnalyst(proc), False),
        "llava-next": (lambda proc, cfg: LLaVANextAnalyst(proc), False),
        "llava-onevision": (lambda proc, cfg: LlavaOnevisionAnalyst(proc, cfg), True),
        "deepseek-ocr-tiny": (lambda proc, cfg: DeepSeekOCRAnalyst(mode="tiny"), False),
        "deepseek-ocr-small": (lambda proc, cfg: DeepSeekOCRAnalyst(mode="small"), False),
        "deepseek-ocr-base": (lambda proc, cfg: DeepSeekOCRAnalyst(mode="base"), False),
        "deepseek-ocr-large": (lambda proc, cfg: DeepSeekOCRAnalyst(mode="large"), False),
        "deepseek-ocr-gundam": (lambda proc, cfg: DeepSeekOCRAnalyst(mode="gundam"), False),
        "phi4-multimodal": (lambda proc, cfg: Phi4MultimodalAnalyst(), False),
    }

    if key not in ANALYST_REGISTRY:
        raise ValueError(f"No analyst registered for model: {model_name}")

    builder, needs_config = ANALYST_REGISTRY[key]

    model_id = MODEL_TO_HF_ID.get(key)
    if model_id is None:
        return builder(None, None)

    processor = AutoProcessor.from_pretrained(model_id)
    config = AutoConfig.from_pretrained(model_id) if needs_config else None

    return builder(processor, config)


__all__ = [
    "Qwen2VLAnalyst",
    "Qwen2_5_VLAnalyst",
    "Qwen3VLAnalyst",
    "InternVLAnalyst",
    "LLaVAAnalyst",
    "LLaVANextAnalyst",
    "LlavaOnevisionAnalyst",
    "DeepSeekOCRAnalyst",
    "Phi4MultimodalAnalyst",
    "load_analyst",
    "map_model_id",
    "SUPPORTED_MODELS",
    "MODEL_TO_HF_ID",
    "DEFAULT_MODEL",
]
