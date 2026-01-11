"""HuggingFace-based VLM Analyst Tests.

This module tests VLM analysts against actual HuggingFace processors
to verify token count calculations are accurate.
"""

from dataclasses import dataclass
from typing import Type

import torch
import pytest
from transformers import AutoProcessor, AutoConfig

from vt_calculator.utils import create_dummy_image
from vt_calculator.video import get_video_metadata, extract_video_frames
from vt_calculator.analysts.analyst import (
    Qwen2_5_VLAnalyst,
    Qwen3VLAnalyst,
    InternVLAnalyst,
    LLaVAAnalyst,
    LLaVANextAnalyst,
    LlavaOnevisionAnalyst,
)


# =============================================================================
# Model Configuration
# =============================================================================


@dataclass
class HFModelConfig:
    """HuggingFace-based VLM model test configuration."""

    name: str
    hf_path: str
    analyst_class: Type
    needs_config: bool = False
    test_image_size: tuple = (800, 800)
    supports_video: bool = False
    video_fps: float = 1.0


HF_MODELS = [
    HFModelConfig(
        name="qwen2.5-vl",
        hf_path="Qwen/Qwen2.5-VL-3B-Instruct",
        analyst_class=Qwen2_5_VLAnalyst,
        supports_video=True,
    ),
    HFModelConfig(
        name="qwen3-vl",
        hf_path="Qwen/Qwen3-VL-2B-Instruct",
        analyst_class=Qwen3VLAnalyst,
        supports_video=True,
    ),
    HFModelConfig(
        name="internvl3",
        hf_path="OpenGVLab/InternVL3-1B-hf",
        analyst_class=InternVLAnalyst,
        needs_config=True,
    ),
    HFModelConfig(
        name="llava",
        hf_path="llava-hf/llava-1.5-7b-hf",
        analyst_class=LLaVAAnalyst,
    ),
    HFModelConfig(
        name="llava-next",
        hf_path="llava-hf/llava-v1.6-mistral-7b-hf",
        analyst_class=LLaVANextAnalyst,
    ),
    HFModelConfig(
        name="llava-onevision",
        hf_path="llava-hf/llava-onevision-qwen2-7b-ov-hf",
        analyst_class=LlavaOnevisionAnalyst,
        needs_config=True,
    ),
]


def get_image_test_configs():
    """Generate pytest params for image tests."""
    return [pytest.param(cfg, id=cfg.name) for cfg in HF_MODELS]


def get_video_test_configs():
    """Generate pytest params for video tests (supported models only)."""
    return [
        pytest.param(cfg, id=f"{cfg.name}-video")
        for cfg in HF_MODELS
        if cfg.supports_video
    ]


# =============================================================================
# Helper Functions
# =============================================================================


def _count_tokens_via_processor(processor, pil_image) -> int:
    """Count image tokens using actual HuggingFace processor."""
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": pil_image,
                }
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = processor(
        text=[text],
        images=[pil_image],
        videos=None,
        padding=True,
        return_tensors="pt",
    )

    if getattr(processor, "image_token", None) is not None:
        image_token_index = processor.tokenizer.convert_tokens_to_ids(
            processor.image_token
        )
    elif getattr(processor, "image_token_id", None) is not None:
        image_token_index = processor.image_token_id
    else:
        raise ValueError("Image token not found in processor")

    input_ids = inputs["input_ids"]
    num_image_tokens_tensor = (input_ids[0] == image_token_index).sum()
    return int(
        num_image_tokens_tensor.item()
        if isinstance(num_image_tokens_tensor, torch.Tensor)
        else num_image_tokens_tensor
    )


def _get_processor_image_token_str(processor) -> str:
    """Get image token string from processor."""
    if getattr(processor, "image_token", None) is not None:
        return processor.image_token
    if getattr(processor, "image_token_id", None) is not None:
        token = processor.tokenizer.convert_ids_to_tokens(processor.image_token_id)
        if isinstance(token, list):
            token = token[0]
        return token
    raise AssertionError("Processor has no image token or image token id")


def _assert_image_token_matches(processor, analyst) -> None:
    """Assert processor and analyst use the same image token."""
    proc_token = _get_processor_image_token_str(processor)
    assert proc_token == analyst.image_token, (
        f"Mismatch between processor-image token ({proc_token}) and "
        f"Analyst-image token ({analyst.image_token})."
    )


def _assert_token_count_matches(counted_tokens: int, analyst_tokens: int) -> None:
    """Assert token counts match between processor and analyst."""
    assert counted_tokens == analyst_tokens, (
        f"Mismatch between processor-counted tokens ({counted_tokens}) and "
        f"Analyst-computed tokens ({analyst_tokens})."
    )


def _count_video_tokens_via_processor(processor, video_path, fps=None) -> int:
    """Count video tokens using actual HuggingFace processor."""
    if "Qwen2" not in processor.__class__.__name__ and "Qwen2" not in str(processor):
        raise NotImplementedError(
            "Video token counting is currently only supported for Qwen2-VL models. "
            "Other models require model-specific video frame loading logic."
        )

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "fps": fps,
                },
                {"type": "text", "text": "Describe this video."},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    video_frames = extract_video_frames(video_path, fps=fps)

    inputs = processor(
        text=[text],
        images=None,
        videos=[video_frames.frames],
        padding=True,
        return_tensors="pt",
    )

    input_ids = inputs["input_ids"][0].tolist()

    video_pad_token_id = processor.tokenizer.convert_tokens_to_ids("<|video_pad|>")
    if video_pad_token_id != processor.tokenizer.unk_token_id:
        return input_ids.count(video_pad_token_id)

    raise ValueError("Could not determine video tokens for processor")


def _create_analyst(config: HFModelConfig, processor, model_config):
    """Create analyst instance from config."""
    if config.needs_config:
        return config.analyst_class(processor, model_config)
    return config.analyst_class(processor)


# =============================================================================
# Tests
# =============================================================================


@pytest.mark.network
@pytest.mark.slow
@pytest.mark.parametrize("config", get_image_test_configs())
def test_analyst_token_count_matches_transformers(config: HFModelConfig):
    """Verify analyst token count matches actual HuggingFace processor count."""
    image = create_dummy_image(
        width=config.test_image_size[1],
        height=config.test_image_size[0],
    )

    processor = AutoProcessor.from_pretrained(config.hf_path)
    model_config = AutoConfig.from_pretrained(config.hf_path) if config.needs_config else None

    counted_tokens = _count_tokens_via_processor(processor, image)

    analyst = _create_analyst(config, processor, model_config)
    result = analyst.calculate_image((image.height, image.width))
    analyst_tokens = int(result["image_token"][1])

    _assert_image_token_matches(processor, analyst)
    _assert_token_count_matches(counted_tokens, analyst_tokens)


@pytest.mark.network
@pytest.mark.slow
@pytest.mark.parametrize("config", get_video_test_configs())
def test_analyst_video_token_count_matches_transformers(config: HFModelConfig, dummy_video):
    """Verify analyst video token count matches actual HuggingFace processor count."""
    processor = AutoProcessor.from_pretrained(config.hf_path)
    model_config = AutoConfig.from_pretrained(config.hf_path)

    counted_tokens = _count_video_tokens_via_processor(
        processor, dummy_video, fps=config.video_fps
    )

    analyst = _create_analyst(config, processor, model_config)
    metadata = get_video_metadata(dummy_video)

    result = analyst.calculate_video(metadata, fps=config.video_fps)
    analyst_tokens = result["number_of_video_tokens"]

    _assert_token_count_matches(counted_tokens, analyst_tokens)
