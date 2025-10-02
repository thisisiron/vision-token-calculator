import torch
import pytest
from transformers import AutoProcessor, AutoConfig

from vt_calculator.utils import create_dummy_image
from vt_calculator.analysts.analyst import (
    Qwen2_5_VLAnalyst,
    InternVLAnalyst,
    LLaVAAnalyst,
    LlavaOnevisionAnalyst
)


def _count_tokens_via_processor(processor, pil_image) -> int:
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
    """Return the processor's image token as a string, with id fallback."""
    if getattr(processor, "image_token", None) is not None:
        return processor.image_token
    if getattr(processor, "image_token_id", None) is not None:
        token = processor.tokenizer.convert_ids_to_tokens(processor.image_token_id)
        if isinstance(token, list):
            token = token[0]
        return token
    raise AssertionError("Processor has no image token or image token id")


def _assert_image_token_matches(processor, analyst) -> None:
    """Assert that the processor and analyst image tokens match."""
    proc_token = _get_processor_image_token_str(processor)
    assert proc_token == analyst.image_token, (
        f"Mismatch between processor-image token ({proc_token}) and "
        f"Analyst-image token ({analyst.image_token})."
    )


def _assert_token_count_matches(counted_tokens: int, analyst_tokens: int) -> None:
    """Assert that the counted tokens equal the analyst-computed tokens."""
    assert counted_tokens == analyst_tokens, (
        f"Mismatch between processor-counted tokens ({counted_tokens}) and "
        f"Analyst-computed tokens ({analyst_tokens})."
    )


@pytest.mark.parametrize(
    "model_path,analyst_factory,image_size,needs_config",
    [
        pytest.param(
            "Qwen/Qwen2.5-VL-3B-Instruct",
            lambda proc, cfg: Qwen2_5_VLAnalyst(proc),
            (800, 800),
            False,
            id="qwen2.5-vl",
        ),
        pytest.param(
            "OpenGVLab/InternVL3-1B-hf",
            lambda proc, cfg: InternVLAnalyst(proc, cfg),
            (800, 800),
            True,
            id="internvl3",
        ),
        pytest.param(
            "llava-hf/llava-1.5-7b-hf",
            lambda proc, cfg: LLaVAAnalyst(proc),
            (800, 800),
            False,
            id="llava",
        ),
        pytest.param(
            "llava-hf/llava-onevision-qwen2-7b-ov-hf",
            lambda proc, cfg: LlavaOnevisionAnalyst(proc),
            (800, 800),
            False,
            id="llava-onevision",
        ),
    ],
)
def test_analyst_token_count_matches_transformers(
    model_path, analyst_factory, image_size, needs_config
):
    """Parametrized test verifying analyst matches processor token behavior."""
    # Create a small deterministic image (image_size is (H, W))
    image = create_dummy_image(width=image_size[1], height=image_size[0])

    # Load the real processor (may download configs on first run)
    processor = AutoProcessor.from_pretrained(model_path)
    config = AutoConfig.from_pretrained(model_path) if needs_config else None

    # Count tokens via processor outputs
    counted_tokens = _count_tokens_via_processor(processor, image)

    # Use the same processor for Analyst
    analyst = analyst_factory(processor, config)
    # PIL.Image.size -> (W, H); analyst expects (H, W)
    result = analyst.calculate((image.height, image.width))
    # Compare only the number of image tokens (not including wrapper tokens)
    analyst_tokens = int(result["image_token"][1])

    _assert_image_token_matches(processor, analyst)
    _assert_token_count_matches(counted_tokens, analyst_tokens)
