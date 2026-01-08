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
    if getattr(processor, "image_token", None) is not None:
        return processor.image_token
    if getattr(processor, "image_token_id", None) is not None:
        token = processor.tokenizer.convert_ids_to_tokens(processor.image_token_id)
        if isinstance(token, list):
            token = token[0]
        return token
    raise AssertionError("Processor has no image token or image token id")


def _assert_image_token_matches(processor, analyst) -> None:
    proc_token = _get_processor_image_token_str(processor)
    assert proc_token == analyst.image_token, (
        f"Mismatch between processor-image token ({proc_token}) and "
        f"Analyst-image token ({analyst.image_token})."
    )


def _assert_token_count_matches(counted_tokens: int, analyst_tokens: int) -> None:
    assert counted_tokens == analyst_tokens, (
        f"Mismatch between processor-counted tokens ({counted_tokens}) and "
        f"Analyst-computed tokens ({analyst_tokens})."
    )


def _count_video_tokens_via_processor(processor, video_path, fps=None) -> int:
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


@pytest.mark.network
@pytest.mark.slow
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
            "Qwen/Qwen3-VL-2B-Instruct",
            lambda proc, cfg: Qwen3VLAnalyst(proc),
            (800, 800),
            False,
            id="qwen3-vl",
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
            "llava-hf/llava-v1.6-mistral-7b-hf",
            lambda proc, cfg: LLaVANextAnalyst(proc),
            (800, 800),
            False,
            id="llava-next",
        ),
        pytest.param(
            "llava-hf/llava-onevision-qwen2-7b-ov-hf",
            lambda proc, cfg: LlavaOnevisionAnalyst(proc, cfg),
            (800, 800),
            True,
            id="llava-onevision",
        ),
    ],
)
def test_analyst_token_count_matches_transformers(
    model_path, analyst_factory, image_size, needs_config
):
    image = create_dummy_image(width=image_size[1], height=image_size[0])

    processor = AutoProcessor.from_pretrained(model_path)
    config = AutoConfig.from_pretrained(model_path) if needs_config else None

    counted_tokens = _count_tokens_via_processor(processor, image)

    analyst = analyst_factory(processor, config)
    result = analyst.calculate_image((image.height, image.width))
    analyst_tokens = int(result["image_token"][1])

    _assert_image_token_matches(processor, analyst)
    _assert_token_count_matches(counted_tokens, analyst_tokens)


@pytest.mark.network
@pytest.mark.slow
@pytest.mark.parametrize(
    "model_path,analyst_factory,fps",
    [
        pytest.param(
            "Qwen/Qwen2.5-VL-3B-Instruct",
            lambda proc, cfg: Qwen2_5_VLAnalyst(proc),
            1.0,
            id="qwen2.5-vl-video",
        ),
        pytest.param(
            "Qwen/Qwen3-VL-2B-Instruct",
            lambda proc, cfg: Qwen3VLAnalyst(proc),
            1.0,
            id="qwen3-vl-video",
        ),
    ],
)
def test_analyst_video_token_count_matches_transformers(
    model_path, analyst_factory, fps, dummy_video
):
    processor = AutoProcessor.from_pretrained(model_path)
    config = AutoConfig.from_pretrained(model_path)

    counted_tokens = _count_video_tokens_via_processor(processor, dummy_video, fps=fps)

    analyst = analyst_factory(processor, config)
    metadata = get_video_metadata(dummy_video)

    result = analyst.calculate_video(metadata, fps=fps)
    analyst_tokens = result["number_of_video_tokens"]

    _assert_token_count_matches(counted_tokens, analyst_tokens)
