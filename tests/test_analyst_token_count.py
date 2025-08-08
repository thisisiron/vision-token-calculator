import torch
from transformers import AutoProcessor

from vt_calculator.utils import create_dummy_image
from vt_calculator.analysts.analyst import Qwen2_5_VLAnalyst


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

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = processor(
        text=[text],
        images=[pil_image],
        videos=None,
        padding=True,
        return_tensors="pt",
    )

    if getattr(processor, "image_token", None) is not None:
        image_token_index = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
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


def test_analyst_token_count_matches_transformers():
    """
    Verify that Analyst's computed token count matches the count measured from
    the real transformers AutoProcessor outputs on a simple dummy image.
    """

    # Create a small deterministic image
    image = create_dummy_image(width=256, height=256)

    # Load the real processor (may download configs on first run)
    model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
    processor = AutoProcessor.from_pretrained(model_path)

    # Count tokens via processor outputs
    counted_tokens = _count_tokens_via_processor(processor, image)

    # Use the same processor for Analyst
    analyst = Qwen2_5_VLAnalyst(processor)
    analyst_tokens = analyst.get_num_image_tokens(image.size)

    assert counted_tokens == analyst_tokens, (
        f"Mismatch between processor-counted tokens ({counted_tokens}) and "
        f"Analyst-computed tokens ({analyst_tokens})."
    )


