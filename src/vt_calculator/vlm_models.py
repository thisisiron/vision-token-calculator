from dataclasses import dataclass


@dataclass
class VLMBase:
    image_token: str = ""
    image_start_token: str = ""
    image_end_token: str = ""


@dataclass
class Qwen2_5_VL(VLMBase):
    image_token: str = "<|image_pad|>"
    image_start_token: str = "<|im_start|>"
    image_end_token: str = "<|im_end|>"


__all__ = [
    "VLMBase",
    "Qwen2_5_VL",
]
