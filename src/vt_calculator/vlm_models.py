from dataclasses import dataclass
import math


def smart_resize(
    height: int,
    width: int,
    factor: int = 28,
    min_pixels: int = 56 * 56,
    max_pixels: int = 14 * 14 * 4 * 1280,
):
    """Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.

    """
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


@dataclass
class VLMAnalyst:
    def __init__(self, processor):
        self.processor = processor

    def get_num_image_patches(self):
        pass

    def get_num_image_tokens(self):
        pass


@dataclass
class Qwen2_5_VLAnalyst(VLMAnalyst):
    def __init__(self, processor):
        super().__init__(processor)

        self.image_token: str = "<|image_pad|>"
        self.image_start_token: str = "<|im_start|>"
        self.image_end_token: str = "<|im_end|>"

        self.patch_size = processor.image_processor.patch_size
        self.merge_size = processor.image_processor.merge_size
        self.min_pixels = processor.image_processor.min_pixels
        self.max_pixels = processor.image_processor.max_pixels

    def get_num_patches(self, image_size):
        height, width = image_size
        factor = self.patch_size * self.merge_size
        resized_height, resized_width = smart_resize(
            height, width, factor, self.min_pixels, self.max_pixels
        )
        grid_h, grid_w = (
            resized_height // self.patch_size,
            resized_width // self.patch_size,
        )
        return grid_h * grid_w

    def get_num_image_tokens(self, image_size):
        num_patches = self.get_num_patches(image_size)
        return num_patches // self.merge_size**2


__all__ = [
    "Qwen2_5_VLAnalyst",
]
