from functools import lru_cache
from typing import Tuple

from ..utils import smart_resize


class VLMAnalyst:
    def __init__(self, processor):
        self.processor = processor

    def get_num_image_patches(self, image_size: Tuple[int, int]) -> int:
        raise NotImplementedError

    def get_num_image_tokens(self, image_size: Tuple[int, int]) -> int:
        raise NotImplementedError

    def calculate(self, image_size: Tuple[int, int]) -> dict:
        raise NotImplementedError


class Qwen2VLAnalyst(VLMAnalyst):
    def __init__(self, processor):
        super().__init__(processor)

        self.image_token: str = "<|image_pad|>"
        self.image_start_token: str = "<|im_start|>"
        self.image_end_token: str = "<|im_end|>"

        self.patch_size = processor.image_processor.patch_size
        self.merge_size = processor.image_processor.merge_size
        self.min_pixels = processor.image_processor.min_pixels
        self.max_pixels = processor.image_processor.max_pixels

    @staticmethod
    @lru_cache(maxsize=1024)
    def _resize_and_grid(
        image_size: Tuple[int, int],
        patch_size: int,
        merge_size: int,
        min_pixels: int,
        max_pixels: int,
    ):
        height, width = image_size
        factor = patch_size * merge_size
        resized_h, resized_w = smart_resize(
            height, width, factor, min_pixels, max_pixels
        )
        grid_h = resized_h // patch_size
        grid_w = resized_w // patch_size
        return resized_h, resized_w, grid_h, grid_w

    def get_num_image_patches(self, image_size: Tuple[int, int]) -> int:
        _, _, grid_h, grid_w = self._resize_and_grid(
            image_size,
            self.patch_size,
            self.merge_size,
            self.min_pixels,
            self.max_pixels,
        )
        return grid_h * grid_w

    def get_num_image_tokens(self, image_size: Tuple[int, int]) -> int:
        # Qwen2-VL: merged tokens = patches / (merge_size^2)
        num_patches = self.get_num_image_patches(image_size)
        return num_patches // (self.merge_size**2)

    def calculate(self, image_size: Tuple[int, int]) -> dict:
        resized_h, resized_w, grid_h, grid_w = self._resize_and_grid(
            image_size,
            self.patch_size,
            self.merge_size,
            self.min_pixels,
            self.max_pixels,
        )
        num_patches = grid_h * grid_w
        num_tokens = num_patches // (self.merge_size**2)

        return {
            "number_of_image_tokens": num_tokens,
            "number_of_image_patches": num_patches,
            "image_size": image_size,
            "resized_size": (resized_w, resized_h),
            "image_token": self.image_token,
            "image_start_token": self.image_start_token,
            "image_end_token": self.image_end_token,
        }


class Qwen2_5_VLAnalyst(Qwen2VLAnalyst):
    pass


__all__ = [
    "Qwen2VLAnalyst",
    "Qwen2_5_VLAnalyst",
]
