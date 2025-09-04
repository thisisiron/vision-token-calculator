from typing import Tuple

from .tools import resize_and_grid, get_optimal_tiled_canvas


class VLMAnalyst:
    def __init__(self, processor):
        self.processor = processor

    def calculate(self, image_size: Tuple[int, int]) -> dict:
        """
        Calculate the number of image tokens for a given image size.

        Args:
            image_size (Tuple[int, int]): The size of the image in (width, height) format.

        Returns:
            dict: A dictionary containing the number of image tokens and other relevant information.
        """
        raise NotImplementedError


class Qwen2VLAnalyst(VLMAnalyst):
    def __init__(self, processor):
        super().__init__(processor)

        self.image_token: str = "<|image_pad|>"
        self.image_start_token: str = "<|vision_start|>"
        self.image_end_token: str = "<|vision_end|>"

        self.patch_size = processor.image_processor.patch_size
        self.merge_size = processor.image_processor.merge_size
        self.min_pixels = processor.image_processor.min_pixels
        self.max_pixels = processor.image_processor.max_pixels

    def calculate(self, image_size: Tuple[int, int]) -> dict:
        resized_h, resized_w, grid_h, grid_w = resize_and_grid(
            image_size,
            self.patch_size,
            self.merge_size,
            self.min_pixels,
            self.max_pixels,
        )
        num_patches = grid_h * grid_w

        # Qwen2-VL: merged tokens = patches / (merge_size^2)
        num_tokens = num_patches // (self.merge_size**2)

        return {
            "number_of_image_patches": num_patches,
            "image_size": image_size,
            "resized_size": (resized_w, resized_h),
            "image_token": (self.image_token, num_tokens),
            "image_start_token": (self.image_start_token, 1),
            "image_end_token": (self.image_end_token, 1),
        }


class Qwen2_5_VLAnalyst(Qwen2VLAnalyst):
    pass


class InternVLAnalyst(VLMAnalyst):
    def __init__(self, processor, config):
        super().__init__(processor)

        self.image_token: str = "<IMG_CONTEXT>"
        self.image_start_token: str = "<img>"
        self.image_end_token: str = "</img>"

        self.min_patches = processor.image_processor.min_patches
        self.max_patches = processor.image_processor.max_patches
        assert (
            processor.image_processor.size["height"]
            == processor.image_processor.size["width"]
        )
        self.patch_size = processor.image_processor.size["height"]

        assert config.vision_config.patch_size[0] == config.vision_config.patch_size[1]
        self.vit_patch_size = config.vision_config.patch_size[0]
        self.pixel_unshuffle_size = 2

        #
        self.image_seq_length = (
            self.patch_size // self.vit_patch_size // self.pixel_unshuffle_size
        )

    def calculate(self, image_size: Tuple[int, int]) -> dict:
        num_patches = 1
        grid_w, grid_h = get_optimal_tiled_canvas(
            image_size,
            (self.patch_size, self.patch_size),
            self.min_patches,
            self.max_patches,
        )
        if grid_w * grid_h > 1:
            num_patches += grid_h * grid_w

        num_tokens = num_patches * (self.image_seq_length**2)

        return {
            "number_of_image_patches": num_patches,
            "image_size": image_size,
            "image_token": (self.image_token, num_tokens),
            "image_start_token": (self.image_start_token, 1),
            "image_end_token": (self.image_end_token, 1),
        }
