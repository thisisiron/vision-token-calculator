from functools import lru_cache
from typing import Tuple

from ..utils import smart_resize


@lru_cache(maxsize=10)
def get_all_supported_aspect_ratios(min_image_tiles: int, max_image_tiles: int) -> list[tuple[int, int]]:
    """
    Computes all allowed aspect ratios for a given minimum and maximum number of input tiles.

    This function calculates all possible arrangements of tiles that can be formed
    within the constraint of the minimum and maximum number of tiles. Each arrangement is
    represented by its aspect ratio (width/height) and the corresponding tile configuration.

    Args:
        min_image_tiles (`int`):
            The minimum number of tiles allowed.
        max_image_tiles (`int`):
            The maximum number of tiles allowed.

    Returns:
        `list[tuple[int, int]]`: A list of tuples, each tuple representing a valid (width, height)
        configuration in terms of number of tiles.

    Example:
        >>> get_all_supported_aspect_ratios(1, 4)
        [(1, 1), (1, 2), (2, 1), (1, 3), (3, 1), (1, 4), (2, 2), (4, 1)]

    """
    aspect_ratios = []
    for width in range(1, max_image_tiles + 1):
        for height in range(1, max_image_tiles + 1):
            if width * height <= max_image_tiles and width * height >= min_image_tiles:
                aspect_ratios.append((width, height))

    aspect_ratios = sorted(aspect_ratios, key=lambda x: x[0] * x[1])

    return aspect_ratios



@lru_cache(maxsize=100)
def get_optimal_tiled_canvas(
    original_image_size: tuple[int, int],
    target_tile_size: tuple[int, int],
    min_image_tiles: int,
    max_image_tiles: int,
) -> tuple[int, int]:
    """
    Given a minimum and maximum number of tiles, find the canvas with the closest aspect ratio to the
    original image aspect ratio.
    In case of tie-breaking condition when two canvases have the same aspect ratio difference, we favor the canvas with
    more tiles, until the area covered by the tiles is more than twice the target area, in order to avoid unnecessarily
    excessive tiling.
    """
    possible_tile_arrangements = get_all_supported_aspect_ratios(min_image_tiles, max_image_tiles)

    original_height, original_width = original_image_size
    target_tile_height, target_tile_width = target_tile_size
    aspect_ratio = original_width / original_height
    area = original_width * original_height

    # find the grid with the best aspect ratio
    best_ratio_diff = float("inf")
    best_grid = (1, 1)
    for grid in possible_tile_arrangements:
        grid_aspect_ratio = grid[0] / grid[1]
        ratio_diff = abs(aspect_ratio - grid_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_grid = grid
        elif ratio_diff == best_ratio_diff:
            # if the aspect ratio difference is the same, we favor the grid with more patches
            # until the area covered by the patches is more than twice the original image area
            if area > 0.5 * target_tile_height * target_tile_width * grid[0] * grid[1]:
                best_grid = grid

    return best_grid



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
        self.image_start_token: str = "<|vision_start|>"
        self.image_end_token: str = "<|vision_end|>"

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

    def calculate(self, image_size: Tuple[int, int]) -> dict:
        resized_h, resized_w, grid_h, grid_w = self._resize_and_grid(
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



class InternVLAnalyst(VLMAnalyst):
    def __init__(self, processor):
        super().__init__(processor)

        self.image_token: str = "<|image_pad|>"
        self.image_start_token: str = "<img>"
        self.image_end_token: str = "</img>"

        self.min_patch_size = processor.image_processor.min_patch_size
        self.max_patch_size = processor.image_processor.max_patch_size
        self.patch_size = processor.image_processor.patch_size

    def calculate(self, image_size: Tuple[int, int]) -> dict:
        num_patches = 1
        grid_w, grid_h = get_optimal_tiled_canvas(
            image_size,
            (self.patch_size, self.patch_size),
            self.min_patch_size,
            self.max_patch_size,
        )
        if grid_w * grid_h > 1:
            num_patches += grid_h * grid_w
        
        num_tokens = num_patches * (self.patch_size**2)
        
        return {
            "number_of_image_tokens": num_tokens,
            "number_of_image_patches": num_patches,
            "image_size": image_size,
            "image_token": self.image_token,
            "image_start_token": self.image_start_token,
            "image_end_token": self.image_end_token,
        }


__all__ = [
    "Qwen2VLAnalyst",
    "Qwen2_5_VLAnalyst",
]
