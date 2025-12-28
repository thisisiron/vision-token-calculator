from functools import lru_cache
from typing import Tuple
import math


def select_best_resolution(original_size: tuple, possible_resolutions: list) -> tuple:
    """
    Selects the best resolution from a list of possible resolutions based on the original size.

    This is done by calculating the effective and wasted resolution for each possible resolution.

    The best fit resolution is the one that maximizes the effective resolution and minimizes the wasted resolution.

    Args:
        original_size (tuple):
            The original size of the image in the format (height, width).
        possible_resolutions (list):
            A list of possible resolutions in the format [(height1, width1), (height2, width2), ...].

    Returns:
        tuple: The best fit resolution in the format (height, width).
    """
    original_height, original_width = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float("inf")

    for height, width in possible_resolutions:
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = (
            int(original_width * scale),
            int(original_height * scale),
        )
        effective_resolution = min(
            downscaled_width * downscaled_height, original_width * original_height
        )
        wasted_resolution = (width * height) - effective_resolution

        if effective_resolution > max_effective_resolution or (
            effective_resolution == max_effective_resolution
            and wasted_resolution < min_wasted_resolution
        ):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (height, width)

    return best_fit


def get_patch_output_size(image_size, target_resolution):
    """
    Given an image and a target resolution, calculate the output size of the image after cropping to the target
    """
    original_height, original_width = image_size
    target_height, target_width = target_resolution

    scale_w = target_width / original_width
    scale_h = target_height / original_height

    if scale_w < scale_h:
        new_width = target_width
        new_height = min(math.ceil(original_height * scale_w), target_height)
    else:
        new_height = target_height
        new_width = min(math.ceil(original_width * scale_h), target_width)

    return new_height, new_width


def get_unpadded_features(
    height,
    width,
    patches_height,
    patches_width,
    scale_height,
    scale_width,
    max_num_patches=None,
):
    """
    Get number of features for a given image with height/width. LLaVA-NeXT is different from LLaVA
    because it divided each image into patches depending on its resolution. Therefore we need to calculate how many
    patches an image is divided into and get the number of features from that.
    """
    current_height = patches_height * scale_height
    current_width = patches_width * scale_width

    original_aspect_ratio = width / height
    current_aspect_ratio = current_width / current_height
    if original_aspect_ratio > current_aspect_ratio:
        new_height = int(round(height * (current_width / width), 7))
        padding = (current_height - new_height) // 2
        current_height -= padding * 2
    else:
        new_width = int(round(width * (current_height / height), 7))
        padding = (current_width - new_width) // 2
        current_width -= padding * 2

    unpadded_features = current_height * current_width
    newline_features = current_height

    if max_num_patches is not None:
        ratio = math.sqrt(
            current_height * current_width / (max_num_patches * patches_height**2)
        )
        if ratio > 1.1:
            unpadded_features = int(current_height // ratio) * int(
                current_width // ratio
            )
            newline_features = int(current_height // ratio)

    return (unpadded_features, newline_features)


@lru_cache(maxsize=10)
def get_all_supported_aspect_ratios(
    min_image_tiles: int, max_image_tiles: int
) -> list[tuple[int, int]]:
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
    possible_tile_arrangements = get_all_supported_aspect_ratios(
        min_image_tiles, max_image_tiles
    )

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


def get_padding_size(original_resolution: tuple, target_resolution: tuple):
    original_height, original_width = original_resolution
    target_height, target_width = target_resolution
    paste_x, r_x = divmod(target_width - original_width, 2)
    paste_y, r_y = divmod(target_height - original_height, 2)
    return (paste_y, paste_y + r_y), (paste_x, paste_x + r_x)


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


@lru_cache(maxsize=1024)
def resize_and_grid(
    image_size: Tuple[int, int],
    patch_size: int,
    merge_size: int,
    min_pixels: int,
    max_pixels: int,
):
    height, width = image_size
    factor = patch_size * merge_size
    resized_h, resized_w = smart_resize(height, width, factor, min_pixels, max_pixels)
    grid_h = resized_h // patch_size
    grid_w = resized_w // patch_size
    return resized_h, resized_w, grid_h, grid_w
