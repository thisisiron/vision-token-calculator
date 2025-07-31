import os
import glob
from typing import Iterable
import numpy as np

from PIL import Image


def get_image_files(directory_path: str):
    """
    Get all image files from the specified directory.

    Args:
        directory_path (str): Path to directory containing images

    Returns:
        list: List of image file paths
    """
    image_extensions = [
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.webp",
    ]
    image_files = []

    for ext in image_extensions:
        for case_ext in [ext, ext.upper()]:
            pattern = os.path.join(directory_path, case_ext)
            image_files += glob.glob(pattern)

    return sorted(image_files)


def calculate_mean(values: Iterable[float]) -> float:
    arr = np.array(values, dtype=float)
    if arr.size == 0:
        return 0.0
    return float(arr.mean())


def calculate_stdev(values: Iterable[float]) -> float:
    arr = np.array(values, dtype=float)
    if arr.size < 2:
        return 0.0
    return float(arr.std(ddof=1))


def create_dummy_image(width: int, height: int):
    """
    Create a dummy image with specified dimensions.

    Args:
        width (int): Image width in pixels
        height (int): Image height in pixels

    Returns:
        PIL.Image.Image: PIL Image object
    """
    # Create a simple black image using np.zeros
    image_array = np.zeros((height, width, 3), dtype=np.uint8)
    image = Image.fromarray(image_array)

    return image