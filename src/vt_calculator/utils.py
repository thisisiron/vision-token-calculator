import os
import glob
from io import BytesIO
from typing import Iterable
import numpy as np

import requests
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


def create_dummy_image(height: int, width: int):
    """
    Create a dummy image with specified dimensions.

    Args:
        height (int): Image height in pixels
        width (int): Image width in pixels

    Returns:
        PIL.Image.Image: PIL Image object
    """
    image_array = np.zeros((height, width, 3), dtype=np.uint8)
    image = Image.fromarray(image_array)

    return image


def create_dummy_video(
    file_path: str, width: int, height: int, fps: int, duration: int
) -> str:
    """
    Create a dummy video file with specified dimensions and duration.

    Args:
        file_path (str): Path where the video file will be created
        width (int): Video width in pixels
        height (int): Video height in pixels
        fps (int): Frames per second
        duration (int): Duration in seconds

    Returns:
        str: Path to the created video file
    """
    import cv2

    frames = int(duration * fps)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(file_path, fourcc, fps, (width, height))

    for _ in range(frames):
        frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        out.write(frame)

    out.release()

    return file_path


def check_transformers_version():
    """
    Check and print the version of the transformers library.

    Returns:
        str: The version of the transformers library, or None if not installed.
    """
    try:
        import transformers

        version = transformers.__version__
        print(f"Transformers version: {version}")

        major_ver = int(version.split(".")[0])
        if major_ver >= 5:
            print("Transformers version 5. Please install version 4.")

        return version
    except ImportError:
        print("Transformers library is not installed.")
        return None
    except Exception:
        return transformers.__version__


def is_url(path: str) -> bool:
    """
    Check if the given path is a URL.

    Args:
        path (str): Path or URL string to check

    Returns:
        bool: True if the path is a URL, False otherwise
    """
    if not isinstance(path, str):
        return False
    return path.lower().startswith(("http://", "https://"))


def is_video(path: str) -> bool:
    """
    Check if the given path is a video file.

    Args:
        path (str): Path string to check

    Returns:
        bool: True if the path is a video, False otherwise
    """
    if not isinstance(path, str):
        return False
    video_extensions = (".mp4", ".avi", ".mkv", ".mov", ".webm")
    return path.lower().endswith(video_extensions)


def load_image_from_url(url: str, timeout: int = 30) -> Image.Image:
    """
    Download image from URL and return as PIL Image.

    Args:
        url (str): URL of the image to download
        timeout (int): Request timeout in seconds (default: 30)

    Returns:
        PIL.Image.Image: PIL Image object

    Raises:
        requests.RequestException: If the request fails
        PIL.UnidentifiedImageError: If the content is not a valid image
    """
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()

    image = Image.open(BytesIO(response.content))
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")

    return image
