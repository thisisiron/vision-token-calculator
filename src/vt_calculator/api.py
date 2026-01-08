"""Library API for vision token calculation."""

import os
from typing import List, Dict, Any, Optional

from PIL import Image

from .analysts import load_analyst, DEFAULT_MODEL, SUPPORTED_MODELS
from .video import get_video_metadata
from .utils import (
    get_image_files,
    calculate_mean,
    calculate_stdev,
    is_url,
    load_image_from_url,
)
from .reporter import (
    print_processing_status,
    print_processing_result,
    print_directory_info,
)


# =============================================================================
# Private Helpers
# =============================================================================


def _extract_total_tokens(result: dict) -> int:
    """Extract total token count from analyst result.

    Handles different result formats from various VLM analysts.

    Args:
        result: Token calculation result dictionary

    Returns:
        Total number of tokens
    """
    if "number_of_image_tokens" in result:
        return int(result["number_of_image_tokens"])

    total = 0
    for key in ["image_token", "image_start_token", "image_end_token"]:
        value = result.get(key)
        if isinstance(value, (list, tuple)) and len(value) == 2:
            total += int(value[1])

    if total > 0:
        return total

    image_token = result.get("image_token")
    if isinstance(image_token, (list, tuple)) and len(image_token) == 2:
        return int(image_token[1])

    return 0


# =============================================================================
# Single Model API
# =============================================================================


def count_image_tokens(image_input, model_name: str = DEFAULT_MODEL) -> dict:
    """Calculate the number of image tokens for a given image.

    Args:
        image_input: Either a file path (str), URL, or PIL Image object
        model_name: Short model name to use for processing

    Returns:
        dict: Dictionary containing token counts and details
    """
    analyst = load_analyst(model_name)

    if isinstance(image_input, str):
        if is_url(image_input):
            image_input = load_image_from_url(image_input)
        else:
            image_input = Image.open(image_input)

    width, height = image_input.size
    result = analyst.calculate_image((height, width))

    if (
        isinstance(result.get("image_token"), (list, tuple))
        and isinstance(result.get("image_start_token"), (list, tuple))
        and isinstance(result.get("image_end_token"), (list, tuple))
    ):
        total_tokens = (
            int(result["image_token"][1])
            + int(result["image_start_token"][1])
            + int(result["image_end_token"][1])
        )
        result["number_of_image_tokens"] = total_tokens

    return result


def count_video_tokens(
    video_path: str,
    model_name: str = DEFAULT_MODEL,
    fps: Optional[float] = None,
    max_frames: Optional[int] = None,
) -> dict:
    """Calculate the number of video tokens for a given video.

    Args:
        video_path: Path to video file
        model_name: Short model name to use for processing
        fps: Frames per second for sampling
        max_frames: Maximum number of frames to extract

    Returns:
        dict: Dictionary containing token counts and details
    """
    analyst = load_analyst(model_name)
    metadata = get_video_metadata(video_path)
    return analyst.calculate_video(metadata, fps=fps, max_frames=max_frames)


def process_directory(directory_path: str, model_name: str) -> dict:
    """Process all images in a directory and calculate batch statistics.

    Args:
        directory_path: Path to directory containing images
        model_name: Short model name to use for processing

    Returns:
        dict: Dictionary containing batch statistics
    """
    image_files = get_image_files(directory_path)

    if not image_files:
        raise ValueError(f"No image files found in directory: {directory_path}")

    print_directory_info(directory_path, len(image_files))

    token_counts = []
    processed_files = []

    for i, image_file in enumerate(image_files, 1):
        filename = os.path.basename(image_file)
        print_processing_status(filename, i, len(image_files))

        result = count_image_tokens(image_file, model_name)
        token_count = int(result["number_of_image_tokens"])
        token_counts.append(token_count)
        processed_files.append(
            {"filename": filename, "size": result["image_size"], "tokens": token_count}
        )
        print_processing_result(True, token_count)

    stats = {
        "total_processed": len(processed_files),
        "total_failed": 0,
        "average_tokens": calculate_mean(token_counts),
        "min_tokens": min(token_counts),
        "max_tokens": max(token_counts),
        "std_deviation": calculate_stdev(token_counts),
        "processed_files": processed_files,
        "failed_files": [],
    }

    return stats


# =============================================================================
# Model Comparison API
# =============================================================================


def parse_compare_models(compare_str: str) -> List[str]:
    """Parse --compare argument into list of model names.

    Args:
        compare_str: Comma-separated model names or 'all'

    Returns:
        List of valid model short names

    Raises:
        ValueError: If any model name is invalid
    """
    if compare_str.lower() == "all":
        return sorted(SUPPORTED_MODELS)

    models = [m.strip().lower() for m in compare_str.split(",")]
    invalid = [m for m in models if m not in SUPPORTED_MODELS]
    if invalid:
        raise ValueError(
            f"Unsupported models: {invalid}. Supported: {sorted(SUPPORTED_MODELS)}"
        )

    return models


def compare_image_tokens(
    image_input,
    model_names: List[str],
) -> Dict[str, Any]:
    """Compare image tokens across multiple models.

    Args:
        image_input: Either a file path (str), URL, or PIL Image object
        model_names: List of model short names to compare

    Returns:
        dict: Comparison results with rankings and summary
    """
    if isinstance(image_input, str):
        if is_url(image_input):
            image = load_image_from_url(image_input)
        else:
            image = Image.open(image_input)
    else:
        image = image_input

    width, height = image.size
    image_size = (height, width)

    results = []
    for model_name in model_names:
        try:
            analyst = load_analyst(model_name)
            result = analyst.calculate_image(image_size)
            total_tokens = _extract_total_tokens(result)

            results.append(
                {
                    "model": model_name,
                    "tokens": total_tokens,
                    "details": result,
                    "error": None,
                }
            )
        except Exception as e:
            results.append(
                {
                    "model": model_name,
                    "tokens": None,
                    "details": None,
                    "error": str(e),
                }
            )

    valid_results = [r for r in results if r["tokens"] is not None]
    valid_results.sort(key=lambda x: x["tokens"])

    summary = {}
    if valid_results:
        summary = {
            "min_tokens": valid_results[0]["tokens"],
            "max_tokens": valid_results[-1]["tokens"],
            "best_model": valid_results[0]["model"],
            "worst_model": valid_results[-1]["model"],
        }

    return {
        "type": "image_comparison",
        "image_size": image_size,
        "results": results,
        "summary": summary,
    }


def compare_video_tokens(
    video_input,
    model_names: List[str],
    fps: Optional[float] = None,
    max_frames: Optional[int] = None,
) -> Dict[str, Any]:
    """Compare video tokens across multiple models.

    Args:
        video_input: Path to video file or metadata dict
        model_names: List of model short names to compare
        fps: Frames per second for sampling
        max_frames: Maximum number of frames to extract

    Returns:
        dict: Comparison results with rankings and summary
    """
    if isinstance(video_input, dict):
        metadata = video_input
    else:
        metadata = get_video_metadata(video_input)

    results = []
    for model_name in model_names:
        try:
            analyst = load_analyst(model_name)
            result = analyst.calculate_video(metadata, fps=fps, max_frames=max_frames)
            total_tokens = result.get("number_of_video_tokens", 0)

            results.append(
                {
                    "model": model_name,
                    "tokens": total_tokens,
                    "details": result,
                    "error": None,
                }
            )
        except NotImplementedError:
            results.append(
                {
                    "model": model_name,
                    "tokens": None,
                    "details": None,
                    "error": "Video not supported",
                }
            )
        except Exception as e:
            results.append(
                {
                    "model": model_name,
                    "tokens": None,
                    "details": None,
                    "error": str(e),
                }
            )

    valid_results = [r for r in results if r["tokens"] is not None]
    valid_results.sort(key=lambda x: x["tokens"])

    summary = {}
    if valid_results:
        summary = {
            "min_tokens": valid_results[0]["tokens"],
            "max_tokens": valid_results[-1]["tokens"],
            "best_model": valid_results[0]["model"],
            "worst_model": valid_results[-1]["model"],
        }

    return {
        "type": "video_comparison",
        "video_metadata": metadata,
        "results": results,
        "summary": summary,
    }
