"""
Vision Token Calculator

A Python tool for calculating the number of tokens generated when processing images
with various Vision Language Models (VLMs).
"""

__version__ = "0.0.3"
__author__ = "Vision Token Calculator"

from .api import (
    count_image_tokens,
    count_video_tokens,
    process_directory,
    parse_compare_models,
    compare_image_tokens,
    compare_video_tokens,
)
from .utils import create_dummy_image, create_dummy_video
from .reporter import display_batch_results, display_comparison_results

__all__ = [
    "count_image_tokens",
    "count_video_tokens",
    "process_directory",
    "parse_compare_models",
    "compare_image_tokens",
    "compare_video_tokens",
    "create_dummy_image",
    "create_dummy_video",
    "display_batch_results",
    "display_comparison_results",
]
