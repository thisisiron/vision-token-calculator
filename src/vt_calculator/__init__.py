"""
Vision Token Calculator

A Python tool for calculating the number of tokens generated when processing images
with various Vision Language Models (VLMs).
"""

__version__ = "0.0.2"
__author__ = "Vision Token Calculator"

from .core.calculator import count_image_tokens, process_directory
from .utils import create_dummy_image
from .reporter import display_batch_results

__all__ = [
    "count_image_tokens",
    "create_dummy_image",
    "process_directory",
    "display_batch_results",
]
