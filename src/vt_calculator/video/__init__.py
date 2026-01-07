"""
Video processing module for vision token calculator.

This module provides video backend selection, metadata extraction,
and frame sampling utilities for all VLM models.
"""

from .backends import VideoBackend, get_default_backend
from .video import get_video_metadata, extract_video_frames, VideoMetadata, VideoFrames

__all__ = [
    "VideoBackend",
    "get_default_backend",
    "get_video_metadata",
    "extract_video_frames",
    "VideoMetadata",
    "VideoFrames",
]
