"""CLI argument parser for vt-calc."""

import argparse

from ..analysts import SUPPORTED_MODELS, DEFAULT_MODEL


def parse_arguments():
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Vision Token Calculator")

    input_group = parser.add_mutually_exclusive_group(required=True)

    input_group.add_argument(
        "--size",
        "-s",
        type=int,
        nargs=2,
        metavar=("HEIGHT", "WIDTH"),
        help='Size of dummy image in format "HEIGHT WIDTH" (e.g., "1080 1920")',
    )

    input_group.add_argument(
        "--image", "-i", type=str, help="Path to image file or directory"
    )

    input_group.add_argument("--video", "-v", type=str, help="Path to video file")

    parser.add_argument(
        "--model-name",
        "-m",
        type=str,
        choices=sorted(SUPPORTED_MODELS),
        default=DEFAULT_MODEL,
        help=f"Short model name to use (default: {DEFAULT_MODEL})",
    )

    parser.add_argument(
        "--compare",
        "-c",
        type=str,
        default=None,
        metavar="MODELS",
        help=(
            "Compare multiple models (comma-separated). "
            "Use 'all' for all supported models. "
            "Example: --compare qwen2.5-vl,internvl3,llava"
        ),
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Frames per second to sample for video analysis (default: model specific)",
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to extract from video",
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Duration of the video in seconds (used for dummy video calculation)",
    )

    return parser.parse_args()
