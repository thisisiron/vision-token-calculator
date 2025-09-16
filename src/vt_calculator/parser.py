import argparse

from .analysts import SUPPORTED_MODELS, DEFAULT_MODEL


def parse_arguments():
    parser = argparse.ArgumentParser(description="Vision Token Calculator")

    input_group = parser.add_mutually_exclusive_group(required=True)

    input_group.add_argument(
        "--size",
        "-s",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        help='Size of dummy image in format "WIDTH HEIGHT" (e.g., "1920 1080")',
    )

    input_group.add_argument(
        "--image", "-i", type=str, help="Path to image file or directory"
    )

    parser.add_argument(
        "--model-name",
        "-m",
        type=str,
        choices=sorted(SUPPORTED_MODELS),
        default=DEFAULT_MODEL,
        help=f"Short model name to use (default: {DEFAULT_MODEL})",
    )

    return parser.parse_args()
