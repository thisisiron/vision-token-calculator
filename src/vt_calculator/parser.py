import argparse


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
        "--model-path",
        "-m",
        type=str,
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="Model path to use (default: Qwen/Qwen2.5-VL-7B-Instruct)",
    )

    return parser.parse_args()
