import os

from ..setup_env import setup_quiet_environment

from PIL import Image
from ..utils import get_image_files, calculate_mean, calculate_stdev, create_dummy_image
from ..parser import parse_arguments
from ..reporter import (
    display_batch_results,
    print_processing_status,
    print_processing_result,
    print_directory_info,
)
from ..reporter import Reporter
from ..analysts import load_analyst, DEFAULT_MODEL


setup_quiet_environment()


def count_image_tokens(image_input, model_name: str = DEFAULT_MODEL):
    """
    Calculate the number of image tokens generated when processing an image.

    Args:
        image_input: Either a file path (str) or PIL Image object
        model_name (str): Short model name to use for processing

    Returns:
        dict: Dictionary containing token counts and details
    """

    # Build analyst via factory (handles aliases and config requirements)
    analyst = load_analyst(model_name)

    if isinstance(image_input, str):
        image_input = Image.open(image_input)

    # PIL.Image.size -> (width, height); analyst expects (height, width)
    width, height = image_input.size
    result = analyst.calculate((height, width))

    # Backward-compatible total token count for batch statistics
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


def process_directory(directory_path: str, model_name: str):
    """
    Process all images in a directory and calculate batch statistics.

    Args:
        directory_path (str): Path to directory containing images
        model_name (str): Short model name to use for processing

    Returns:
        dict: Dictionary containing batch statistics
    """
    # Get all image files
    image_files = get_image_files(directory_path)

    if not image_files:
        raise ValueError(f"No image files found in directory: {directory_path}")

    print_directory_info(directory_path, len(image_files))

    token_counts = []
    processed_files = []

    # Process each image
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

    # Calculate statistics
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


def main():
    """
    Main function to demonstrate image token counting.
    """
    args = parse_arguments()

    if args.image:
        if os.path.isdir(args.image):
            stats = process_directory(args.image, args.model_name)
            display_batch_results(stats, args.model_name)
        else:
            print(f"Using existing image: {args.image}")

            result = count_image_tokens(args.image, args.model_name)

            reporter = Reporter()
            reporter.print(result, args.model_name, f"{args.image}")

    elif args.size:
        height, width = args.size
        image_input = create_dummy_image(height, width)
        print(f"Using dummy image: {height} x {width}")

        # Calculate tokens
        result = count_image_tokens(image_input, args.model_name)

        # Display results using Reporter
        reporter = Reporter()
        reporter.print(result, args.model_name, "Dummy image")



if __name__ == "__main__":
    main()
