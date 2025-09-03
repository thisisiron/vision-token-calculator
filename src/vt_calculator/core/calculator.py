import os

from ..setup_env import setup_quiet_environment

from transformers import AutoProcessor
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
from ..analysts.analyst import Qwen2_5_VLAnalyst


setup_quiet_environment()


def count_image_tokens(image_input, model_path: str = "Qwen/Qwen2.5-VL-7B-Instruct"):
    """
    Calculate the number of image tokens generated when processing an image.

    Args:
        image_input: Either a file path (str) or PIL Image object
        model_path (str): Model path to use for processing

    Returns:
        dict: Dictionary containing token counts and details
    """

    # Load only the processor (no need for the full model)
    processor = AutoProcessor.from_pretrained(model_path)

    if isinstance(image_input, str):
        image_input = Image.open(image_input)

    analyst = Qwen2_5_VLAnalyst(processor)

    # PIL.Image.size -> (width, height); analyst expects (height, width)
    width, height = image_input.size
    result = analyst.calculate((width, height))

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


def process_directory(directory_path: str, model_path: str):
    """
    Process all images in a directory and calculate batch statistics.

    Args:
        directory_path (str): Path to directory containing images
        model_path (str): Model path to use for processing

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

        result = count_image_tokens(image_file, model_path)
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
            stats = process_directory(args.image, args.model_path)
            display_batch_results(stats, args.model_path)
        else:
            # Use existing single image file
            print(f"Using existing image: {args.image}")

            # Calculate tokens
            result = count_image_tokens(args.image, args.model_path)

            # Display results using Reporter
            reporter = Reporter()
            reporter.display_single_image_results(
                result, args.model_path, f"Existing image: {args.image}"
            )

    elif args.size:
        # Create dummy image with specified dimensions
        width, height = args.size
        image_input = create_dummy_image(width, height)
        print(f"Using dummy image: {width} x {height}")

        # Calculate tokens
        result = count_image_tokens(image_input, args.model_path)

        # Display results using Reporter
        reporter = Reporter()
        reporter.display_single_image_results(result, args.model_path, "Dummy image")

    return 0


if __name__ == "__main__":
    main()
