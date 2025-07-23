import os

from .setup_env import setup_quiet_environment

setup_quiet_environment()

from transformers import AutoProcessor
from PIL import Image
import argparse
import numpy as np
from .utils import get_image_files, calculate_mean, calculate_stdev
from .printer import (
    display_batch_results,
    display_single_image_results,
    print_processing_status,
    print_processing_result,
    print_directory_info,
)


def create_dummy_image(width: int, height: int):
    """
    Create a dummy image with specified dimensions.

    Args:
        width (int): Image width in pixels
        height (int): Image height in pixels

    Returns:
        PIL.Image.Image: PIL Image object
    """
    # Create a simple black image using np.zeros
    image_array = np.zeros((height, width, 3), dtype=np.uint8)
    image = Image.fromarray(image_array)

    return image


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

    # Create messages with file path
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_input,
                }
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    image_inputs = [image_input]
    video_inputs = None

    # Process inputs
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    if "qwen" in model_path.lower():
        grid_t, grid_h, grid_w = inputs["image_grid_thw"][0].tolist()
        reszied_height = grid_h * processor.image_processor.patch_size
        reszied_width = grid_w * processor.image_processor.patch_size

    # Calculate token counts
    input_ids = inputs["input_ids"]
    if processor.image_token is not None:
        image_token = processor.image_token
        image_token_index = processor.tokenizer.convert_tokens_to_ids(image_token)
    elif processor.image_token_id is not None:
        image_token_index = processor.image_token_id
    else:
        raise ValueError("Image token not found in processor")

    num_image_tokens = (input_ids[0] == image_token_index).sum()

    # Get detailed token information
    processor_info = {
        "number_of_image_tokens": num_image_tokens,
        "image_size": image_input.size,
        "image_token": processor.tokenizer.decode(image_token_index),
        "resized_size": (reszied_width, reszied_height),
    }

    return processor_info


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


def main():
    """
    Main function to demonstrate image token counting.
    """
    args = parse_arguments()

    if args.image:
        # If --image points to a directory, process it as a batch
        if os.path.isdir(args.image):
            stats = process_directory(args.image, args.model_path)
            display_batch_results(stats, args.model_path)
        else:
            # Use existing single image file
            print(f"Using existing image: {args.image}")

            # Calculate tokens
            result = count_image_tokens(args.image, args.model_path)

            # Display results
            display_single_image_results(
                result, args.model_path, f"Existing image: {args.image}"
            )

    elif args.size:
        # Create dummy image with specified dimensions
        width, height = args.size
        image_input = create_dummy_image(width, height)
        print(f"Using dummy image: {width} x {height}")

        # Calculate tokens
        result = count_image_tokens(image_input, args.model_path)

        # Display results
        display_single_image_results(
            result, args.model_path, f"Dummy image: {width} x {height}"
        )

    return 0


if __name__ == "__main__":
    main()
