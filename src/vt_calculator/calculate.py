import torch
import os
import warnings

# Import and setup quiet environment to disable HF warnings
try:
    from .setup_env import setup_quiet_environment
    setup_quiet_environment()
except ImportError:
    # Fallback: minimal setup if setup_env is not available
    import os
    import warnings
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
    warnings.filterwarnings('ignore')

from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import argparse
import numpy as np
import glob
from .printer import (
    display_batch_results,
    display_single_image_results,
    print_processing_status,
    print_processing_result,
    print_directory_info
)


def calculate_mean(values):
    """
    Calculate the arithmetic mean of a list of numbers.
    
    Args:
        values (list): List of numeric values
        
    Returns:
        float: Mean value
    """
    if not values:
        return 0.0
    return sum(values) / len(values)


def calculate_stdev(values):
    """
    Calculate the sample standard deviation of a list of numbers.
    
    Args:
        values (list): List of numeric values
        
    Returns:
        float: Standard deviation
    """
    if len(values) < 2:
        return 0.0
    
    mean = calculate_mean(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return variance ** 0.5


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
    
    # Process vision information
    if "qwen" in processor.__class__.__name__.lower():
        image_inputs, video_inputs = process_vision_info(messages)
    else:
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
    
    # Calculate token counts
    input_ids = inputs['input_ids']
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
        'number_of_image_tokens': num_image_tokens,
        'image_size': image_input.size,
        'image_token': processor.tokenizer.decode(image_token_index)
    }
    
    return processor_info


def get_image_files(directory_path: str):
    """
    Get all image files from the specified directory.
    
    Args:
        directory_path (str): Path to directory containing images
        
    Returns:
        list: List of image file paths
    """
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.tiff', '*.tif', '*.webp']
    image_files = []
    
    for ext in image_extensions:
        for case_ext in [ext, ext.upper()]:
            pattern = os.path.join(directory_path, case_ext)
            image_files += glob.glob(pattern)
    
    return sorted(image_files)


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
        token_count = int(result['number_of_image_tokens'])
        token_counts.append(token_count)
        processed_files.append({
            'filename': filename,
            'size': result['image_size'],
            'tokens': token_count
        })
        print_processing_result(True, token_count)
    
    # Calculate statistics
    stats = {
        'total_processed': len(processed_files),
        'total_failed': 0,
        'average_tokens': calculate_mean(token_counts),
        'min_tokens': min(token_counts),
        'max_tokens': max(token_counts),
        'std_deviation': calculate_stdev(token_counts),
        'processed_files': processed_files,
        'failed_files': []
    }
    
    return stats




def parse_arguments():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Vision Token Calculator - Process single images or batch process directories"
    )
    
    # Create mutually exclusive group for input methods
    input_group = parser.add_mutually_exclusive_group(required=True)
    
    # Image size argument
    input_group.add_argument(
        '--size', '-s',
        type=int,
        nargs=2,
        metavar=('WIDTH', 'HEIGHT'),
        help='Size of dummy image in format "WIDTH HEIGHT" (e.g., "1920 1080")'
    )
    
    # Existing image path
    input_group.add_argument(
        '--image', '-i',
        type=str,
        help='Path to existing image file'
    )
    
    # Directory path for batch processing
    input_group.add_argument(
        '--img_dir', '-d',
        type=str,
        help='Directory path containing images for batch processing'
    )
    
    # Model path
    parser.add_argument(
        '--model-path', '-m',
        type=str,
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help='Model path to use (default: Qwen/Qwen2.5-VL-7B-Instruct)'
    )
    
    return parser.parse_args()


def main():
    """
    Main function to demonstrate image token counting.
    """
    args = parse_arguments()
    
    # Check if directory processing is requested
    if args.img_dir:
        # Batch processing mode
        # Process directory
        stats = process_directory(args.img_dir, args.model_path)
        display_batch_results(stats, args.model_path)
        
    elif args.image:
        # Use existing image file
        print(f"Using existing image: {args.image}")
        
        # Calculate tokens
        result = count_image_tokens(args.image, args.model_path)
        
        # Display results
        display_single_image_results(result, args.model_path, f"Existing image: {args.image}")
        
    elif args.size:
        # Create dummy image with specified dimensions
        width, height = args.size
        image_input = create_dummy_image(width, height)
        print(f"Using dummy image: {width} x {height}")
        
        # Calculate tokens
        result = count_image_tokens(image_input, args.model_path)
        
        # Display results
        display_single_image_results(result, args.model_path, f"Dummy image: {width} x {height}")
    
    return 0
        


if __name__ == "__main__":
    main()
