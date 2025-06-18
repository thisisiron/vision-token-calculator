import torch
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import os
import argparse
import numpy as np


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
    print(f"Loading processor: {model_path}")
    
    # Load only the processor (no need for the full model)
    processor = AutoProcessor.from_pretrained(model_path)
    
    if isinstance(image_input, str):
        image_input = Image.open(image_input)
        
    image_size = image_input.size
    print(f"Image size: {image_size[0]}x{image_size[1]}")
    
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
        'image_tokens': num_image_tokens,
        'image_size': image_size,
    }
    
    return processor_info


def parse_arguments():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Vision Token Calculator",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Image size arguments
    parser.add_argument(
        '--width',
        type=int,
        help='Width of dummy image (pixels)'
    )
    
    parser.add_argument(
        '--height',
        type=int,
        help='Height of dummy image (pixels)'
    )
    
    # Existing image path
    parser.add_argument(
        '--image', '-i',
        type=str,
        help='Path to existing image file (used instead of creating dummy image)'
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
    
    # Determine image input
    image_input = None
    
    try:
        
        if args.image:
            # Use existing image file
            image_input = args.image
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Specified image file not found: {image_input}")
            print(f"Using existing image: {image_input}")
            
        elif args.width and args.height:
            # Create dummy image (PIL object)
            image_input = create_dummy_image(args.width, args.height)
            
        else:
            # Default behavior - create a default dummy image or use test image
            if os.path.exists("test_image.jpg"):
                image_input = "test_image.jpg"
                print("Using default test image: test_image.jpg")
            else:
                # Create default 512x512 image
                print("Creating default dummy image (512x512)...")
                image_input = create_dummy_image(512, 512)
        
        # Calculate tokens
        result = count_image_tokens(image_input, getattr(args, 'model_path'))
        
        # Display results
        print("\nToken Analysis Results:")
        print("-" * 40)
        print(f"Image size: {result['image_size'][0]}x{result['image_size'][1]}")
        print(f"Image tokens: {result['image_tokens']}")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    main() 