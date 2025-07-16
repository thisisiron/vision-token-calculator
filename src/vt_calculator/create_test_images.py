#!/usr/bin/env python3

from PIL import Image
import os
import random


def create_test_image(width, height, filename, color=None):
    """Create a solid color test image with specified dimensions.

    Args:
        width (int): Image width in pixels
        height (int): Image height in pixels
        filename (str): Output file path
        color (tuple[int, int, int] | None): Optional RGB color
    """
    if color is None:
        color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
        )

    image = Image.new("RGB", (width, height), color)
    image.save(filename)
    print(f"Created: {filename}")


def main():
    # Create test_images directory
    os.makedirs("test_images", exist_ok=True)

    # Create test images with different sizes
    test_sizes = [
        (640, 480),
        (800, 600),
        (1024, 768),
        (1280, 720),
        (1920, 1080),
        (512, 512),
        (256, 256),
        (2048, 1536),
    ]

    for i, (width, height) in enumerate(test_sizes, 1):
        filename = f"test_images/test_{i}_{width}x{height}.jpg"
        create_test_image(width, height, filename)

    print(f"\nCreated {len(test_sizes)} test images in 'test_images' directory")


if __name__ == "__main__":
    main()
