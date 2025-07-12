#!/usr/bin/env python3

from PIL import Image, ImageDraw, ImageFont
import os
import random


def create_test_image(width, height, filename, color=None):
    """Create a test image with specified dimensions and color."""
    if color is None:
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    image = Image.new("RGB", (width, height), color)
    draw = ImageDraw.Draw(image)

    # Add some text
    try:
        # Try to use a default font, fall back to default if not available
        font = ImageFont.load_default()
    except:
        font = None

    text = f"{width}x{height}"
    if font:
        # Get text bounding box for centering
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = (width - text_width) // 2
        y = (height - text_height) // 2
        draw.text((x, y), text, fill=(255, 255, 255), font=font)

    # Add some geometric shapes
    draw.rectangle(
        [width // 4, height // 4, 3 * width // 4, 3 * height // 4],
        outline=(255, 255, 255),
        width=3,
    )

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
