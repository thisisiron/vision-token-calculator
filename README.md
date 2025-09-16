# Vision Token Calculator

A Python tool for calculating the number of tokens generated when processing images with Vision Language Models (VLMs).

## Features

- Calculate image tokens for VLMs
- Support both existing images and dummy images
- Simple command line interface (CLI)

## Installation

### Option 1: PyPI (recommended)

```bash
pip install vt-calc
```

### Option 2: From source (editable for development)

```bash
pip install -e .
```

## Usage

Using the vt-calc command (after pip install -e .)

After installing with `pip install -e .`, you can use the `vt-calc` command directly:

```bash
# Single image
vt-calc --image path/to/your/image.jpg

# Directory (batch processing)
vt-calc --image path/to/your/images_dir

# Dummy image with specific dimensions (Width x Height)
vt-calc --size 1920 1080

# Choose a short model name (default: qwen2.5-vl)
vt-calc --image path/to/your/image.jpg -m qwen2.5-vl

# Show help
vt-calc --help
```

### CLI options

- `-i, --image`: Path to an image file or a directory of images
- `-s, --size WIDTH HEIGHT`: Create a dummy image of the given size
- `-m, --model-name`: Short model name to use (default: `qwen2.5-vl`)

Supported input formats for directory processing: `.jpg`, `.jpeg`, `.png`, `.webp` (case-insensitive).

### Example output (single image)

```text
==================================================
 VISION TOKEN ANALYSIS RESULTS 
==================================================
Model                  : Qwen/Qwen2.5-VL-7B-Instruct
Image Source           : Existing image: examples/cat.jpg
Original Image Size (W x H)     : 1024 x 768
Resized Image Size (W x H) : 1024 x 768
Image Token            : <image>
Number of Image Tokens : 256
==================================================
```

### Example output (multi image)
```text
Processing directory: test_images/
Found 8 images to process...

[1/8] Processing: test_1_640x480.jpg ✓ (391 tokens)
[2/8] Processing: test_2_800x600.jpg ✓ (609 tokens)
[3/8] Processing: test_3_1024x768.jpg ✓ (999 tokens)
[4/8] Processing: test_4_1280x720.jpg ✓ (1196 tokens)
[5/8] Processing: test_5_1920x1080.jpg ✓ (2691 tokens)
[6/8] Processing: test_6_512x512.jpg ✓ (324 tokens)
[7/8] Processing: test_7_256x256.jpg ✓ (81 tokens)
[8/8] Processing: test_8_2048x1536.jpg ✓ (4015 tokens)

==================================================
 BATCH ANALYSIS RESULTS 
==================================================
Model                     : Qwen/Qwen2.5-VL-7B-Instruct
Total Images Processed    : 8
Average Vision Tokens     : 1288.2
Minimum Vision Tokens     : 81
Maximum Vision Tokens     : 4015
Standard Deviation        : 1370.5
==================================================
```

## Supported Models

Qwen2-VL, Qwen2.5-VL, InternVL3


## License

This project is licensed under the MIT License — see the `LICENSE` file for details.