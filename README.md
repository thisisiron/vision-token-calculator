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

# Dummy image with specific dimensions
vt-calc --size 1920 1080

# Specify a different model (Hugging Face model id)
vt-calc --image path/to/your/image.jpg --model-path "Qwen/Qwen2.5-VL-7B-Instruct"

# Show help
vt-calc --help
```

### CLI options

- `-i, --image`: Path to an image file or a directory of images
- `-s, --size WIDTH HEIGHT`: Create a dummy image of the given size
- `-m, --model-path`: Model to use (default: `Qwen/Qwen2.5-VL-7B-Instruct`)

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

## Supported Models

| Model | Model size |
|------------|------------|
| Qwen2.5-VL | 3B / 7B / 32B / 72B |
| Gemma3 | 4B / 12B / 27B |
| InternVL3 | 1B / 2B / 8B / 14B / 38B / 78B |


## License

This project is licensed under the MIT License — see the `LICENSE` file for details.