# Vision Token Calculator

A Python tool for calculating the number of tokens generated when processing images with various Vision Language Models (VLMs).

## Features

- Calculate image tokens for different VLMs
- Support for both existing images and dummy images
- Detailed token analysis including image size and token count
- Easy-to-use command line interface

## Installation

### Option 1: Install from PyPI (recommended)

```bash
pip install vt-calc
```

### Option 2: Install as editable package (for development)

```bash
pip install -e .
```

### Option 3: Install dependencies only

```bash
pip install -r requirements.txt
```

## Usage

### Method 1: Using the vt-calc command (after pip install -e .)

After installing with `pip install -e .`, you can use the `vt-calc` command directly:

```bash
# Using an existing image
vt-calc --image path/to/your/image.jpg

# Creating a dummy image with specific dimensions
vt-calc --size 1920 1080

# Specifying a different model
vt-calc --image path/to/your/image.jpg --model-path "model/path"
```

### Method 2: Direct python execution

```bash
# Using an existing image
python calculate.py --image path/to/your/image.jpg

# Creating a dummy image with specific dimensions
python calculate.py --size 1920 1080

# Specifying a different model
python calculate.py --image path/to/your/image.jpg --model-path "model/path"
```

## Supported Models

| Model | Model size |
|------------|------------|
| Qwen2.5-VL | 3B / 7B / 32B / 72B |
| Gemma3 | 4B / 12B / 27B |
| InternVL3 | 1B / 2B / 8B / 14B / 38B / 78B |


## License

This project is licensed under the MIT License - see the LICENSE file for details. 