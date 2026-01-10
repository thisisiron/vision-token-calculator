# Vision Token Calculator

[![PyPI version](https://badge.fury.io/py/vt-calc.svg)](https://badge.fury.io/py/vt-calc)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A Python tool for calculating the number of tokens generated when processing images with Vision Language Models (VLMs).

## Quick Start

```bash
pip install vt-calc
vt-calc --size 1920 1080                    # Calculate tokens for 1920x1080 image
vt-calc --image photo.jpg -m qwen2.5-vl     # Calculate tokens for an image
vt-calc --compare all --size 1920 1080      # Compare all models
```

## Features

- Calculate image/video tokens for VLMs
- **Multi-model comparison** - Compare token counts across multiple models
- Support both existing images and dummy images
- Support remote images via URL (http/https)
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

### Basic Commands

```bash
# Single image
vt-calc --image path/to/your/image.jpg

# Image from URL
vt-calc --image https://example.com/image.jpg

# Directory (batch processing)
vt-calc --image path/to/your/images_dir

# Dummy image with specific dimensions (Height x Width)
vt-calc --size 1920 1080

# Choose a model (default: qwen2.5-vl)
vt-calc --image photo.jpg -m internvl3
```

### Video Processing

```bash
# Calculate tokens for a video file
vt-calc --video path/to/video.mp4 -m qwen2.5-vl

# Specify frame sampling rate (FPS)
vt-calc --video video.mp4 --fps 2.0

# Limit maximum number of frames
vt-calc --video video.mp4 --max-frames 100
```

### Model Comparison

```bash
# Compare specific models (comma-separated)
vt-calc --image photo.jpg --compare qwen2.5-vl,internvl3,llava

# Compare all supported models
vt-calc --size 1920 1080 --compare all

# Compare models for video
vt-calc --video video.mp4 --compare qwen2.5-vl,llava-next --fps 2.0
```

### CLI Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--image` | `-i` | Path to image file, directory, or URL | - |
| `--video` | `-v` | Path to video file | - |
| `--size` | `-s` | Create dummy image (HEIGHT WIDTH) | - |
| `--model-name` | `-m` | Model name to use | `qwen2.5-vl` |
| `--compare` | `-c` | Compare models (comma-separated or `all`) | - |
| `--fps` | - | Frames per second for video sampling | - |
| `--max-frames` | - | Maximum frames to extract from video | - |
| `--duration` | - | Duration in seconds (dummy video) | - |

Supported input formats: `.jpg`, `.jpeg`, `.png`, `.webp` (case-insensitive)

### Example Output

<details>
<summary>Single Image Analysis</summary>

```text
Using dummy image (H×W): 1024×768
                        ╔══════════════════════════════╗
                        ║ VISION TOKEN ANALYSIS REPORT ║
                        ╚══════════════════════════════╝
╭───────────────────────────────── MODEL INFO ─────────────────────────────────╮
│   Model Name                qwen2.5-vl                                       │
╰──────────────────────────────────────────────────────────────────────────────╯
╭───────────────────────────────── IMAGE INFO ─────────────────────────────────╮
│   Original Size (H×W)       1024×768                                         │
│   Resized Size (H×W)        1036×756                                         │
╰──────────────────────────────────────────────────────────────────────────────╯
╭───────────────────────────────── TOKEN INFO ─────────────────────────────────╮
│   Image Token               999                                              │
╰──────────────────────────────────────────────────────────────────────────────╯
```

</details>

<details>
<summary>Model Comparison</summary>

```text
Comparing models for dummy image (H×W): 1920×1080

                           ╔════════════════════════╗
                           ║ IMAGE MODEL COMPARISON ║
                           ╚════════════════════════╝
                          Dummy image (H×W): 1920×1080
                          Resolution (H×W): 1920×1080

                                  Token Comparison
╭────────┬─────────────────────┬────────────┬────────────┬──────────────────┬────────╮
│  Rank  │ Model               │     Tokens │   px/token │ Efficiency       │ Status │
├────────┼─────────────────────┼────────────┼────────────┼──────────────────┼────────┤
│  🥇 1  │ deepseek-ocr-tiny   │         73 │     3591.0 │ █████████░ Best  │   ✓    │
│  🥈 2  │ deepseek-ocr-small  │        111 │     3690.1 │ █████████░       │   ✓    │
│  🥉 3  │ deepseek-ocr-base   │        273 │     3840.9 │ █████████░       │   ✓    │
│   4    │ deepseek-ocr-large  │        421 │     3891.7 │ █████████░       │   ✓    │
│   5    │ llava               │        576 │      196.0 │ █████████░       │   ✓    │
│   6    │ deepseek-ocr-gundam │      1,113 │      942.1 │ ████████░░       │   ✓    │
│   7    │ llava-next          │      1,968 │      129.1 │ ███████░░░       │   ✓    │
│   8    │ internvl3           │      2,306 │      696.3 │ ██████░░░░       │   ✓    │
│   9    │ qwen2-vl            │      2,693 │      783.4 │ ██████░░░░       │   ✓    │
│   10   │ qwen2.5-vl          │      2,693 │      783.4 │ ██████░░░░       │   ✓    │
│   11   │ llava-onevision     │      7,317 │      283.4 │ ░░░░░░░░░░       │   ✓    │
│   12   │ phi4-multimodal     │      7,553 │      744.0 │ ░░░░░░░░░░       │   ✓    │
╰────────┴─────────────────────┴────────────┴────────────┴──────────────────┴────────╯

╭────────────────────────────────── Summary ───────────────────────────────────╮
│ Best: deepseek-ocr-tiny (73 tokens)                                          │
│ Worst: phi4-multimodal (7,553 tokens)                                        │
│ Potential Savings: 7,480 tokens (99.0%)                                      │
╰──────────────────────────────────────────────────────────────────────────────╯
```

</details>

## Supported Models

| Model | Option | Image | Video |
|-------|--------|:-----:|:-----:|
| Qwen2-VL | `qwen2-vl` | ✓ | ✓ |
| Qwen2.5-VL | `qwen2.5-vl` | ✓ | ✓ |
| Qwen3-VL | `qwen3-vl` | ✓ | ✓ |
| LLaVA | `llava` | ✓ | ✓ |
| LLaVA-NeXT | `llava-next` | ✓ | |
| LLaVA-OneVision | `llava-onevision` | ✓ | ✓ |
| InternVL3 | `internvl3` | ✓ | ✓ |
| DeepSeek-OCR (tiny) | `deepseek-ocr-tiny` | ✓ | |
| DeepSeek-OCR (small) | `deepseek-ocr-small` | ✓ | |
| DeepSeek-OCR (base) | `deepseek-ocr-base` | ✓ | |
| DeepSeek-OCR (large) | `deepseek-ocr-large` | ✓ | |
| DeepSeek-OCR (gundam) | `deepseek-ocr-gundam` | ✓ | |
| Phi-4-Multimodal | `phi4-multimodal` | ✓ | |

## License

This project is licensed under the MIT License — see the `LICENSE` file for details.
