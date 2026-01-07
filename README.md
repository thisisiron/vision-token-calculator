# Vision Token Calculator

A Python tool for calculating the number of tokens generated when processing images with Vision Language Models (VLMs).

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

Using the vt-calc command (after pip install -e .)

After installing with `pip install -e .`, you can use the `vt-calc` command directly:

```bash
# Single image
vt-calc --image path/to/your/image.jpg

# Image from URL
vt-calc --image https://example.com/image.jpg

# Directory (batch processing)
vt-calc --image path/to/your/images_dir

# Dummy image with specific dimensions (Width x Height)
vt-calc --size 1920 1080

# Choose a short model name (default: qwen2.5-vl)
vt-calc --image path/to/your/image.jpg -m qwen2.5-vl

# Calculate tokens for a video file
vt-calc --video path/to/video.mp4 -m qwen2.5-vl

# Specify frame sampling rate (FPS)
vt-calc --video video.mp4 --fps 2.0

# Limit maximum number of frames
vt-calc --video video.mp4 --max-frames 100

# Compare multiple models (comma-separated)
vt-calc --image photo.jpg --compare qwen2.5-vl,internvl3,llava

# Compare all supported models
vt-calc --size 1920 1080 --compare all

# Compare models for video
vt-calc --video video.mp4 --compare qwen2.5-vl,llava-next --fps 2.0

# Show help
vt-calc --help
```

### CLI options

- `-i, --image`: Path to an image file, a directory of images, or an image URL
- `-v, --video`: Path to a video file
- `-s, --size HEIGHT WIDTH`: Create a dummy image of the given size
- `-m, --model-name`: Short model name to use (default: `qwen2.5-vl`)
- `-c, --compare`: Compare multiple models (comma-separated list or `all`)
- `--fps`: Frames per second for video sampling
- `--max-frames`: Maximum number of frames to extract from video
- `--duration`: Duration in seconds (for dummy video calculation)

Supported input formats for directory processing: `.jpg`, `.jpeg`, `.png`, `.webp` (case-insensitive).

### Example output (single image)

```text
Using dummy image: 1024 x 768
                        ╔══════════════════════════════╗
                        ║ VISION TOKEN ANALYSIS REPORT ║
                        ╚══════════════════════════════╝
╭───────────────────────────────── MODEL INFO ─────────────────────────────────╮
│                                                                              │
│   Model Name                qwen2.5-vl                                       │
│                                                                              │
╰──────────────────────────────────────────────────────────────────────────────╯
╭───────────────────────────────── IMAGE INFO ─────────────────────────────────╮
│                                                                              │
│   Image Source              Dummy image                                      │
│   Original Size (H x W)     1024 x 768                                       │
│   Resized Size (H x W)      1036 x 756                                       │
│                                                                              │
╰──────────────────────────────────────────────────────────────────────────────╯
╭───────────────────────────────── PATCH INFO ─────────────────────────────────╮
│                                                                              │
│   Patch Size (ViT)          14                                               │
│   Grid Size (H x W)         74 x 54                                          │
│   Number of Patches         3996                                             │
│                                                                              │
╰──────────────────────────────────────────────────────────────────────────────╯
╭───────────────────────────────── TOKEN INFO ─────────────────────────────────╮
│                                                                              │
│   Image Token               999                                              │
│   (<|image_pad|>)                                                            │
│   Image Start Token         1                                                │
│   (<|vision_start|>)                                                         │
│   Image End Token           1                                                │
│   (<|vision_end|>)                                                           │
│                                                                              │
╰──────────────────────────────────────────────────────────────────────────────╯
╭──────────────────────────────── TOKEN FORMAT ────────────────────────────────╮
│               <|vision_start|><|image_pad|>*999<|vision_end|>                │
╰──────────────────────────────────────────────────────────────────────────────╯
```

### Example output (multi image)

```text
Processing directory: test_images/
Found 8 images to process...

[1/8] Processing: test_1_640x480.jpg ✓ (393 tokens)
[2/8] Processing: test_2_800x600.jpg ✓ (611 tokens)
[3/8] Processing: test_3_1024x768.jpg ✓ (1001 tokens)
[4/8] Processing: test_4_1280x720.jpg ✓ (1198 tokens)
[5/8] Processing: test_5_1920x1080.jpg ✓ (2693 tokens)
[6/8] Processing: test_6_512x512.jpg ✓ (326 tokens)
[7/8] Processing: test_7_256x256.jpg ✓ (83 tokens)
[8/8] Processing: test_8_2048x1536.jpg ✓ (4017 tokens)

       BATCH ANALYSIS REPORT
╭────────────────────────┬────────────╮
│ Model                  │ qwen2.5-vl │
│ Total Images Processed │ 8          │
│ Average Vision Tokens  │ 1290.2     │
│ Minimum Vision Tokens  │ 83         │
│ Maximum Vision Tokens  │ 4017       │
│ Standard Deviation     │ 1370.5     │
╰────────────────────────┴────────────╯
```

### Example output (model comparison)

```text
Comparing models for dummy image: 1080 x 1920
                        ╔══════════════════════════════╗
                        ║    IMAGE MODEL COMPARISON    ║
                        ╚══════════════════════════════╝
                           Dummy image: 1080x1920
                           Resolution: 1920x1080

                         Token Comparison
╭────────┬─────────────────┬────────────┬──────────────────────┬──────────╮
│  Rank  │ Model           │     Tokens │ Efficiency           │  Status  │
├────────┼─────────────────┼────────────┼──────────────────────┼──────────┤
│ 🥇 1   │ qwen2.5-vl      │      2,693 │ ██████████ Best      │    ✓     │
│ 🥈 2   │ qwen2-vl        │      2,693 │ ██████████           │    ✓     │
│ 3      │ internvl3       │      3,584 │ ███████░░░           │    ✓     │
│ 4      │ llava-next      │      4,096 │ █████░░░░░           │    ✓     │
│ 5      │ llava           │      4,624 │ ████░░░░░░           │    ✓     │
╰────────┴─────────────────┴────────────┴──────────────────────┴──────────╯

╭─────────────────────────── Summary ───────────────────────────╮
│ Best: qwen2.5-vl (2,693 tokens)                               │
│ Worst: llava (4,624 tokens)                                   │
│ Potential Savings: 1,931 tokens (41.8%)                       │
╰───────────────────────────────────────────────────────────────╯
```

## Supported Models

| Model | Option |
|-------|--------|
| Qwen2-VL | qwen2-vl |
| Qwen2.5-VL | qwen2.5-vl |
| Qwen3-VL | qwen3-vl |
| InternVL3 | internvl3 |
| LLaVA | llava |
| LLaVA-NeXT | llava-next |
| LLaVA-OneVision | llava-onevision |

## License

This project is licensed under the MIT License — see the `LICENSE` file for details.
