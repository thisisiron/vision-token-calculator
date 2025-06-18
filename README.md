# Vision Token Calculator

A Python tool for calculating the number of tokens generated when processing images with various Vision Language Models (VLMs).

## Features

- Calculate image tokens for different VLMs
- Support for both existing images and dummy images
- Detailed token analysis including image size and token count
- Easy-to-use command line interface

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Using an existing image

```bash
python vision_token_calculator.py --image path/to/your/image.jpg
```

### Creating a dummy image with specific dimensions

```bash
python vision_token_calculator.py --size 1920 1080
```

### Specifying a different model

```bash
python vision_token_calculator.py --image path/to/your/image.jpg --model-path "model/path"
```

## Supported Models

| Model | Model size |
|------------|------------|
| Qwen2.5-VL | 3B / 7B / 32B / 72B |
| Gemma3 | 4B / 12B / 27B |
| InternVL3 | 1B / 2B / 8B / 14B / 38B / 78B |


## License

This project is licensed under the MIT License - see the LICENSE file for details. 