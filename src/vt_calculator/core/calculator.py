import os

from ..setup_env import setup_quiet_environment

from PIL import Image
from ..utils import (
    get_image_files,
    calculate_mean,
    calculate_stdev,
    create_dummy_image,
    is_url,
    load_image_from_url,
    is_video,
)
from ..video import get_video_metadata
from ..parser import parse_arguments
from ..reporter import (
    display_batch_results,
    display_comparison_results,
    print_processing_status,
    print_processing_result,
    print_directory_info,
)
from ..reporter import Reporter
from ..analysts import load_analyst, DEFAULT_MODEL, SUPPORTED_MODELS
from typing import List, Dict, Any


setup_quiet_environment()


def parse_compare_models(compare_str: str) -> List[str]:
    """Parse --compare argument into list of model names.

    Args:
        compare_str: Comma-separated model names or 'all'

    Returns:
        List of valid model short names

    Raises:
        ValueError: If any model name is invalid
    """
    if compare_str.lower() == "all":
        return sorted(SUPPORTED_MODELS)

    models = [m.strip().lower() for m in compare_str.split(",")]
    invalid = [m for m in models if m not in SUPPORTED_MODELS]
    if invalid:
        raise ValueError(
            f"Unsupported models: {invalid}. Supported: {sorted(SUPPORTED_MODELS)}"
        )

    return models


def _extract_total_tokens(result: dict) -> int:
    """Extract total token count from analyst result.

    Handles different result formats from various VLM analysts.

    Args:
        result: Token calculation result dictionary

    Returns:
        Total number of tokens
    """
    # If pre-calculated total exists
    if "number_of_image_tokens" in result:
        return int(result["number_of_image_tokens"])

    # Sum up individual token components (Qwen-style)
    total = 0
    for key in ["image_token", "image_start_token", "image_end_token"]:
        value = result.get(key)
        if isinstance(value, (list, tuple)) and len(value) == 2:
            total += int(value[1])

    if total > 0:
        return total

    # Fallback for single token format (LLaVA-style)
    image_token = result.get("image_token")
    if isinstance(image_token, (list, tuple)) and len(image_token) == 2:
        return int(image_token[1])

    return 0


def compare_image_tokens(
    image_input,
    model_names: List[str],
) -> Dict[str, Any]:
    """Compare image tokens across multiple models."""
    if isinstance(image_input, str):
        if is_url(image_input):
            image = load_image_from_url(image_input)
        else:
            image = Image.open(image_input)
    else:
        image = image_input

    width, height = image.size
    image_size = (height, width)

    results = []
    for model_name in model_names:
        try:
            analyst = load_analyst(model_name)
            result = analyst.calculate_image(image_size)
            total_tokens = _extract_total_tokens(result)

            results.append({
                "model": model_name,
                "tokens": total_tokens,
                "details": result,
                "error": None,
            })
        except Exception as e:
            results.append({
                "model": model_name,
                "tokens": None,
                "details": None,
                "error": str(e),
            })

    valid_results = [r for r in results if r["tokens"] is not None]
    valid_results.sort(key=lambda x: x["tokens"])

    summary = {}
    if valid_results:
        summary = {
            "min_tokens": valid_results[0]["tokens"],
            "max_tokens": valid_results[-1]["tokens"],
            "best_model": valid_results[0]["model"],
            "worst_model": valid_results[-1]["model"],
        }

    return {
        "type": "image_comparison",
        "image_size": image_size,
        "results": results,
        "summary": summary,
    }


def compare_video_tokens(
    video_input,
    model_names: List[str],
    fps: float | None = None,
    max_frames: int | None = None,
) -> Dict[str, Any]:
    """Compare video tokens across multiple models."""
    if isinstance(video_input, dict):
        metadata = video_input
    else:
        metadata = get_video_metadata(video_input)

    results = []
    for model_name in model_names:
        try:
            analyst = load_analyst(model_name)
            result = analyst.calculate_video(metadata, fps=fps, max_frames=max_frames)
            total_tokens = result.get("number_of_video_tokens", 0)

            results.append({
                "model": model_name,
                "tokens": total_tokens,
                "details": result,
                "error": None,
            })
        except NotImplementedError:
            results.append({
                "model": model_name,
                "tokens": None,
                "details": None,
                "error": "Video not supported",
            })
        except Exception as e:
            results.append({
                "model": model_name,
                "tokens": None,
                "details": None,
                "error": str(e),
            })

    valid_results = [r for r in results if r["tokens"] is not None]
    valid_results.sort(key=lambda x: x["tokens"])

    summary = {}
    if valid_results:
        summary = {
            "min_tokens": valid_results[0]["tokens"],
            "max_tokens": valid_results[-1]["tokens"],
            "best_model": valid_results[0]["model"],
            "worst_model": valid_results[-1]["model"],
        }

    return {
        "type": "video_comparison",
        "video_metadata": metadata,
        "results": results,
        "summary": summary,
    }


def count_image_tokens(image_input, model_name: str = DEFAULT_MODEL):
    """
    Calculate the number of image tokens generated when processing an image.

    Args:
        image_input: Either a file path (str) or PIL Image object
        model_name (str): Short model name to use for processing

    Returns:
        dict: Dictionary containing token counts and details
    """

    # Build analyst via factory (handles aliases and config requirements)
    analyst = load_analyst(model_name)

    if isinstance(image_input, str):
        if is_url(image_input):
            image_input = load_image_from_url(image_input)
        else:
            image_input = Image.open(image_input)

    # PIL.Image.size -> (width, height); analyst expects (height, width)
    width, height = image_input.size
    result = analyst.calculate_image((height, width))

    # Backward-compatible total token count for batch statistics
    if (
        isinstance(result.get("image_token"), (list, tuple))
        and isinstance(result.get("image_start_token"), (list, tuple))
        and isinstance(result.get("image_end_token"), (list, tuple))
    ):
        total_tokens = (
            int(result["image_token"][1])
            + int(result["image_start_token"][1])
            + int(result["image_end_token"][1])
        )
        result["number_of_image_tokens"] = total_tokens

    return result


def count_video_tokens(
    video_path: str,
    model_name: str = DEFAULT_MODEL,
    fps: float | None = None,
    max_frames: int | None = None,
):
    analyst = load_analyst(model_name)
    metadata = get_video_metadata(video_path)
    return analyst.calculate_video(metadata, fps=fps, max_frames=max_frames)


def process_directory(directory_path: str, model_name: str):
    """
    Process all images in a directory and calculate batch statistics.

    Args:
        directory_path (str): Path to directory containing images
        model_name (str): Short model name to use for processing

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

        result = count_image_tokens(image_file, model_name)
        token_count = int(result["number_of_image_tokens"])
        token_counts.append(token_count)
        processed_files.append(
            {"filename": filename, "size": result["image_size"], "tokens": token_count}
        )
        print_processing_result(True, token_count)

    # Calculate statistics
    stats = {
        "total_processed": len(processed_files),
        "total_failed": 0,
        "average_tokens": calculate_mean(token_counts),
        "min_tokens": min(token_counts),
        "max_tokens": max(token_counts),
        "std_deviation": calculate_stdev(token_counts),
        "processed_files": processed_files,
        "failed_files": [],
    }

    return stats


def main():
    """
    Main function to demonstrate image token counting.
    """
    args = parse_arguments()

    if args.compare:
        try:
            model_names = parse_compare_models(args.compare)
        except ValueError as e:
            print(f"Error: {e}")
            return

        if args.video:
            if is_video(args.video):
                comparison = compare_video_tokens(
                    args.video, model_names, args.fps, args.max_frames
                )
                display_comparison_results(comparison, f"Video: {args.video}")
            else:
                print(f"Error: {args.video} is not a valid video file.")

        elif args.image:
            if os.path.isdir(args.image):
                print("Error: --compare does not support directories yet.")
                return
            comparison = compare_image_tokens(args.image, model_names)
            source = f"URL: {args.image}" if is_url(args.image) else args.image
            display_comparison_results(comparison, source)

        elif args.size:
            height, width = args.size

            if args.fps is not None or args.duration is not None:
                fps = args.fps if args.fps else 1.0
                duration = args.duration if args.duration else 1.0
                total_frames = int(duration * fps)

                metadata = {
                    "width": width,
                    "height": height,
                    "duration": duration,
                    "total_frames": total_frames,
                }

                print(f"Comparing models for dummy video: {width}x{height} @ {fps}fps, {duration}s")
                comparison = compare_video_tokens(
                    metadata, model_names, args.fps, args.max_frames
                )
                display_comparison_results(comparison, f"Dummy video: {width}x{height}")
            else:
                image_input = create_dummy_image(height, width)
                print(f"Comparing models for dummy image: {height} x {width}")
                comparison = compare_image_tokens(image_input, model_names)
                display_comparison_results(comparison, f"Dummy image: {height}x{width}")
        return

    if args.video:
        if is_video(args.video):
            result = count_video_tokens(
                args.video, args.model_name, args.fps, args.max_frames
            )
            reporter = Reporter()
            reporter.print(result, args.model_name, f"Video: {args.video}")
        else:
            print(f"Error: {args.video} is not a valid video file.")

    elif args.image:
        if is_url(args.image):
            print(f"Loading image from URL: {args.image}")
            result = count_image_tokens(args.image, args.model_name)
            reporter = Reporter()
            reporter.print(result, args.model_name, f"URL: {args.image}")
        elif os.path.isdir(args.image):
            stats = process_directory(args.image, args.model_name)
            display_batch_results(stats, args.model_name)
        else:
            print(f"Using existing image: {args.image}")

            result = count_image_tokens(args.image, args.model_name)

            reporter = Reporter()
            reporter.print(result, args.model_name, f"{args.image}")

    elif args.size:
        height, width = args.size

        if args.fps is not None or args.duration is not None:
            # Treat as dummy video
            fps = args.fps if args.fps else 1.0
            duration = args.duration if args.duration else 1.0
            total_frames = int(duration * fps)

            # Construct metadata directly without creating file
            metadata = {
                "width": width,
                "height": height,
                "duration": duration,
                "total_frames": total_frames,
            }

            print(f"Using dummy video: {width}x{height} @ {fps}fps, {duration}s")
            
            analyst = load_analyst(args.model_name)
            result = analyst.calculate_video(metadata, args.fps, args.max_frames)

            reporter = Reporter()
            reporter.print(result, args.model_name, "Dummy video")
        else:
            # Treat as dummy image
            image_input = create_dummy_image(height, width)
            print(f"Using dummy image: {height} x {width}")

            # Calculate tokens
            result = count_image_tokens(image_input, args.model_name)

            # Display results using Reporter
            reporter = Reporter()
            reporter.print(result, args.model_name, "Dummy image")



if __name__ == "__main__":
    main()
