"""CLI command handlers."""

import os

from ..api import (
    count_image_tokens,
    count_video_tokens,
    process_directory,
    parse_compare_models,
    compare_image_tokens,
    compare_video_tokens,
)
from ..analysts import load_analyst
from ..reporter import Reporter, display_batch_results, display_comparison_results
from ..utils import create_dummy_image, is_url, is_video


def handle_compare_command(args):
    """Handle --compare flag for model comparison.

    Args:
        args: Parsed CLI arguments
    """
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

            print(
                f"Comparing models for dummy video (H×W): {height}×{width} @ {fps}fps, {duration}s"
            )
            comparison = compare_video_tokens(
                metadata, model_names, args.fps, args.max_frames
            )
            display_comparison_results(
                comparison, f"Dummy video (H×W): {height}×{width}"
            )
        else:
            image_input = create_dummy_image(height, width)
            print(f"Comparing models for dummy image (H×W): {height}×{width}")
            comparison = compare_image_tokens(image_input, model_names)
            display_comparison_results(
                comparison, f"Dummy image (H×W): {height}×{width}"
            )


def handle_video_command(args):
    """Handle --video flag.

    Args:
        args: Parsed CLI arguments
    """
    if is_video(args.video):
        result = count_video_tokens(
            args.video, args.model_name, args.fps, args.max_frames
        )
        reporter = Reporter()
        reporter.print(result, args.model_name, f"Video: {args.video}")
    else:
        print(f"Error: {args.video} is not a valid video file.")


def handle_image_command(args):
    """Handle --image flag.

    Args:
        args: Parsed CLI arguments
    """
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


def handle_size_command(args):
    """Handle --size flag for dummy image/video.

    Args:
        args: Parsed CLI arguments
    """
    height, width = args.size

    if args.fps is not None or args.duration is not None:
        # Treat as dummy video
        fps = args.fps if args.fps else 1.0
        duration = args.duration if args.duration else 1.0
        total_frames = int(duration * fps)

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
        reporter.print(result, args.model_name, f"Dummy video (H×W): {height}×{width}")
    else:
        # Treat as dummy image
        image_input = create_dummy_image(height, width)
        print(f"Using dummy image: {height} x {width}")

        result = count_image_tokens(image_input, args.model_name)

        reporter = Reporter()
        reporter.print(result, args.model_name, f"Dummy image (H×W): {height}×{width}")
