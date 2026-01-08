"""CLI entry point for vt-calc."""

from .parser import parse_arguments
from .handlers import (
    handle_compare_command,
    handle_video_command,
    handle_image_command,
    handle_size_command,
)


def main():
    """Main CLI entry point."""
    args = parse_arguments()

    if args.compare:
        handle_compare_command(args)
    elif args.video:
        handle_video_command(args)
    elif args.image:
        handle_image_command(args)
    elif args.size:
        handle_size_command(args)


if __name__ == "__main__":
    main()
