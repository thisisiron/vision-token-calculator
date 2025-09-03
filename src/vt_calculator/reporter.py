SEPARATOR = "=" * 50


def display_batch_results(stats: dict, model_path: str):
    """
    Display batch processing results.

    Args:
        stats (dict): Statistics dictionary from process_directory
        model_path (str): Model path used for processing
    """
    print("\n" + SEPARATOR)
    print(" BATCH ANALYSIS RESULTS ")
    print(SEPARATOR)

    print(f"Model                     : {model_path}")
    print(f"Total Images Processed    : {stats['total_processed']}")
    if stats["total_failed"] > 0:
        print(f"Total Images Failed       : {stats['total_failed']}")
    print(f"Average Vision Tokens     : {stats['average_tokens']:.1f}")
    print(f"Minimum Vision Tokens     : {stats['min_tokens']}")
    print(f"Maximum Vision Tokens     : {stats['max_tokens']}")
    if stats["std_deviation"] > 0:
        print(f"Standard Deviation        : {stats['std_deviation']:.1f}")

    print(SEPARATOR)

    # Show failed files if any
    if stats["failed_files"]:
        print("\nFailed Files:")
        for failed in stats["failed_files"]:
            print(f"  - {failed['filename']}: {failed['error']}")


def print_processing_status(filename: str, current: int, total: int):
    """
    Print processing status for batch operations.

    Args:
        filename (str): Name of file being processed
        current (int): Current file number
        total (int): Total number of files
    """
    print(f"[{current}/{total}] Processing: {filename} ", end="", flush=True)


def print_processing_result(success: bool, token_count: int = None, error: str = None):
    if success:
        print(f"✓ ({token_count} tokens)")
    else:
        print(f"✗ (Error: {error})")


def print_directory_info(directory_path: str, file_count: int):
    print(f"Processing directory: {directory_path}")
    print(f"Found {file_count} images to process...")
    print()


class Reporter:
    """Reporter for displaying single-image analysis results."""

    def __init__(self, label_width: int = 24):
        self.label_width = label_width

    def _print_kv(self, label: str, value: str, label_width: int = None) -> None:
        if label_width is None:
            label_width = self.label_width
        padding = " " * max(1, label_width - len(label))
        print(f"{label}{padding}: {value}")

    def display_single_image_results(
        self, result: dict, model_path: str, image_source: str = None
    ) -> None:
        """
        Display single image analysis results.

        Args:
            result (dict): Token count result dictionary
            model_path (str): Model path used for processing
            image_source (str): Optional description of image source
        """
        print("\n" + SEPARATOR)
        print(" VISION TOKEN ANALYSIS RESULTS ")
        print(SEPARATOR)

        # MODEL INFO
        print()
        print("[MODEL INFO]")
        self._print_kv("Model Name", model_path)

        # IMAGE INFO
        print()
        print("[IMAGE INFO]")
        if image_source:
            self._print_kv("Image Source", image_source)

        if "image_size" in result and isinstance(result["image_size"], (list, tuple)):
            self._print_kv(
                "Original Size (W x H)",
                f"{result['image_size'][0]} x {result['image_size'][1]}",
            )

        if "resized_size" in result and isinstance(
            result["resized_size"], (list, tuple)
        ):
            self._print_kv(
                "Resized Size (W x H)",
                f"{result['resized_size'][0]} x {result['resized_size'][1]}",
            )

        # Print token tuples like (token_name, count) in a compact block
        tokens_to_show = []
        for key in [
            "image_token",
            "image_start_token",
            "image_end_token",
        ]:
            value = result.get(key)
            if isinstance(value, (list, tuple)) and len(value) == 2:
                tokens_to_show.append(value)

        if tokens_to_show:
            print()
            print("[TOKEN INFO]")
            # Use a shared width so the colon aligns with the KV sections.
            max_name_len = max(len(str(name)) for name, _ in tokens_to_show)
            width = max(self.label_width, max_name_len)
            for token_name, token_count in tokens_to_show:
                name_padded = str(token_name).ljust(width)
                print(f"{name_padded} : {token_count}")

        print(SEPARATOR)


def display_single_image_results(
    result: dict, model_path: str, image_source: str = None
) -> None:
    """Compatibility wrapper that uses Reporter to display single image results."""
    reporter = Reporter()
    reporter.display_single_image_results(result, model_path, image_source)
