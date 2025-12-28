from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

console = Console()

SEPARATOR = "=" * 72


def display_batch_results(stats: dict, model_name: str):
    """
    Display batch processing results using Rich tables.

    Args:
        stats (dict): Statistics dictionary from process_directory
        model_name (str): Short model name used for processing
    """
    table = Table(title="BATCH ANALYSIS REPORT", box=box.ROUNDED, show_header=False)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Model", model_name)
    table.add_row("Total Images Processed", str(stats["total_processed"]))
    if stats["total_failed"] > 0:
        table.add_row("Total Images Failed", str(stats["total_failed"]), style="red")
    table.add_row("Average Vision Tokens", f"{stats['average_tokens']:.1f}")
    table.add_row("Minimum Vision Tokens", str(stats["min_tokens"]))
    table.add_row("Maximum Vision Tokens", str(stats["max_tokens"]))
    if stats["std_deviation"] > 0:
        table.add_row("Standard Deviation", f"{stats['std_deviation']:.1f}")

    console.print(table)

    # Show failed files if any
    if stats["failed_files"]:
        console.print("\n[bold red]Failed Files:[/bold red]")
        for failed in stats["failed_files"]:
            console.print(f"  - {failed['filename']}: {failed['error']}")


def print_processing_status(filename: str, current: int, total: int):
    """
    Print processing status for batch operations.

    Args:
        filename (str): Name of file being processed
        current (int): Current file number
        total (int): Total number of files
    """
    console.print(f"[{current}/{total}] Processing: {filename} ", end="")


def print_processing_result(success: bool, token_count: int = None, error: str = None):
    if success:
        console.print(f"[green]✓ ({token_count} tokens)[/green]")
    else:
        console.print(f"[red]✗ (Error: {error})[/red]")


def print_directory_info(directory_path: str, file_count: int):
    console.print(f"[bold]Processing directory:[/bold] {directory_path}")
    console.print(f"Found {file_count} images to process...")
    console.print()


class Reporter:
    """Reporter for displaying single-image analysis results."""

    def __init__(self, label_width: int = 42):
        self.label_width = label_width

    def print(self, result: dict, model_name: str, image_source: str = None) -> None:
        """
        Display single image analysis results using Rich tables.

        Args:
            result (dict): Token count result dictionary
            model_name (str): Short model name used for processing
            image_source (str): Optional description of image source
        """
        # Main Layout Table
        grid = Table.grid(expand=True)
        grid.add_column()

        # Title
        grid.add_row(
            Panel(
                "[bold cyan]VISION TOKEN ANALYSIS REPORT[/bold cyan]",
                box=box.DOUBLE,
                expand=False,
            )
        )

        # MODEL INFO
        model_table = Table(box=box.SIMPLE, show_header=False, expand=True)
        model_table.add_column("Key", style="cyan", ratio=1)
        model_table.add_column("Value", style="bold white", ratio=2)
        model_table.add_row("Model Name", model_name)
        grid.add_row(
            Panel(
                model_table,
                title="[bold]MODEL INFO[/bold]",
                border_style="blue",
                box=box.ROUNDED,
            )
        )

        # IMAGE INFO
        image_table = Table(box=box.SIMPLE, show_header=False, expand=True)
        image_table.add_column("Key", style="cyan", ratio=1)
        image_table.add_column("Value", style="bold white", ratio=2)
        image_table.add_row("Image Source", image_source)
        image_table.add_row(
            "Original Size (H x W)",
            f"{result['image_size'][0]} x {result['image_size'][1]}",
        )
        image_table.add_row(
            "Resized Size (H x W)",
            f"{result['resized_size'][0]} x {result['resized_size'][1]}",
        )
        grid.add_row(
            Panel(
                image_table,
                title="[bold]IMAGE INFO[/bold]",
                border_style="green",
                box=box.ROUNDED,
            )
        )

        # PATCH INFO
        patch_table = Table(box=box.SIMPLE, show_header=False, expand=True)
        patch_table.add_column("Key", style="cyan", ratio=1)
        patch_table.add_column("Value", style="bold white", ratio=2)
        patch_table.add_row("Patch Size (ViT)", str(result["patch_size"]))
        if "tile_size" in result:
            patch_table.add_row("Tile Size", str(result["tile_size"]))
        if "grid_size" in result:
            patch_table.add_row(
                "Grid Size (H x W)",
                f"{result['grid_size'][0]} x {result['grid_size'][1]}",
            )
        patch_table.add_row(
            "Number of Patches",
            f"{result['number_of_image_patches']} {'(global patch)' if result['has_global_patch'] else ''}",
        )
        grid.add_row(
            Panel(
                patch_table,
                title="[bold]PATCH INFO[/bold]",
                border_style="yellow",
                box=box.ROUNDED,
            )
        )

        # TOKEN INFO
        token_info_table = Table(box=box.SIMPLE, show_header=False, expand=True)
        token_info_table.add_column("Key", style="cyan", ratio=1)
        token_info_table.add_column("Value", style="bold white", ratio=2)

        # Prepare token info items
        items_to_show = []
        for key in ["image_token", "image_start_token", "image_end_token"]:
            value = result.get(key)
            if isinstance(value, (list, tuple)) and len(value) == 2:
                token_symbol, token_count = value
                display_label = key.replace("_", " ").title()
                display_name = f"{display_label} ({token_symbol})"
                items_to_show.append((display_name, token_count))

        if items_to_show:
            for display_name, token_count in items_to_show:
                token_info_table.add_row(display_name, str(token_count))
            grid.add_row(
                Panel(
                    token_info_table,
                    title="[bold]TOKEN INFO[/bold]",
                    border_style="magenta",
                    box=box.ROUNDED,
                )
            )

            # TOKEN FORMAT
            format_panel = Panel(
                Text(
                    result["image_token_format"], style="bold white", justify="center"
                ),
                title="[bold]TOKEN FORMAT[/bold]",
                border_style="white",
                box=box.ROUNDED,
            )
            grid.add_row(format_panel)

        console.print(grid)
