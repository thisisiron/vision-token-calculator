import shutil
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.align import Align
from rich import box

MAX_CONSOLE_WIDTH = 120
MIN_CONSOLE_WIDTH = 80
DEFAULT_TERMINAL_SIZE = (24, 80)

try:
    terminal_width = shutil.get_terminal_size(DEFAULT_TERMINAL_SIZE).columns
    console_width = min(terminal_width, MAX_CONSOLE_WIDTH)
    console_width = max(console_width, MIN_CONSOLE_WIDTH)
except Exception:
    console_width = MAX_CONSOLE_WIDTH

console = Console(width=console_width)

SEPARATOR = "=" * 72


def display_comparison_results(comparison: dict, source: str):
    """Display model comparison results."""
    is_video = comparison["type"] == "video_comparison"
    results = comparison["results"]
    summary = comparison["summary"]

    title = "VIDEO MODEL COMPARISON" if is_video else "IMAGE MODEL COMPARISON"
    console.print()
    console.print(
        Align.center(
            Panel(f"[bold cyan]{title}[/bold cyan]", box=box.DOUBLE, expand=False)
        )
    )

    if is_video:
        meta = comparison["video_metadata"]
        info_text = f"Resolution (H×W): {meta['height']}×{meta['width']} | Duration: {meta.get('duration', 0):.1f}s"
    else:
        h, w = comparison["image_size"]
        info_text = f"Resolution (H×W): {h}×{w}"

    console.print(Align.center(f"[dim]{source}[/dim]"))
    console.print(Align.center(f"[dim]{info_text}[/dim]"))
    console.print()

    table = Table(
        title="Token Comparison",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Rank", style="dim", width=6, justify="center")
    table.add_column("Model", style="bold", min_width=15)
    table.add_column("Tokens", justify="right", min_width=10)
    table.add_column("px/token", justify="right", min_width=10)
    table.add_column("Efficiency", min_width=16)
    table.add_column("Status", justify="center", width=8)

    sorted_results = sorted(
        results, key=lambda x: (x["tokens"] is None, x["tokens"] or float("inf"))
    )

    valid_tokens = [r["tokens"] for r in sorted_results if r["tokens"] is not None]
    max_tokens = max(valid_tokens) if valid_tokens else 1

    rank = 0
    for result in sorted_results:
        model = result["model"]
        tokens = result["tokens"]
        error = result["error"]
        details = result.get("details", {})

        if error:
            table.add_row(
                "-", model, "[red]N/A[/red]", "[red]N/A[/red]", f"[dim]{error[:20]}[/dim]", "[red]✗[/red]"
            )
        else:
            rank += 1
            efficiency = 1 - (tokens / max_tokens) if max_tokens > 0 else 0
            bar_filled = int(efficiency * 10)
            bar = "█" * bar_filled + "░" * (10 - bar_filled)

            # Calculate px/token using each model's resized_size
            if is_video:
                resized = details.get("resized_size", (0, 0))
                sampled_frames = details.get("sampled_frames", 1)
                total_pixels = resized[0] * resized[1] * sampled_frames
            else:
                resized = details.get("resized_size", comparison["image_size"])
                total_pixels = resized[0] * resized[1]
            px_per_token = total_pixels / tokens if tokens > 0 else 0

            if rank == 1:
                rank_str = "[green]🥇 1[/green]"
                tokens_str = f"[green bold]{tokens:,}[/green bold]"
                px_str = f"[green]{px_per_token:.1f}[/green]"
                bar_str = f"[green]{bar}[/green] Best"
            elif rank == 2 and len(valid_tokens) > 2:
                rank_str = "[yellow]🥈 2[/yellow]"
                tokens_str = f"[yellow]{tokens:,}[/yellow]"
                px_str = f"[yellow]{px_per_token:.1f}[/yellow]"
                bar_str = f"[yellow]{bar}[/yellow]"
            elif rank == 3:
                rank_str = "[#cd7f32]🥉 3[/#cd7f32]"
                tokens_str = f"[#cd7f32]{tokens:,}[/#cd7f32]"
                px_str = f"[#cd7f32]{px_per_token:.1f}[/#cd7f32]"
                bar_str = f"[#cd7f32]{bar}[/#cd7f32]"
            elif rank == len(valid_tokens):
                rank_str = f"[red]{rank}[/red]"
                tokens_str = f"[red]{tokens:,}[/red]"
                px_str = f"[red]{px_per_token:.1f}[/red]"
                bar_str = f"[red]{bar}[/red]"
            else:
                rank_str = str(rank)
                tokens_str = f"{tokens:,}"
                px_str = f"{px_per_token:.1f}"
                bar_str = bar

            table.add_row(rank_str, model, tokens_str, px_str, bar_str, "[green]✓[/green]")

    console.print(table)

    if summary:
        savings = summary["max_tokens"] - summary["min_tokens"]
        savings_pct = (
            (savings / summary["max_tokens"] * 100) if summary["max_tokens"] > 0 else 0
        )

        summary_text = (
            f"[green]Best:[/green] {summary['best_model']} ({summary['min_tokens']:,} tokens)\n"
            f"[red]Worst:[/red] {summary.get('worst_model', 'N/A')} ({summary['max_tokens']:,} tokens)\n"
            f"[cyan]Potential Savings:[/cyan] {savings:,} tokens ({savings_pct:.1f}%)"
        )

        console.print()
        console.print(
            Panel(
                summary_text,
                title="[bold]Summary[/bold]",
                border_style="cyan",
                box=box.ROUNDED,
            )
        )


def display_batch_results(stats: dict, model_name: str):
    """
    Display batch processing results using Rich tables.

    Args:
        stats (dict): Statistics dictionary from process_directory
        model_name (str): Short model name used for processing
    """
    console.print()
    table = Table(
        title="BATCH ANALYSIS REPORT",
        box=box.ROUNDED,
        show_header=False,
        title_justify="center",
    )
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


def print_processing_result(
    success: bool, token_count: Optional[int] = None, error: Optional[str] = None
):
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

    def print(
        self, result: dict, model_name: str, image_source: Optional[str] = None
    ) -> None:
        """
        Display single image/video analysis results using Rich tables.

        Args:
            result (dict): Token count result dictionary
            model_name (str): Short model name used for processing
            image_source (str): Optional description of image/video source
        """
        is_video = result.get("type") == "video"
        title = (
            "VISION TOKEN ANALYSIS REPORT"
            if not is_video
            else "VIDEO TOKEN ANALYSIS REPORT"
        )

        # Main Layout Table
        grid = Table.grid(expand=True)
        grid.add_column()

        # Title
        grid.add_row(
            Align.center(
                Panel(
                    f"[bold cyan]{title}[/bold cyan]",
                    box=box.DOUBLE,
                    expand=False,
                )
            )
        )

        # MODEL INFO
        model_table = Table(box=box.SIMPLE, show_header=False, expand=True)
        model_table.add_column("Key", style="cyan", ratio=1)
        model_table.add_column("Value", style="bold white", ratio=2)
        model_table.add_row("Model Name", model_name)

        # Add Processing Method
        processing_method = result.get("processing_method", "")
        if processing_method:
            method_display = {
                "native_resolution": "Native Resolution",
                "tile_based": "Tile-based",
                "fixed_resolution": "Fixed Resolution",
            }.get(processing_method, processing_method)
            model_table.add_row("Processing Method", method_display)

        grid.add_row(
            Panel(
                model_table,
                title="[bold]MODEL INFO[/bold]",
                border_style="blue",
                box=box.ROUNDED,
            )
        )

        input_table = Table(box=box.SIMPLE, show_header=False, expand=True)
        input_table.add_column("Key", style="cyan", ratio=1)
        input_table.add_column("Value", style="bold white", ratio=2)
        input_table.add_row("Source", image_source)

        if is_video:
            input_table.add_row("Duration", f"{result['duration']:.2f}s")
            input_table.add_row("FPS (Sampled)", f"{result['fps']:.2f}")
            input_table.add_row("Sampled Frames", str(result["sampled_frames"]))
            input_table.add_row(
                "Resolution (H×W)",
                f"{result['resized_size'][0]}×{result['resized_size'][1]} (Resized)",
            )
        else:
            input_table.add_row(
                "Original Size (H×W)",
                f"{result['image_size'][0]}×{result['image_size'][1]}",
            )
            input_table.add_row(
                "Resized Size (H×W)",
                f"{result['resized_size'][0]}×{result['resized_size'][1]}",
            )

        grid.add_row(
            Panel(
                input_table,
                title="[bold]VIDEO INFO[/bold]"
                if is_video
                else "[bold]IMAGE INFO[/bold]",
                border_style="green",
                box=box.ROUNDED,
            )
        )

        # TILE INFO (for tile-based models)
        if not is_video and processing_method == "tile_based":
            tile_table = Table(box=box.SIMPLE, show_header=False, expand=True)
            tile_table.add_column("Key", style="cyan", ratio=1)
            tile_table.add_column("Value", style="bold white", ratio=2)

            tile_table.add_row("Tile Size", str(result.get("tile_size", "N/A")))
            if "tile_grid" in result:
                tile_table.add_row(
                    "Tile Grid (H×W)",
                    f"{result['tile_grid'][0]}×{result['tile_grid'][1]}",
                )
            num_tiles = result.get("number_of_tiles", "N/A")
            global_note = " (incl. global)" if result.get("has_global_patch") else ""
            tile_table.add_row("Number of Tiles", f"{num_tiles}{global_note}")

            grid.add_row(
                Panel(
                    tile_table,
                    title="[bold]TILE INFO[/bold]",
                    border_style="yellow",
                    box=box.ROUNDED,
                )
            )

        # PATCH INFO
        patch_table = Table(box=box.SIMPLE, show_header=False, expand=True)
        patch_table.add_column("Key", style="cyan", ratio=1)
        patch_table.add_column("Value", style="bold white", ratio=2)

        if is_video:
            if "grid_size" in result:
                patch_table.add_row(
                    "Grid Size (per frame)",
                    f"{result['grid_size'][0]} x {result['grid_size'][1]}",
                )
        elif processing_method == "tile_based":
            # Tile-based: show patch info within tiles
            patch_table.add_row(
                "Patch Size (ViT)", str(result.get("patch_size", "N/A"))
            )
            patches_per_tile = result.get("patches_per_tile", "N/A")
            if patches_per_tile != "N/A":
                tile_size = result.get("tile_size", 0)
                patch_size = result.get("patch_size", 1)
                if tile_size and patch_size:
                    patches_dim = tile_size // patch_size
                    patch_table.add_row(
                        "Patches per Tile",
                        f"{patches_per_tile} ({patches_dim}×{patches_dim})",
                    )
            patch_table.add_row(
                "Total Patches", str(result.get("total_patches", "N/A"))
            )
        else:
            # Native resolution / Fixed resolution
            patch_table.add_row(
                "Patch Size (ViT)", str(result.get("patch_size", "N/A"))
            )
            if "patch_grid" in result:
                patch_table.add_row(
                    "Patch Grid (H×W)",
                    f"{result['patch_grid'][0]}×{result['patch_grid'][1]}",
                )
            patch_table.add_row(
                "Total Patches", str(result.get("total_patches", result.get("number_of_image_patches", "N/A")))
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

        items_to_show = []
        keys_to_check = [
            "number_of_video_tokens",
            "image_token",
            "image_newline_token",
            "image_separator_token",
            "image_start_token",
            "image_end_token",
        ]

        for key in keys_to_check:
            if key == "number_of_video_tokens" and key in result:
                items_to_show.append(("Total Video Tokens", result[key]))
                continue

            value = result.get(key)
            if isinstance(value, (list, tuple)) and len(value) == 2:
                token_symbol, token_count = value
                display_label = key.replace("_", " ").title()
                display_name = f"{display_label} ({token_symbol})"
                items_to_show.append((display_name, token_count))

        # Add total tokens if available (for DeepSeek-OCR)
        if "number_of_image_tokens" in result and result.get("image_newline_token"):
            items_to_show.append(("Total Image Tokens", result["number_of_image_tokens"]))

        if items_to_show:
            for display_name, token_count in items_to_show:
                token_info_table.add_row(display_name, str(token_count))

            # Calculate Pixels per Token
            resized_h, resized_w = result.get("resized_size", (0, 0))
            if is_video:
                total_pixels = resized_h * resized_w * result.get("sampled_frames", 1)
                total_tokens = result.get("number_of_video_tokens", 0)
            else:
                total_pixels = resized_h * resized_w
                # Use number_of_image_tokens if available, otherwise fall back to image_token
                total_tokens = result.get("number_of_image_tokens") or result.get("image_token", (None, 0))[1]

            if total_tokens and total_tokens > 0:
                pixels_per_token = total_pixels / total_tokens
                token_info_table.add_row(
                    "Pixels per Token", f"{pixels_per_token:.1f} px/token"
                )

            grid.add_row(
                Panel(
                    token_info_table,
                    title="[bold]TOKEN INFO[/bold]",
                    border_style="magenta",
                    box=box.ROUNDED,
                )
            )

            # TOKEN FORMAT
            if "token_format" in result or "image_token_format" in result:
                fmt = result.get("token_format", result.get("image_token_format"))
                format_panel = Panel(
                    Text(fmt, style="bold white", justify="center"),
                    title="[bold]TOKEN FORMAT[/bold]",
                    border_style="white",
                    box=box.ROUNDED,
                )
                grid.add_row(format_panel)

        console.print(grid)
