"""Main CLI entry point for AudioScript."""

import glob
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from audioscript import __version__
from audioscript.config.settings import TranscriptionTier, get_settings
from audioscript.processors.audio_processor import AudioProcessor
from audioscript.utils.file_utils import ProcessingManifest

# Create typer app
app = typer.Typer(
    name="audioscript",
    help="CLI tool for audio transcription",
    add_completion=False,
)

# Create rich console for pretty output
console = Console()


def version_callback(value: bool) -> None:
    """Print the version and exit."""
    if value:
        console.print(f"AudioScript version: {__version__}")
        raise typer.Exit()


@app.command()
def main(
    input: Optional[str] = typer.Option(
        None,
        "--input",
        "-i",
        help="Input audio file or glob pattern",
    ),
    output_dir: str = typer.Option(
        "./output",
        "--output-dir",
        "-o",
        help="Directory to save transcription outputs",
    ),
    tier: TranscriptionTier = typer.Option(
        TranscriptionTier.DRAFT,
        "--tier",
        "-t",
        help="Transcription quality tier (draft or high_quality)",
    ),
    version: str = typer.Option(
        "1.0",
        "--version",
        help="Version of the transcription",
    ),
    clean_audio: bool = typer.Option(
        False,
        "--clean-audio",
        help="Clean audio before transcription",
    ),
    summarize: bool = typer.Option(
        False,
        "--summarize",
        help="Generate a summary of the transcription",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Force re-processing of already processed files",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Model to use for transcription (optional)",
    ),
    no_retry: bool = typer.Option(
        False,
        "--no-retry",
        help="Do not retry failed transcriptions",
    ),
    show_version: bool = typer.Option(
        False,
        "--version",
        "-v",
        callback=version_callback,
        is_eager=True,
        help="Show the version and exit",
    ),
) -> None:
    """Process audio files and generate transcriptions."""
    # Convert CLI args to dict for merging with file config
    cli_args = {
        "input": input,
        "output_dir": output_dir,
        "tier": tier,
        "version": version,
        "clean_audio": clean_audio,
        "summarize": summarize,
        "force": force,
        "model": model,
        "no_retry": no_retry,
    }

    # Get merged settings from CLI args and config file
    settings = get_settings(cli_args)

    # Validate input files exist
    if not settings["input"]:
        console.print("[bold red]Error:[/] No input files specified. Use --input option or config file.")
        raise typer.Exit(code=1)

    # Find input files using glob pattern
    input_pattern = settings["input"]
    input_files = glob.glob(input_pattern, recursive=True)

    if not input_files:
        console.print(f"[bold yellow]Warning:[/] No files found matching pattern: {input_pattern}")
        raise typer.Exit(code=0)

    # Create output directory if it doesn't exist
    output_path = Path(settings["output_dir"])
    output_path.mkdir(parents=True, exist_ok=True)

    # Show settings and file count
    console.print(f"[bold green]AudioScript[/] v{__version__}")
    console.print(f"Found [bold]{len(input_files)}[/] audio files to process")
    console.print(f"Transcription tier: [bold]{settings['tier']}[/]")
    console.print(f"Output directory: [bold]{output_path.absolute()}[/]")

    # Set up manifest file
    manifest_path = output_path / "manifest.json"
    manifest = ProcessingManifest(manifest_path)

    # Create processor
    processor = AudioProcessor(settings, manifest)

    # Process files with progress bar
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"[cyan]Processing {len(input_files)} files...", total=len(input_files))

        successful = 0
        failed = 0

        for audio_file in input_files:
            file_path = Path(audio_file)
            progress.update(task, description=f"[cyan]Processing {file_path.name}...")

            if processor.process_file(file_path):
                successful += 1
            else:
                failed += 1

            # Update progress
            progress.advance(task)

    # Show summary
    console.print(f"\n[bold green]Processing complete![/]")
    console.print(f"Successful: [bold green]{successful}[/]")

    if failed > 0:
        console.print(f"Failed: [bold red]{failed}[/]")

    # Show where to find results
    console.print(f"\nResults saved to: [bold]{output_path.absolute()}[/]")
    console.print(f"Manifest file: [bold]{manifest_path.absolute()}[/]")


if __name__ == "__main__":
    app()