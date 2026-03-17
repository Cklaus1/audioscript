"""Structured output formatting for agent-friendly CLI.

Key pattern: Rich Console writes to stderr, structured JSON to stdout.
This lets agents pipe stdout while humans see Rich progress on stderr.
"""

import json
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.table import Table

from audioscript import __version__


class OutputFormat(str, Enum):
    JSON = "json"
    TABLE = "table"
    QUIET = "quiet"
    YAML = "yaml"


class ExitCode:
    SUCCESS = 0
    TRANSCRIPTION_ERROR = 1
    AUTH_ERROR = 2
    VALIDATION_ERROR = 3
    INTERNAL_ERROR = 4

    @staticmethod
    def classify(exc: Exception) -> int:
        """Map exception types to exit codes."""
        msg = str(exc).lower()
        if "token" in msg or "auth" in msg or "huggingface" in msg or "credentials" in msg:
            return ExitCode.AUTH_ERROR
        if isinstance(exc, (ValueError, FileNotFoundError, TypeError)):
            return ExitCode.VALIDATION_ERROR
        if isinstance(exc, (RuntimeError, OSError)):
            return ExitCode.TRANSCRIPTION_ERROR
        return ExitCode.INTERNAL_ERROR


@dataclass
class CLIContext:
    """Shared state across all subcommands."""

    format: OutputFormat = OutputFormat.JSON
    dry_run: bool = False
    pipe: bool = False
    fields: Optional[List[str]] = None
    timeout: Optional[int] = None
    console: Console = field(default_factory=lambda: Console(stderr=True))
    start_time: float = field(default_factory=time.time)

    @property
    def is_structured(self) -> bool:
        """True when output should be machine-readable (no decoration)."""
        return self.format in (OutputFormat.JSON, OutputFormat.QUIET, OutputFormat.YAML)


def auto_detect_format(explicit: Optional[str], quiet: bool) -> OutputFormat:
    """Determine output format: explicit flag > --quiet > auto-detect (tty=table, pipe=json)."""
    if quiet:
        return OutputFormat.QUIET
    if explicit and explicit != "auto":
        return OutputFormat(explicit)
    if sys.stdout.isatty():
        return OutputFormat.TABLE
    return OutputFormat.JSON


def _filter_fields(data: Any, fields: List[str]) -> Any:
    """Filter data dict to only include requested dot-notation fields.

    Example: fields=["results.file", "results.status"] on
    {"results": [{"file": "a.mp3", "status": "ok", "extra": 1}]}
    returns {"results": [{"file": "a.mp3", "status": "ok"}]}
    """
    if not isinstance(data, dict):
        return data

    # Group sub-fields by top-level key
    top_level_keys: set = set()
    nested: Dict[str, List[str]] = {}
    for field_path in fields:
        parts = field_path.split(".", 1)
        key = parts[0]
        if len(parts) == 1:
            top_level_keys.add(key)
        else:
            nested.setdefault(key, []).append(parts[1])

    result = {}
    for key in top_level_keys:
        if key in data:
            result[key] = data[key]

    for key, sub_fields in nested.items():
        if key in top_level_keys or key not in data:
            continue
        value = data[key]
        if isinstance(value, list):
            result[key] = [
                _filter_fields(item, sub_fields) if isinstance(item, dict) else item
                for item in value
            ]
        elif isinstance(value, dict):
            result[key] = _filter_fields(value, sub_fields)
        else:
            result[key] = value

    return result


def emit(ctx: CLIContext, command: str, data: Any) -> None:
    """Emit a successful result in the configured format."""
    if ctx.fields:
        data = _filter_fields(data, ctx.fields)

    envelope = {
        "ok": True,
        "command": command,
        "data": data,
        "meta": {
            "version": __version__,
            "elapsed_seconds": round(time.time() - ctx.start_time, 2),
        },
    }
    if ctx.format == OutputFormat.JSON:
        json.dump(envelope, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
    elif ctx.format == OutputFormat.QUIET:
        json.dump(envelope, sys.stdout, default=str)
        sys.stdout.write("\n")
    elif ctx.format == OutputFormat.YAML:
        _print_yaml(envelope)
    elif ctx.format == OutputFormat.TABLE:
        _print_table(ctx.console, command, data)


def emit_ndjson(data: Dict[str, Any]) -> None:
    """Emit one NDJSON line to stdout (for --pipe batch streaming)."""
    json.dump(data, sys.stdout, default=str)
    sys.stdout.write("\n")
    sys.stdout.flush()


def emit_progress(ctx: CLIContext, file: str, percent: float, message: str = "") -> None:
    """Emit a structured progress event to stderr in structured mode.

    In table mode, Rich handles progress bars.
    In JSON/quiet/YAML mode, emit NDJSON progress events to stderr.
    """
    if ctx.is_structured:
        event = {
            "event": "progress",
            "file": file,
            "percent": round(percent, 1),
        }
        if message:
            event["message"] = message
        event["elapsed_seconds"] = round(time.time() - ctx.start_time, 2)
        sys.stderr.write(json.dumps(event, default=str) + "\n")
        sys.stderr.flush()


def emit_error(
    ctx: CLIContext,
    code: int,
    error_type: str,
    message: str,
    *,
    hint: Optional[str] = None,
    docs_url: Optional[str] = None,
) -> None:
    """Emit a structured error and exit with the typed code."""
    error_data: Dict[str, Any] = {
        "code": code,
        "error_type": error_type,
        "message": message,
    }
    if hint:
        error_data["hint"] = hint
    if docs_url:
        error_data["docs_url"] = docs_url

    envelope = {
        "ok": False,
        "error": error_data,
    }
    if ctx.is_structured:
        json.dump(envelope, sys.stdout, default=str)
        sys.stdout.write("\n")
    else:
        ctx.console.print(f"[bold red]Error ({error_type}):[/] {message}")
        if hint:
            ctx.console.print(f"[dim]Hint: {hint}[/]")
    raise SystemExit(code)


def _print_yaml(data: Any) -> None:
    """Render data as YAML to stdout."""
    try:
        import yaml
        sys.stdout.write(yaml.dump(data, default_flow_style=False, sort_keys=False))
    except ImportError:
        # Fallback to JSON if PyYAML not available
        json.dump(data, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")


def _print_table(console: Console, command: str, data: Any) -> None:
    """Render data as a Rich table to stderr (human-readable mode)."""
    if isinstance(data, dict):
        table = Table(title=command, show_header=True)
        table.add_column("Key", style="bold")
        table.add_column("Value")
        for key, value in data.items():
            if isinstance(value, (list, dict)):
                value = json.dumps(value, default=str)
            table.add_row(str(key), str(value))
        console.print(table)
    elif isinstance(data, list):
        if data and isinstance(data[0], dict):
            table = Table(title=command, show_header=True)
            keys = list(data[0].keys())
            for k in keys:
                table.add_column(k)
            for row in data:
                table.add_row(*[str(row.get(k, "")) for k in keys])
            console.print(table)
        else:
            for item in data:
                console.print(str(item))
    else:
        console.print(str(data))
