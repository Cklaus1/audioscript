"""Service command — install audioscript sync as a systemd user service.

Generates a systemd user unit file for ``audioscript sync --watch`` and
provides install/start/stop/status/uninstall subcommands.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import typer

from audioscript.cli.output import CLIContext, ExitCode, emit, emit_error

service_app = typer.Typer(
    name="service",
    help="Install audioscript sync as a background service.",
)

SERVICE_NAME = "audioscript-sync"
UNIT_FILENAME = f"{SERVICE_NAME}.service"


def _unit_dir() -> Path:
    """Return the systemd user unit directory."""
    return Path.home() / ".config" / "systemd" / "user"


def _unit_path() -> Path:
    """Return the full path to the service unit file."""
    return _unit_dir() / UNIT_FILENAME


def _audioscript_bin() -> str:
    """Find the audioscript executable path."""
    which = shutil.which("audioscript")
    if which:
        return which
    # Fall back to running as a module
    return f"{sys.executable} -m audioscript"


def _generate_unit(working_dir: str | None = None) -> str:
    """Generate the systemd user unit file contents."""
    exec_start = f"{_audioscript_bin()} sync --watch"
    work_dir = working_dir or os.getcwd()

    # Collect relevant env vars to forward
    env_lines = []
    for var in ("HF_TOKEN", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "NVIDIA_API_KEY"):
        value = os.environ.get(var)
        if value:
            env_lines.append(f"Environment={var}={value}")

    env_section = "\n".join(env_lines)

    return f"""\
[Unit]
Description=AudioScript Sync Watch
After=network.target

[Service]
ExecStart={exec_start}
WorkingDirectory={work_dir}
{env_section}
Restart=on-failure
RestartSec=10

[Install]
WantedBy=default.target
"""


def _run_systemctl(*args: str) -> subprocess.CompletedProcess[str]:
    """Run a systemctl --user command."""
    return subprocess.run(
        ["systemctl", "--user", *args],
        capture_output=True,
        text=True,
    )


@service_app.command(name="install")
def install(
    ctx: typer.Context,
    working_dir: str = typer.Option(
        None, "--working-dir", "-w",
        help="Working directory for the service (default: current dir).",
    ),
) -> None:
    """Write the systemd user service file and enable it.

    Example: audioscript service install --working-dir /home/user/audio
    """
    cli: CLIContext = ctx.obj

    unit_dir = _unit_dir()
    unit_dir.mkdir(parents=True, exist_ok=True)

    unit_content = _generate_unit(working_dir)
    unit_path = _unit_path()
    unit_path.write_text(unit_content, encoding="utf-8")

    # Reload systemd and enable
    _run_systemctl("daemon-reload")
    result = _run_systemctl("enable", SERVICE_NAME)

    if result.returncode != 0:
        emit_error(
            cli, ExitCode.INTERNAL_ERROR, "service",
            f"Failed to enable service: {result.stderr.strip()}",
        )
        return

    cli.console.print(f"[green]Service installed:[/] {unit_path}")
    cli.console.print(f"Run [bold]audioscript service start[/] to start it.")

    emit(cli, "service", {
        "action": "install",
        "unit_path": str(unit_path),
        "status": "enabled",
    })


@service_app.command(name="start")
def start(ctx: typer.Context) -> None:
    """Start the audioscript-sync service."""
    cli: CLIContext = ctx.obj

    if not _unit_path().exists():
        emit_error(
            cli, ExitCode.VALIDATION_ERROR, "service",
            "Service not installed. Run: audioscript service install",
        )
        return

    result = _run_systemctl("start", SERVICE_NAME)
    if result.returncode != 0:
        emit_error(
            cli, ExitCode.INTERNAL_ERROR, "service",
            f"Failed to start service: {result.stderr.strip()}",
        )
        return

    cli.console.print(f"[green]Service started:[/] {SERVICE_NAME}")
    emit(cli, "service", {"action": "start", "status": "started"})


@service_app.command(name="stop")
def stop(ctx: typer.Context) -> None:
    """Stop the audioscript-sync service."""
    cli: CLIContext = ctx.obj

    result = _run_systemctl("stop", SERVICE_NAME)
    if result.returncode != 0:
        emit_error(
            cli, ExitCode.INTERNAL_ERROR, "service",
            f"Failed to stop service: {result.stderr.strip()}",
        )
        return

    cli.console.print(f"[yellow]Service stopped:[/] {SERVICE_NAME}")
    emit(cli, "service", {"action": "stop", "status": "stopped"})


@service_app.command(name="status")
def status(ctx: typer.Context) -> None:
    """Show the status of the audioscript-sync service."""
    cli: CLIContext = ctx.obj

    result = _run_systemctl("status", SERVICE_NAME)

    # Parse active state
    active_line = ""
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("Active:"):
            active_line = stripped
            break

    if not _unit_path().exists():
        cli.console.print(f"[dim]Service not installed.[/]")
        emit(cli, "service", {"action": "status", "installed": False, "active": False})
        return

    is_active = "active (running)" in result.stdout
    cli.console.print(result.stdout)

    emit(cli, "service", {
        "action": "status",
        "installed": True,
        "active": is_active,
        "active_line": active_line,
        "unit_path": str(_unit_path()),
    })


@service_app.command(name="uninstall")
def uninstall(ctx: typer.Context) -> None:
    """Stop, disable, and remove the audioscript-sync service file."""
    cli: CLIContext = ctx.obj

    unit_path = _unit_path()
    if not unit_path.exists():
        cli.console.print("[dim]Service not installed — nothing to remove.[/]")
        emit(cli, "service", {"action": "uninstall", "status": "not_installed"})
        return

    _run_systemctl("stop", SERVICE_NAME)
    _run_systemctl("disable", SERVICE_NAME)

    unit_path.unlink(missing_ok=True)
    _run_systemctl("daemon-reload")

    cli.console.print(f"[yellow]Service uninstalled:[/] {unit_path}")
    emit(cli, "service", {"action": "uninstall", "status": "removed"})
