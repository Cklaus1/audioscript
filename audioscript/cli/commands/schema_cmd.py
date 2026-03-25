"""Schema introspection command — lets agents discover capabilities."""

import typer

from audioscript.cli.output import CLIContext, emit

_AVAILABLE_MODELS = [
    "tiny", "tiny.en", "base", "base.en", "small", "small.en",
    "medium", "medium.en", "large-v1", "large-v2", "large-v3", "turbo",
]

schema_app = typer.Typer(name="schema", help="Introspect AudioScript capabilities.")


@schema_app.command("models")
def schema_models(ctx: typer.Context) -> None:
    """List available Whisper models."""
    cli: CLIContext = ctx.obj
    emit(cli, "schema.models", {
        "models": _AVAILABLE_MODELS,
        "tier_mapping": {
            "draft": "base",
            "balanced": "turbo",
            "high_quality": "large-v3",
        },
    })


@schema_app.command("tiers")
def schema_tiers(ctx: typer.Context) -> None:
    """List available quality tiers."""
    cli: CLIContext = ctx.obj
    emit(cli, "schema.tiers", {
        "tiers": [
            {"name": "draft", "model": "base", "description": "Fast, lower accuracy"},
            {"name": "balanced", "model": "turbo", "description": "Fast with near-large accuracy (809M params)"},
            {"name": "high_quality", "model": "large-v3", "description": "Highest accuracy, slowest"},
        ],
    })


@schema_app.command("formats")
def schema_formats(ctx: typer.Context) -> None:
    """List available output formats."""
    cli: CLIContext = ctx.obj
    emit(cli, "schema.formats", {
        "output_formats": ["json", "markdown", "txt", "vtt", "srt", "tsv", "all"],
        "cli_formats": ["json", "table", "quiet", "yaml"],
    })


@schema_app.command("config")
def schema_config(ctx: typer.Context) -> None:
    """Show all configuration fields with types and defaults."""
    from audioscript.config.settings import AudioScriptConfig

    cli: CLIContext = ctx.obj
    schema = AudioScriptConfig.model_json_schema()
    emit(cli, "schema.config", schema)


@schema_app.command("env")
def schema_env(ctx: typer.Context) -> None:
    """List supported environment variables."""
    cli: CLIContext = ctx.obj
    emit(cli, "schema.env", {
        "env_vars": [
            {"name": "HF_TOKEN", "description": "HuggingFace access token for diarization/VAD", "used_by": ["transcribe", "diarize", "vad"]},
            {"name": "AUDIOSCRIPT_FORMAT", "description": "Default output format (json, table, quiet, yaml)", "used_by": ["all"]},
            {"name": "AUDIOSCRIPT_OUTPUT_DIR", "description": "Default output directory", "used_by": ["transcribe"]},
            {"name": "AUDIOSCRIPT_TIER", "description": "Default quality tier (draft, balanced, high_quality)", "used_by": ["transcribe"]},
            {"name": "AUDIOSCRIPT_MODEL", "description": "Default Whisper model name", "used_by": ["transcribe", "detect-language"]},
            {"name": "AUDIOSCRIPT_LOG", "description": "Stderr log level (debug, info, warning, error)", "used_by": ["all"]},
            {"name": "AUDIOSCRIPT_LOG_FILE", "description": "Directory for JSON-line log files", "used_by": ["all"]},
            {"name": "ANTHROPIC_API_KEY", "description": "Anthropic API key for LLM analysis (summary, title, action items)", "used_by": ["transcribe", "sync", "analyze"]},
        ],
    })
