"""Speaker identity management commands."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from audioscript.cli.output import CLIContext, ExitCode, emit, emit_error

speakers_app = typer.Typer(
    name="speakers",
    help="Manage speaker identities across calls.",
)


def _get_db(db_path: str | None, output_dir: str = "./output"):
    """Load the speaker identity DB from the given or default path."""
    from audioscript.speakers.identity_db import SpeakerIdentityDB

    if db_path:
        return SpeakerIdentityDB(Path(db_path))
    # Try common locations
    for candidate in [
        Path(output_dir) / "speaker_identities.json",
        Path("./transcripts/sync/speaker_identities.json"),
        Path("./speaker_identities.json"),
    ]:
        if candidate.exists():
            return SpeakerIdentityDB(candidate)
    # Create at default location
    return SpeakerIdentityDB(Path(output_dir) / "speaker_identities.json")


@speakers_app.command("list")
def speakers_list(
    ctx: typer.Context,
    db: Optional[str] = typer.Option(None, "--db", help="Path to speaker identity DB"),
    status: Optional[str] = typer.Option(None, "--status", help="Filter by status: unknown, candidate, probable, confirmed"),
) -> None:
    """List all known speaker identities."""
    cli: CLIContext = ctx.obj
    identity_db = _get_db(db)

    identities = identity_db.list_identities(status=status)
    speakers = []
    for i in identities:
        speakers.append({
            "id": i.speaker_cluster_id,
            "name": i.canonical_name,
            "status": i.status,
            "calls": i.total_calls,
            "minutes": round(i.total_speaking_seconds / 60, 1),
            "first_seen": i.first_seen,
            "last_seen": i.last_seen,
            "co_speakers": i.typical_co_speakers[:5],
        })

    speakers.sort(key=lambda s: s["calls"], reverse=True)

    emit(cli, "speakers.list", {
        "total": len(speakers),
        "confirmed": sum(1 for s in speakers if s["status"] == "confirmed"),
        "unknown": sum(1 for s in speakers if s["status"] in ("unknown", "candidate")),
        "speakers": speakers,
    })


@speakers_app.command("summary")
def speakers_summary(
    ctx: typer.Context,
    db: Optional[str] = typer.Option(None, "--db", help="Path to speaker identity DB"),
) -> None:
    """Show unknown speaker review queue."""
    cli: CLIContext = ctx.obj
    identity_db = _get_db(db)

    from audioscript.speakers.reporter import UnknownSpeakerReporter

    reporter = UnknownSpeakerReporter(identity_db)
    summary = reporter.generate_summary()

    emit(cli, "speakers.summary", summary)


@speakers_app.command("label")
def speakers_label(
    ctx: typer.Context,
    cluster_id: str = typer.Argument(help="Speaker cluster ID (e.g. spk_a91f)"),
    name: str = typer.Argument(help="Name to assign (e.g. Chris)"),
    db: Optional[str] = typer.Option(None, "--db", help="Path to speaker identity DB"),
) -> None:
    """Assign a name to a speaker cluster.

    Example: audioscript speakers label spk_a91f Chris
    """
    cli: CLIContext = ctx.obj
    identity_db = _get_db(db)

    identity = identity_db.get_identity(cluster_id)
    if not identity:
        emit_error(
            cli, ExitCode.VALIDATION_ERROR, "validation",
            f"Speaker cluster not found: {cluster_id}",
            hint="Use 'audioscript speakers list' to see all clusters.",
        )
        return

    identity_db.confirm_identity(cluster_id, name, source="user_confirmation")

    emit(cli, "speakers.label", {
        "cluster_id": cluster_id,
        "name": name,
        "status": "confirmed",
        "previous_name": identity.canonical_name,
        "total_calls": identity.total_calls,
    })


@speakers_app.command("merge")
def speakers_merge(
    ctx: typer.Context,
    cluster_a: str = typer.Argument(help="First speaker cluster ID"),
    cluster_b: str = typer.Argument(help="Second speaker cluster ID (will be merged into first)"),
    db: Optional[str] = typer.Option(None, "--db", help="Path to speaker identity DB"),
) -> None:
    """Merge two speaker clusters (when they're the same person).

    The second cluster is merged into the first. The first cluster's name
    and status are preserved.
    """
    cli: CLIContext = ctx.obj
    identity_db = _get_db(db)

    if not identity_db.get_identity(cluster_a):
        emit_error(cli, ExitCode.VALIDATION_ERROR, "validation", f"Cluster not found: {cluster_a}")
        return
    if not identity_db.get_identity(cluster_b):
        emit_error(cli, ExitCode.VALIDATION_ERROR, "validation", f"Cluster not found: {cluster_b}")
        return

    success = identity_db.merge_clusters(cluster_a, cluster_b)
    if not success:
        emit_error(cli, ExitCode.INTERNAL_ERROR, "merge", "Merge operation failed")
        return

    merged = identity_db.get_identity(cluster_a)
    emit(cli, "speakers.merge", {
        "kept": cluster_a,
        "merged": cluster_b,
        "name": merged.canonical_name if merged else None,
        "total_calls": merged.total_calls if merged else 0,
        "total_minutes": round(merged.total_speaking_seconds / 60, 1) if merged else 0,
    })


@speakers_app.command("enroll")
def speakers_enroll(
    ctx: typer.Context,
    name: str = typer.Argument(help="Speaker name to enroll"),
    sample: Path = typer.Argument(help="Path to voice sample audio file"),
    db: Optional[str] = typer.Option(None, "--db", help="Path to speaker identity DB"),
    hf_token: Optional[str] = typer.Option(None, "--hf-token", help="HuggingFace token"),
) -> None:
    """Enroll a speaker from a voice sample.

    Example: audioscript speakers enroll Chris voice_sample.wav
    """
    import os
    cli: CLIContext = ctx.obj
    identity_db = _get_db(db)

    if not sample.exists():
        emit_error(cli, ExitCode.VALIDATION_ERROR, "validation", f"Sample file not found: {sample}")
        return

    token = hf_token or os.environ.get("HF_TOKEN")
    if not token:
        emit_error(
            cli, ExitCode.AUTH_ERROR, "auth",
            "HuggingFace token required for enrollment (diarization).",
            hint="Set HF_TOKEN env var or pass --hf-token.",
        )
        return

    try:
        from audioscript.speakers.enrollment import SpeakerEnrollment
        enrollment = SpeakerEnrollment(identity_db)
        cluster_id = enrollment.enroll_from_audio(name, sample, hf_token=token)

        emit(cli, "speakers.enroll", {
            "name": name,
            "cluster_id": cluster_id,
            "sample": str(sample),
            "status": "confirmed",
        })
    except Exception as e:
        emit_error(cli, ExitCode.TRANSCRIPTION_ERROR, "enrollment", str(e))


@speakers_app.command("split")
def speakers_split(
    ctx: typer.Context,
    cluster_id: str = typer.Argument(help="Speaker cluster ID to split"),
    db: Optional[str] = typer.Option(None, "--db", help="Path to speaker identity DB"),
) -> None:
    """Split a polluted cluster by detecting embedding outliers.

    When diarization incorrectly merges two different voices into one cluster,
    use this to separate them based on embedding distance from the centroid.
    """
    cli: CLIContext = ctx.obj
    identity_db = _get_db(db)

    identity = identity_db.get_identity(cluster_id)
    if not identity:
        emit_error(cli, ExitCode.VALIDATION_ERROR, "validation", f"Cluster not found: {cluster_id}")
        return

    if identity.total_calls < 2:
        emit_error(
            cli, ExitCode.VALIDATION_ERROR, "validation",
            f"Cluster {cluster_id} has only {identity.total_calls} call(s) — need at least 2 to split.",
        )
        return

    # For now, report the cluster info and suggest manual review
    # Full automatic split (embedding outlier detection) requires stored per-occurrence embeddings
    # which we don't persist yet — that's a future enhancement
    emit(cli, "speakers.split", {
        "cluster_id": cluster_id,
        "name": identity.canonical_name,
        "total_calls": identity.total_calls,
        "status": "manual_review_needed",
        "hint": "Automatic split requires per-occurrence embedding storage (future feature). "
                "Use 'speakers merge' to manually reassign after creating new clusters.",
    })
