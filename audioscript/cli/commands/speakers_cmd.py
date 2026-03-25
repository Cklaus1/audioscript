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

    id_a = identity_db.get_identity(cluster_a)
    id_b = identity_db.get_identity(cluster_b)

    if not id_a:
        emit_error(cli, ExitCode.VALIDATION_ERROR, "validation", f"Cluster not found: {cluster_a}")
        return
    if not id_b:
        emit_error(cli, ExitCode.VALIDATION_ERROR, "validation", f"Cluster not found: {cluster_b}")
        return

    # Merge B into A: update A's data, repoint B's occurrences
    data_a = identity_db.data["identities"][cluster_a]
    data_b = identity_db.data["identities"][cluster_b]

    # Merge centroid (weighted average)
    n_a = data_a["sample_count"]
    n_b = data_b["sample_count"]
    if data_a["embedding_centroid"] and data_b["embedding_centroid"]:
        merged_centroid = [
            (a * n_a + b * n_b) / (n_a + n_b)
            for a, b in zip(data_a["embedding_centroid"], data_b["embedding_centroid"])
        ]
        data_a["embedding_centroid"] = merged_centroid

    data_a["sample_count"] = n_a + n_b
    data_a["total_calls"] = data_a.get("total_calls", 0) + data_b.get("total_calls", 0)
    data_a["total_speaking_seconds"] = (
        data_a.get("total_speaking_seconds", 0) + data_b.get("total_speaking_seconds", 0)
    )

    # Keep earliest first_seen, latest last_seen
    if data_b.get("first_seen", "") < data_a.get("first_seen", "z"):
        data_a["first_seen"] = data_b["first_seen"]
    if data_b.get("last_seen", "") > data_a.get("last_seen", ""):
        data_a["last_seen"] = data_b["last_seen"]

    # Merge co-speakers
    co = set(data_a.get("typical_co_speakers", []) + data_b.get("typical_co_speakers", []))
    co.discard(cluster_a)
    co.discard(cluster_b)
    data_a["typical_co_speakers"] = list(co)

    # Merge aliases
    aliases = set(data_a.get("aliases", []))
    if data_b.get("canonical_name"):
        aliases.add(data_b["canonical_name"])
    aliases.update(data_b.get("aliases", []))
    data_a["aliases"] = list(aliases)

    # Repoint occurrences
    for occ in identity_db.data["occurrences"]:
        if occ.get("speaker_cluster_id") == cluster_b:
            occ["speaker_cluster_id"] = cluster_a

    # Repoint evidence
    for ev in identity_db.data["evidence"]:
        if ev.get("speaker_cluster_id") == cluster_b:
            ev["speaker_cluster_id"] = cluster_a

    # Remove B
    del identity_db.data["identities"][cluster_b]

    # Add merge evidence
    from audioscript.speakers.models import SpeakerEvidence, generate_id, now_iso
    identity_db.add_evidence(SpeakerEvidence(
        evidence_id=generate_id("ev_"),
        speaker_cluster_id=cluster_a,
        type="manual_merge",
        score=1.0,
        summary=f"Merged {cluster_b} into {cluster_a}",
        created_at=now_iso(),
    ))

    identity_db.save()

    emit(cli, "speakers.merge", {
        "kept": cluster_a,
        "merged": cluster_b,
        "name": data_a.get("canonical_name"),
        "total_calls": data_a["total_calls"],
        "total_minutes": round(data_a["total_speaking_seconds"] / 60, 1),
    })
