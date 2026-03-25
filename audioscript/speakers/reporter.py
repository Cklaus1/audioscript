"""Unknown speaker reporter — generates review queue and summary."""

from __future__ import annotations

from audioscript.speakers.identity_db import SpeakerIdentityDB
from audioscript.speakers.models import (
    SpeakerIdentity,
    SpeakerReviewItem,
    generate_id,
    now_iso,
)


class UnknownSpeakerReporter:
    """Generates prioritized review queue and summary for unknown speakers."""

    def __init__(self, identity_db: SpeakerIdentityDB) -> None:
        self.db = identity_db

    def generate_review_queue(self, min_calls: int = 1) -> list[SpeakerReviewItem]:
        """Generate prioritized list of unknown speakers needing attention.

        Sorted by priority score (highest first).
        """
        unknowns = self.db.get_unknown_speakers(min_calls=min_calls)
        items: list[SpeakerReviewItem] = []

        for identity in unknowns:
            priority = self._compute_priority(identity)
            reason = self._determine_reason(identity)
            summary = self._build_summary(identity)

            items.append(SpeakerReviewItem(
                review_item_id=generate_id("rev_"),
                speaker_cluster_id=identity.speaker_cluster_id,
                priority_score=round(priority, 3),
                reason=reason,
                summary=summary,
                candidate_names=[],  # Phase 2: calendar/transcript candidates
                status="open",
                created_at=now_iso(),
            ))

        items.sort(key=lambda x: x.priority_score, reverse=True)
        return items

    def generate_summary(self) -> dict:
        """Generate unknown speaker summary for CLI/JSON output."""
        all_identities = self.db.list_identities()
        unknowns = [i for i in all_identities if i.status in ("unknown", "candidate")]
        confirmed = [i for i in all_identities if i.status == "confirmed"]
        probable = [i for i in all_identities if i.status == "probable"]

        review_items = self.generate_review_queue()

        return {
            "total_clusters": len(all_identities),
            "confirmed": len(confirmed),
            "probable": len(probable),
            "unknown": len(unknowns),
            "confirmed_speakers": [
                {
                    "id": i.speaker_cluster_id,
                    "name": i.canonical_name,
                    "calls": i.total_calls,
                    "minutes": round(i.total_speaking_seconds / 60, 1),
                }
                for i in sorted(confirmed, key=lambda x: x.total_calls, reverse=True)
            ],
            "review_queue": [
                {
                    "id": item.speaker_cluster_id,
                    "calls": next(
                        (i.total_calls for i in unknowns if i.speaker_cluster_id == item.speaker_cluster_id),
                        0,
                    ),
                    "minutes": round(next(
                        (i.total_speaking_seconds for i in unknowns if i.speaker_cluster_id == item.speaker_cluster_id),
                        0,
                    ) / 60, 1),
                    "last_seen": next(
                        (i.last_seen for i in unknowns if i.speaker_cluster_id == item.speaker_cluster_id),
                        "",
                    ),
                    "priority": item.priority_score,
                    "reason": item.reason,
                }
                for item in review_items[:20]  # Top 20
            ],
        }

    def _compute_priority(self, identity: SpeakerIdentity) -> float:
        """Compute priority score for an unknown speaker.

        Formula: 0.35*minutes + 0.25*calls + 0.20*importance + 0.20*resolvability
        All values normalized to 0-1 range.
        """
        # Normalize minutes (cap at 120 min = 1.0)
        minutes = identity.total_speaking_seconds / 60
        norm_minutes = min(1.0, minutes / 120)

        # Normalize calls (cap at 20 calls = 1.0)
        norm_calls = min(1.0, identity.total_calls / 20)

        # Importance: placeholder (0.5 default, Phase 2 uses call importance)
        importance = 0.5

        # Resolvability: higher if they appear frequently (more data to match)
        resolvability = min(1.0, identity.sample_count / 10)

        return (
            0.35 * norm_minutes
            + 0.25 * norm_calls
            + 0.20 * importance
            + 0.20 * resolvability
        )

    def _determine_reason(self, identity: SpeakerIdentity) -> str:
        """Determine the main reason this speaker needs review."""
        if identity.total_calls >= 5:
            return "frequent_unknown_speaker"
        if identity.total_speaking_seconds >= 600:  # 10+ minutes
            return "high_speaking_time"
        return "new_unknown_speaker"

    def _build_summary(self, identity: SpeakerIdentity) -> str:
        """Build a human-readable summary of an unknown speaker."""
        minutes = round(identity.total_speaking_seconds / 60, 1)
        parts = [
            f"Unknown speaker in {identity.total_calls} call(s)",
            f"{minutes} min total",
        ]
        if identity.typical_co_speakers:
            parts.append(f"co-speakers: {', '.join(identity.typical_co_speakers[:3])}")
        return ". ".join(parts)
