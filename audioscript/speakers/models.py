"""Core data models for the speaker identity system."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def generate_speaker_id() -> str:
    """Generate a stable speaker cluster ID: spk_ + 8 hex chars."""
    return f"spk_{uuid.uuid4().hex[:8]}"


def generate_id(prefix: str = "") -> str:
    """Generate a unique ID with optional prefix."""
    return f"{prefix}{uuid.uuid4().hex[:12]}"


def now_iso() -> str:
    """Current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()


@dataclass
class SpeakerIdentity:
    """Stable identity across calls — may be named or unnamed.

    A speaker can be confirmed with no name but stable internal identity.
    spk_a91f appearing in 12 calls is valuable even unnamed.
    """

    speaker_cluster_id: str
    canonical_name: str | None = None
    aliases: list[str] = field(default_factory=list)
    status: str = "unknown"  # unknown | candidate | probable | confirmed
    embedding_centroid: list[float] = field(default_factory=list)
    sample_count: int = 0
    first_seen: str = ""
    last_seen: str = ""
    total_calls: int = 0
    total_speaking_seconds: float = 0.0
    typical_co_speakers: list[str] = field(default_factory=list)
    created_from: str = "auto_cluster"  # auto_cluster | enrollment | user_confirmed
    updated_at: str = ""

    @property
    def display_name(self) -> str:
        """Return canonical name, first alias, or cluster ID."""
        if self.canonical_name:
            return self.canonical_name
        if self.aliases:
            return self.aliases[0]
        return self.speaker_cluster_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "speaker_cluster_id": self.speaker_cluster_id,
            "canonical_name": self.canonical_name,
            "aliases": self.aliases,
            "status": self.status,
            "embedding_centroid": self.embedding_centroid,
            "sample_count": self.sample_count,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "total_calls": self.total_calls,
            "total_speaking_seconds": self.total_speaking_seconds,
            "typical_co_speakers": self.typical_co_speakers,
            "created_from": self.created_from,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SpeakerIdentity:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class SpeakerOccurrence:
    """A speaker's appearance in one call."""

    occurrence_id: str
    call_id: str
    speaker_cluster_id: str
    local_label: str  # SPEAKER_00
    display_name: str | None = None
    identity_status: str = "unknown"
    resolution_source: str | None = None
    resolution_confidence: float = 0.0
    total_speaking_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "occurrence_id": self.occurrence_id,
            "call_id": self.call_id,
            "speaker_cluster_id": self.speaker_cluster_id,
            "local_label": self.local_label,
            "display_name": self.display_name,
            "identity_status": self.identity_status,
            "resolution_source": self.resolution_source,
            "resolution_confidence": self.resolution_confidence,
            "total_speaking_seconds": self.total_speaking_seconds,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SpeakerOccurrence:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class SpeakerEvidence:
    """Evidence for an identity decision — makes every linking explainable."""

    evidence_id: str
    speaker_cluster_id: str
    call_id: str | None = None
    type: str = "auto_cluster"  # db_match | auto_cluster | user_confirmation | manual_merge
    score: float = 0.0
    summary: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "speaker_cluster_id": self.speaker_cluster_id,
            "call_id": self.call_id,
            "type": self.type,
            "score": self.score,
            "summary": self.summary,
            "details": self.details,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SpeakerEvidence:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class SpeakerReviewItem:
    """An item in the unknown speaker review queue."""

    review_item_id: str
    speaker_cluster_id: str
    priority_score: float = 0.0
    reason: str = ""
    summary: str = ""
    candidate_names: list[str] = field(default_factory=list)
    status: str = "open"  # open | resolved | dismissed
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "review_item_id": self.review_item_id,
            "speaker_cluster_id": self.speaker_cluster_id,
            "priority_score": self.priority_score,
            "reason": self.reason,
            "summary": self.summary,
            "candidate_names": self.candidate_names,
            "status": self.status,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SpeakerReviewItem:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
