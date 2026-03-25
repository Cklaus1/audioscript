"""Transcript-based name extraction for speaker identity resolution.

Regex-based extraction of name hints from transcript text. No LLM needed.
Detects self-introductions, direct address, and announcement patterns.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

# Common false positives to reject
_FALSE_POSITIVES = {
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
    "January", "February", "March", "April", "May", "June", "July",
    "August", "September", "October", "November", "December",
    "Thanks", "Thank", "Hello", "Yeah", "Yes", "Right", "Well",
    "Actually", "Basically", "Sure", "Okay", "Sorry", "Please",
    "Google", "Microsoft", "Amazon", "Apple", "Facebook", "Netflix",
    "The", "This", "That", "What", "Where", "When", "How",
}


@dataclass
class NameHint:
    """A name mention detected in the transcript."""

    name: str
    speaker_label: str  # Who said it (the speaker of this segment)
    timestamp: float
    pattern: str  # Which pattern matched
    confidence: float  # 0.3-0.7

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "speaker_label": self.speaker_label,
            "timestamp": self.timestamp,
            "pattern": self.pattern,
            "confidence": self.confidence,
        }


# Patterns ordered by confidence (highest first)
# Each: (compiled regex, pattern name, confidence, group index for name)
_PATTERNS: list[tuple[re.Pattern, str, float]] = [
    # Self-introductions (high confidence)
    (re.compile(r"(?:this is|my name is)\s+([A-Z][a-z]{1,14})", re.IGNORECASE), "self_intro", 0.7),
    (re.compile(r"(?:i'm|i am)\s+([A-Z][a-z]{1,14})\s+from", re.IGNORECASE), "self_intro_org", 0.7),

    # Addressing someone (medium confidence)
    (re.compile(r"(?:hey|hi|hello)\s+([A-Z][a-z]{1,14})(?:\s|,|\.|\!)", re.IGNORECASE), "greeting", 0.5),
    (re.compile(r"(?:thanks|thank you)\s+([A-Z][a-z]{1,14})(?:\s|,|\.|\!|$)", re.IGNORECASE), "thanks", 0.5),

    # Announcement (medium confidence)
    (re.compile(r"(?:i'm|i am)\s+([A-Z][a-z]{1,14})(?:\s|,|\.|\!|$)", re.IGNORECASE), "self_announce", 0.6),
    (re.compile(r"([A-Z][a-z]{1,14})\s+here(?:\s|,|\.|\!|$)"), "here_announce", 0.5),

    # Direct address mid-sentence (lower confidence)
    (re.compile(r"(?:right|so|well|okay)\s+([A-Z][a-z]{1,14})(?:\s|,)"), "mid_address", 0.4),
]


def _is_valid_name(name: str) -> bool:
    """Check if a candidate name is plausible (not a false positive)."""
    if len(name) < 2 or len(name) > 15:
        return False
    if name in _FALSE_POSITIVES:
        return False
    if not name[0].isupper():
        return False
    if not name.isalpha():
        return False
    return True


def extract_name_hints(segments: list[dict[str, Any]]) -> list[NameHint]:
    """Extract name mentions from transcript segments.

    Scans each segment's text against regex patterns to find
    self-introductions, greetings, and direct address patterns.

    Args:
        segments: Transcript segments with 'text', 'speaker', 'start' fields.

    Returns:
        List of NameHint objects, sorted by confidence (highest first).
    """
    hints: list[NameHint] = []
    seen: set[tuple[str, str]] = set()  # (name, speaker) dedup

    for seg in segments:
        text = seg.get("text", "")
        speaker = seg.get("speaker", "")
        timestamp = seg.get("start", 0.0)

        if not text:
            continue

        for pattern, pattern_name, confidence in _PATTERNS:
            for match in pattern.finditer(text):
                name = match.group(1).strip()

                if not _is_valid_name(name):
                    continue

                # Dedup: same name from same speaker only once
                key = (name.lower(), speaker)
                if key in seen:
                    continue
                seen.add(key)

                hints.append(NameHint(
                    name=name,
                    speaker_label=speaker,
                    timestamp=timestamp,
                    pattern=pattern_name,
                    confidence=confidence,
                ))

    # Sort by confidence (highest first)
    hints.sort(key=lambda h: h.confidence, reverse=True)
    return hints


def match_hints_to_speakers(
    hints: list[NameHint],
    diarization_segments: list[dict[str, Any]],
) -> dict[str, list[NameHint]]:
    """Group name hints by which speaker they likely refer to.

    Logic:
    - Self-introductions (self_intro, self_announce, self_intro_org):
      The name refers to the speaker who said it.
    - Greetings/thanks (greeting, thanks, mid_address):
      The name refers to a DIFFERENT speaker (the one being addressed).
      We try to find who was speaking just before or after.
    """
    result: dict[str, list[NameHint]] = {}

    for hint in hints:
        if hint.pattern in ("self_intro", "self_announce", "self_intro_org", "here_announce"):
            # Name refers to the speaker themselves
            target = hint.speaker_label
        else:
            # Name refers to someone else — try to find who
            # For now, just mark it as "addressed by {speaker}"
            target = f"addressed_by_{hint.speaker_label}"

        result.setdefault(target, []).append(hint)

    return result
