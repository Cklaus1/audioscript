"""Calendar integration via ms365-cli for speaker identity resolution.

Shells out to ms365-cli (`node /path/to/ms365-cli/dist/index.js calendar view`)
to fetch calendar events and match them to calls by timestamp. Generates
identity candidates from event attendees.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)

# Default ms365-cli path (bflow's location)
_DEFAULT_MS365_PATH = "/root/projects/ms365-cli/dist/index.js"


@dataclass
class CalendarEvent:
    """A calendar event matched to a call."""

    event_id: str
    title: str
    start_time: str
    end_time: str
    timezone: str = "UTC"
    attendees: list[dict[str, str]] = field(default_factory=list)  # [{name, email, type}]
    organizer: dict[str, str] = field(default_factory=dict)  # {name, email}
    is_online_meeting: bool = False
    recurring_series_id: str | None = None
    web_link: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "title": self.title,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "timezone": self.timezone,
            "attendees": self.attendees,
            "organizer": self.organizer,
            "is_online_meeting": self.is_online_meeting,
            "recurring_series_id": self.recurring_series_id,
        }

    @property
    def attendee_names(self) -> list[str]:
        return [a["name"] for a in self.attendees if a.get("name")]

    @property
    def attendee_emails(self) -> list[str]:
        return [a["email"] for a in self.attendees if a.get("email")]


class CalendarJoiner:
    """Matches calls to calendar events via ms365-cli."""

    def __init__(self, ms365_path: str | None = None) -> None:
        self._ms365_path = ms365_path or self._find_ms365()

    @staticmethod
    def _find_ms365() -> str | None:
        """Find ms365-cli binary."""
        # Check PATH first
        found = shutil.which("ms365")
        if found:
            return found
        # Check bflow's dev path
        import os
        if os.path.exists(_DEFAULT_MS365_PATH):
            return f"node {_DEFAULT_MS365_PATH}"
        return None

    def is_available(self) -> bool:
        """Check if ms365-cli is installed and authenticated."""
        if not self._ms365_path:
            return False
        try:
            result = self._run_ms365(["auth", "status", "--output", "json"])
            return result is not None
        except Exception:
            return False

    def fetch_events(
        self,
        start: str,
        end: str,
    ) -> list[CalendarEvent]:
        """Fetch calendar events for a time range.

        Args:
            start: ISO datetime string (e.g. "2026-03-25T13:00:00Z")
            end: ISO datetime string
        """
        result = self._run_ms365([
            "calendar", "view",
            "--startDateTime", start,
            "--endDateTime", end,
            "--output", "json",
        ])

        if not result:
            return []

        events = []
        items = result if isinstance(result, list) else result.get("value", [])

        for item in items:
            events.append(self._parse_event(item))

        return events

    def match_call(
        self,
        call_start: str,
        duration_seconds: float,
        window_minutes: int = 30,
    ) -> CalendarEvent | None:
        """Match a call to a calendar event by timestamp overlap.

        Fetches events in a ±window around the call start time.
        Returns the best match (highest time overlap).
        """
        try:
            start_dt = datetime.fromisoformat(call_start.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            logger.warning("Cannot parse call start time: %s", call_start)
            return None

        # Fetch events in a window around the call
        window = timedelta(minutes=window_minutes)
        fetch_start = (start_dt - window).isoformat()
        fetch_end = (start_dt + timedelta(seconds=duration_seconds) + window).isoformat()

        events = self.fetch_events(fetch_start, fetch_end)
        if not events:
            return None

        # Find best overlap
        call_end_dt = start_dt + timedelta(seconds=duration_seconds)
        best_event = None
        best_overlap = 0.0

        for event in events:
            try:
                evt_start = datetime.fromisoformat(event.start_time.replace("Z", "+00:00"))
                evt_end = datetime.fromisoformat(event.end_time.replace("Z", "+00:00"))
            except (ValueError, TypeError):
                continue

            # Compute overlap
            overlap_start = max(start_dt, evt_start)
            overlap_end = min(call_end_dt, evt_end)
            overlap_seconds = max(0, (overlap_end - overlap_start).total_seconds())

            if overlap_seconds > best_overlap:
                best_overlap = overlap_seconds
                best_event = event

        # Require at least 60 seconds of overlap
        if best_event and best_overlap >= 60:
            logger.info(
                "Matched call to calendar event: '%s' (%.0fs overlap)",
                best_event.title, best_overlap,
            )
            return best_event

        return None

    def generate_candidates(
        self,
        event: CalendarEvent,
        resolved_cluster_ids: set[str],
        confirmed_names: set[str],
    ) -> list[dict[str, Any]]:
        """Generate identity candidates from event attendees.

        Returns candidates for attendees whose names don't match
        any already-confirmed speaker.
        """
        candidates = []
        confirmed_lower = {n.lower() for n in confirmed_names}

        for attendee in event.attendees:
            name = attendee.get("name", "")
            email = attendee.get("email", "")

            if not name:
                continue

            # Skip if already confirmed in this call
            if name.lower() in confirmed_lower:
                continue

            candidates.append({
                "name": name,
                "email": email,
                "score": 0.5,
                "source": "calendar_attendee",
                "event_title": event.title,
                "event_id": event.event_id,
            })

        return candidates

    def _run_ms365(self, args: list[str]) -> Any:
        """Execute an ms365-cli command and return parsed JSON."""
        if not self._ms365_path:
            return None

        cmd_parts = self._ms365_path.split()  # handles "node /path/index.js"
        cmd = cmd_parts + args

        try:
            result = subprocess.run(
                cmd,
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode != 0:
                logger.debug("ms365-cli error: %s", result.stderr[:200])
                return None

            return json.loads(result.stdout)
        except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError) as e:
            logger.debug("ms365-cli failed: %s", e)
            return None

    @staticmethod
    def _parse_event(item: dict[str, Any]) -> CalendarEvent:
        """Parse a Microsoft Graph event into CalendarEvent."""
        attendees = []
        for att in item.get("attendees", []):
            email_addr = att.get("emailAddress", {})
            attendees.append({
                "name": email_addr.get("name", ""),
                "email": email_addr.get("address", ""),
                "type": att.get("type", "required"),
            })

        organizer_data = item.get("organizer", {}).get("emailAddress", {})

        return CalendarEvent(
            event_id=item.get("id", ""),
            title=item.get("subject", ""),
            start_time=item.get("start", {}).get("dateTime", ""),
            end_time=item.get("end", {}).get("dateTime", ""),
            timezone=item.get("start", {}).get("timeZone", "UTC"),
            attendees=attendees,
            organizer={
                "name": organizer_data.get("name", ""),
                "email": organizer_data.get("address", ""),
            },
            is_online_meeting=item.get("isOnlineMeeting", False),
            recurring_series_id=item.get("seriesMasterId"),
            web_link=item.get("webLink"),
        )
