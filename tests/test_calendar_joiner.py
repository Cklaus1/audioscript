"""Tests for audioscript.speakers.calendar — CalendarEvent and CalendarJoiner."""

from __future__ import annotations

import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from audioscript.speakers.calendar import CalendarEvent, CalendarJoiner

# ---------------------------------------------------------------------------
# Sample Microsoft Graph event JSON
# ---------------------------------------------------------------------------
SAMPLE_GRAPH_EVENT = {
    "id": "AAMkAGI2TG93AAA=",
    "subject": "Weekly Standup",
    "start": {"dateTime": "2026-03-25T14:00:00.0000000+00:00", "timeZone": "UTC"},
    "end": {"dateTime": "2026-03-25T14:30:00.0000000+00:00", "timeZone": "UTC"},
    "attendees": [
        {
            "emailAddress": {"name": "Chris Adams", "address": "chris@example.com"},
            "type": "required",
        },
        {
            "emailAddress": {"name": "Dana Lee", "address": "dana@example.com"},
            "type": "optional",
        },
    ],
    "organizer": {
        "emailAddress": {"name": "Pat Morgan", "address": "pat@example.com"},
    },
    "isOnlineMeeting": True,
    "seriesMasterId": "AAMkAGI2series",
    "webLink": "https://outlook.office365.com/owa/?itemid=AAMkAGI2TG93AAA%3D",
}


# ---------------------------------------------------------------------------
# CalendarEvent property tests
# ---------------------------------------------------------------------------

class TestCalendarEvent:
    def test_attendee_names(self):
        event = CalendarEvent(
            event_id="e1",
            title="Test",
            start_time="2026-03-25T14:00:00",
            end_time="2026-03-25T14:30:00",
            attendees=[
                {"name": "Chris Adams", "email": "chris@example.com"},
                {"name": "Dana Lee", "email": "dana@example.com"},
                {"name": "", "email": "no-name@example.com"},
            ],
        )
        assert event.attendee_names == ["Chris Adams", "Dana Lee"]

    def test_attendee_emails(self):
        event = CalendarEvent(
            event_id="e1",
            title="Test",
            start_time="2026-03-25T14:00:00",
            end_time="2026-03-25T14:30:00",
            attendees=[
                {"name": "Chris Adams", "email": "chris@example.com"},
                {"name": "Dana Lee", "email": ""},
            ],
        )
        assert event.attendee_emails == ["chris@example.com"]


# ---------------------------------------------------------------------------
# CalendarJoiner tests
# ---------------------------------------------------------------------------

class TestCalendarJoiner:
    def _make_joiner(self) -> CalendarJoiner:
        return CalendarJoiner(ms365_path="/usr/bin/ms365")

    # -- is_available --

    @patch("audioscript.speakers.calendar.subprocess.run")
    def test_is_available_true(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0, stdout='{"status":"logged_in"}', stderr=""
        )
        joiner = self._make_joiner()
        assert joiner.is_available() is True
        mock_run.assert_called_once()

    def test_is_available_false_no_path(self):
        joiner = CalendarJoiner(ms365_path=None)
        # _ms365_path is None because we passed None and _find_ms365 likely returns None
        joiner._ms365_path = None
        assert joiner.is_available() is False

    @patch("audioscript.speakers.calendar.subprocess.run", side_effect=FileNotFoundError)
    def test_is_available_false_file_not_found(self, mock_run):
        joiner = self._make_joiner()
        assert joiner.is_available() is False

    # -- _parse_event --

    def test_parse_event(self):
        event = CalendarJoiner._parse_event(SAMPLE_GRAPH_EVENT)
        assert event.event_id == "AAMkAGI2TG93AAA="
        assert event.title == "Weekly Standup"
        assert event.start_time == "2026-03-25T14:00:00.0000000+00:00"
        assert event.end_time == "2026-03-25T14:30:00.0000000+00:00"
        assert event.is_online_meeting is True
        assert event.recurring_series_id == "AAMkAGI2series"
        assert len(event.attendees) == 2
        assert event.attendees[0]["name"] == "Chris Adams"
        assert event.organizer["name"] == "Pat Morgan"

    # -- fetch_events --

    @patch("audioscript.speakers.calendar.subprocess.run")
    def test_fetch_events_calls_m365(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({"value": [SAMPLE_GRAPH_EVENT]}),
            stderr="",
        )
        joiner = self._make_joiner()
        events = joiner.fetch_events(
            "2026-03-25T13:00:00Z", "2026-03-25T15:00:00Z"
        )
        assert len(events) == 1
        assert events[0].title == "Weekly Standup"
        # Verify correct CLI args
        call_args = mock_run.call_args[0][0]
        assert "calendar" in call_args
        assert "view" in call_args
        assert "--startDateTime" in call_args

    # -- match_call --

    @patch("audioscript.speakers.calendar.subprocess.run")
    def test_match_call_best_overlap(self, mock_run):
        """match_call returns event with best overlap."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({"value": [SAMPLE_GRAPH_EVENT]}),
            stderr="",
        )
        joiner = self._make_joiner()
        # Call starts at 14:00 and lasts 20 min (1200s) — fully inside the 30-min event
        result = joiner.match_call("2026-03-25T14:00:00Z", 1200)
        assert result is not None
        assert result.title == "Weekly Standup"

    @patch("audioscript.speakers.calendar.subprocess.run")
    def test_match_call_no_overlap(self, mock_run):
        """match_call returns None when overlap < 60s."""
        # Event is 14:00–14:30, call starts at 10:00 and lasts 30s
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({"value": [SAMPLE_GRAPH_EVENT]}),
            stderr="",
        )
        joiner = self._make_joiner()
        result = joiner.match_call("2026-03-25T10:00:00Z", 30)
        assert result is None

    # -- generate_candidates --

    def test_generate_candidates_excludes_confirmed(self):
        event = CalendarJoiner._parse_event(SAMPLE_GRAPH_EVENT)
        joiner = self._make_joiner()
        candidates = joiner.generate_candidates(
            event,
            resolved_cluster_ids=set(),
            confirmed_names={"Chris Adams"},
        )
        names = [c["name"] for c in candidates]
        assert "Chris Adams" not in names
        assert "Dana Lee" in names

    def test_generate_candidates_returns_score_half(self):
        event = CalendarJoiner._parse_event(SAMPLE_GRAPH_EVENT)
        joiner = self._make_joiner()
        candidates = joiner.generate_candidates(
            event,
            resolved_cluster_ids=set(),
            confirmed_names=set(),
        )
        assert len(candidates) == 2
        for c in candidates:
            assert c["score"] == 0.5
            assert c["source"] == "calendar_attendee"
            assert c["event_title"] == "Weekly Standup"
