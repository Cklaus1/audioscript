"""Token and cost tracking for LLM usage."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Pricing per million tokens (as of 2026-03)
MODEL_PRICING: dict[str, dict[str, float]] = {
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.00},
    "claude-opus-4-6": {"input": 15.00, "output": 75.00},
}

# Default fallback
_DEFAULT_PRICING = {"input": 3.00, "output": 15.00}


@dataclass
class UsageRecord:
    """A single LLM usage event."""

    timestamp: float
    model: str
    call_id: str  # file hash or identifier
    task: str  # "analyze_transcript"
    input_tokens: int
    output_tokens: int
    cost_usd: float
    duration_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "model": self.model,
            "call_id": self.call_id,
            "task": self.task,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "duration_seconds": round(self.duration_seconds, 2),
        }


class CostTracker:
    """Tracks LLM token usage and costs across sync sessions."""

    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self._session_records: list[UsageRecord] = []

    def record(
        self,
        model: str,
        call_id: str,
        task: str,
        input_tokens: int,
        output_tokens: int,
        duration_seconds: float = 0.0,
    ) -> UsageRecord:
        """Record a usage event and compute cost."""
        pricing = MODEL_PRICING.get(model, _DEFAULT_PRICING)
        cost = (
            (input_tokens / 1_000_000) * pricing["input"]
            + (output_tokens / 1_000_000) * pricing["output"]
        )

        record = UsageRecord(
            timestamp=time.time(),
            model=model,
            call_id=call_id,
            task=task,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            duration_seconds=duration_seconds,
        )

        self._session_records.append(record)
        self._append_to_log(record)

        logger.info(
            "LLM usage: %s | %d in + %d out = $%.4f",
            task, input_tokens, output_tokens, cost,
        )

        return record

    def session_summary(self) -> dict[str, Any]:
        """Summary of LLM usage for the current session."""
        total_input = sum(r.input_tokens for r in self._session_records)
        total_output = sum(r.output_tokens for r in self._session_records)
        total_cost = sum(r.cost_usd for r in self._session_records)

        return {
            "calls": len(self._session_records),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_cost_usd": round(total_cost, 4),
        }

    def cumulative_summary(self) -> dict[str, Any]:
        """Summary of all-time LLM usage from the log file."""
        records = self._load_log()
        total_input = sum(r.get("input_tokens", 0) for r in records)
        total_output = sum(r.get("output_tokens", 0) for r in records)
        total_cost = sum(r.get("cost_usd", 0) for r in records)

        return {
            "total_calls": len(records),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_cost_usd": round(total_cost, 4),
        }

    def _append_to_log(self, record: UsageRecord) -> None:
        """Append a record to the JSONL log file."""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record.to_dict()) + "\n")

    def _load_log(self) -> list[dict[str, Any]]:
        """Load all records from the JSONL log."""
        if not self.log_path.exists():
            return []
        records = []
        with open(self.log_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return records
