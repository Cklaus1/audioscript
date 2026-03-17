"""Structured file logging for AudioScript.

Logs go to JSON-line files (one JSON object per line) for agent debugging.
Separate from Rich stderr output — controlled via environment variables:

  AUDIOSCRIPT_LOG       - Log level for stderr (debug, info, warning, error). Default: warning.
  AUDIOSCRIPT_LOG_FILE  - Directory for JSON-line log files. Disabled if unset.
"""

import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional


class JsonLineHandler(logging.Handler):
    """Logging handler that writes JSON-line format to a file."""

    def __init__(self, log_dir: str):
        super().__init__()
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        date_str = datetime.now().strftime("%Y-%m-%d")
        self.log_path = self.log_dir / f"audioscript-{date_str}.jsonl"
        self._file = open(self.log_path, "a")

    def emit(self, record: logging.LogRecord) -> None:
        try:
            entry = {
                "ts": datetime.fromtimestamp(record.created).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            }
            if record.exc_info and record.exc_info[1]:
                entry["exception"] = str(record.exc_info[1])
            self._file.write(json.dumps(entry) + "\n")
            self._file.flush()
        except Exception:
            pass

    def close(self) -> None:
        try:
            self._file.close()
        except Exception:
            pass
        super().close()


def setup_logging() -> None:
    """Configure logging based on AUDIOSCRIPT_LOG and AUDIOSCRIPT_LOG_FILE env vars."""
    root_logger = logging.getLogger("audioscript")

    # Stderr log level
    level_name = os.environ.get("AUDIOSCRIPT_LOG", "warning").upper()
    level = getattr(logging, level_name, logging.WARNING)
    root_logger.setLevel(level)

    # Stderr handler (minimal, non-Rich)
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(level)
    stderr_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    root_logger.addHandler(stderr_handler)

    # JSON-line file handler
    log_dir = os.environ.get("AUDIOSCRIPT_LOG_FILE")
    if log_dir:
        json_handler = JsonLineHandler(log_dir)
        json_handler.setLevel(logging.DEBUG)  # Capture everything to file
        root_logger.addHandler(json_handler)
