"""Audio file metadata extraction.

Extracts embedded metadata from audio files for cataloging and filtering:
  - File info: size, format, content hash
  - Audio properties: duration, sample rate, channels, bitrate, codec
  - Tags: title, artist, album, date, genre, comment, encoder/software
  - Recording: device, location (GPS), description
  - Custom/extended tags

Uses mutagen for tag reading and ffprobe for codec/stream details.
Falls back gracefully if either is unavailable.
"""

import json
import logging
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def extract_metadata(file_path: Path, content_hash: str | None = None) -> Dict[str, Any]:
    """Extract all available metadata from an audio file.

    Args:
        file_path: Path to the audio file.
        content_hash: Pre-computed SHA-256 hash (avoids re-hashing).

    Returns a dict with sections: file, audio, tags, recording.
    Missing/unavailable sections are omitted rather than null.
    """
    meta: Dict[str, Any] = {}

    # --- File info ---
    meta["file"] = _extract_file_info(file_path, content_hash=content_hash)

    # --- Audio properties via ffprobe ---
    audio_info = _extract_ffprobe(file_path)
    if audio_info:
        meta["audio"] = audio_info

    # --- Tags via mutagen ---
    tags = _extract_tags(file_path)
    if tags:
        meta["tags"] = tags

    # --- Recording context (device, location, etc.) ---
    recording = _extract_recording_context(file_path)
    if recording:
        meta["recording"] = recording

    return meta


def _extract_file_info(file_path: Path, content_hash: str | None = None) -> Dict[str, Any]:
    """Basic file system info."""
    stat = file_path.stat()
    info: Dict[str, Any] = {
        "name": file_path.name,
        "path": str(file_path.absolute()),
        "size_bytes": stat.st_size,
        "size_human": _human_size(stat.st_size),
        "extension": file_path.suffix.lstrip(".").lower(),
        "mime_type": _guess_mime(file_path),
        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
    }

    # Content hash — reuse pre-computed if available
    if content_hash:
        info["content_hash"] = content_hash
    else:
        try:
            from audioscript.utils.file_utils import get_file_hash
            info["content_hash"] = get_file_hash(file_path)
        except Exception:
            pass

    return info


def _guess_mime(file_path: Path) -> Optional[str]:
    """Guess MIME type from extension."""
    import mimetypes
    mime, _ = mimetypes.guess_type(str(file_path))
    return mime


def _extract_ffprobe(file_path: Path) -> Optional[Dict[str, Any]]:
    """Extract audio stream info via ffprobe."""
    try:
        cmd = [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_format", "-show_streams",
            str(file_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if proc.returncode != 0:
            logger.debug("ffprobe failed for %s: %s", file_path, proc.stderr)
            return None

        data = json.loads(proc.stdout)

        result: Dict[str, Any] = {}

        # Format-level info
        fmt = data.get("format", {})
        if fmt.get("duration"):
            result["duration_seconds"] = round(float(fmt["duration"]), 2)
        if fmt.get("bit_rate"):
            result["bitrate_kbps"] = round(int(fmt["bit_rate"]) / 1000)
        if fmt.get("format_name"):
            result["format"] = fmt["format_name"]
        if fmt.get("format_long_name"):
            result["format_name"] = fmt["format_long_name"]

        # Format tags (often contain recording date, etc.)
        fmt_tags = fmt.get("tags", {})
        if fmt_tags:
            result["format_tags"] = _normalize_tag_dict(fmt_tags)

        # Audio stream info
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "audio":
                result["codec"] = stream.get("codec_name")
                result["codec_name"] = stream.get("codec_long_name")
                if stream.get("sample_rate"):
                    result["sample_rate"] = int(stream["sample_rate"])
                if stream.get("channels"):
                    result["channels"] = stream["channels"]
                if stream.get("channel_layout"):
                    result["channel_layout"] = stream["channel_layout"]
                if stream.get("bits_per_sample"):
                    result["bits_per_sample"] = stream["bits_per_sample"]

                # Stream-level tags
                stream_tags = stream.get("tags", {})
                if stream_tags and not fmt_tags:
                    result["format_tags"] = _normalize_tag_dict(stream_tags)
                break

        return result if result else None

    except FileNotFoundError:
        logger.debug("ffprobe not found, skipping audio property extraction")
        return None
    except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception) as e:
        logger.debug("ffprobe extraction failed: %s", e)
        return None


def _extract_tags(file_path: Path) -> Optional[Dict[str, Any]]:
    """Extract embedded tags via mutagen."""
    try:
        import mutagen
        from mutagen.easyid3 import EasyID3

        audio = mutagen.File(file_path, easy=True)
        if audio is None:
            return None

        tags: Dict[str, Any] = {}

        # Standard tag mapping — try common names across formats
        _TAG_MAP = {
            "title": ["title"],
            "artist": ["artist", "albumartist"],
            "author": ["author", "owner"],
            "album": ["album"],
            "date": ["date", "year"],
            "genre": ["genre"],
            "comment": ["comment", "description"],
            "encoder": ["encoder", "encodedby", "encoded_by"],
            "track_number": ["tracknumber"],
            "composer": ["composer"],
            "copyright": ["copyright"],
            "language": ["language"],
        }

        for canonical, keys in _TAG_MAP.items():
            for key in keys:
                try:
                    val = audio.get(key)
                    if val:
                        # mutagen returns lists for most tags
                        text = val[0] if isinstance(val, list) and len(val) == 1 else val
                        if isinstance(text, list):
                            text = "; ".join(str(t) for t in text)
                        tags[canonical] = str(text)
                        break
                except Exception:
                    continue

        # Duration from mutagen (may differ slightly from ffprobe)
        if hasattr(audio, "info") and audio.info:
            if hasattr(audio.info, "length") and audio.info.length:
                tags["duration_seconds"] = round(audio.info.length, 2)

        return tags if tags else None

    except ImportError:
        logger.debug("mutagen not installed, skipping tag extraction")
        return None
    except Exception as e:
        logger.debug("Tag extraction failed for %s: %s", file_path, e)
        return None


def _extract_recording_context(file_path: Path) -> Optional[Dict[str, Any]]:
    """Extract recording-specific metadata (device, GPS, extended tags).

    Many recording apps and devices store info in non-standard tags.
    This function checks mutagen's raw (non-easy) mode for extended data.
    """
    try:
        import mutagen

        audio = mutagen.File(file_path)
        if audio is None or audio.tags is None:
            return None

        recording: Dict[str, Any] = {}

        # --- ID3 extended tags (MP3) ---
        tag_type = type(audio.tags).__name__

        if tag_type in ("ID3", "MP3"):
            _extract_id3_recording(audio.tags, recording)
        elif tag_type in ("MP4Tags",):
            _extract_mp4_recording(audio.tags, recording)
        elif hasattr(audio.tags, "items"):
            # Vorbis comments (FLAC, OGG) — all tags are flat key=value
            _extract_vorbis_recording(audio.tags, recording)

        return recording if recording else None

    except ImportError:
        return None
    except Exception as e:
        logger.debug("Recording context extraction failed: %s", e)
        return None


def _extract_id3_recording(tags: Any, recording: Dict[str, Any]) -> None:
    """Extract recording context from ID3 tags (MP3)."""
    # TXXX frames hold extended user-defined text
    for key in tags:
        if key.startswith("TXXX:"):
            field_name = key[5:].lower()  # e.g., "TXXX:RECORDING_DEVICE" -> "recording_device"
            value = str(tags[key])
            _store_recording_field(field_name, value, recording)

    # Standard ID3 frames for recording context
    _ID3_RECORDING_MAP = {
        "TSSE": "encoder_settings",  # Software/hardware encoder
        "TSRC": "isrc",              # International Standard Recording Code
        "TOFN": "original_filename",
        "TOWN": "owner",
        "TOPE": "original_artist",
        "TDOR": "original_date",
        "TDRC": "recording_date",
        "TDRL": "release_date",
        "TRSN": "radio_station",
        "TRSO": "radio_station_owner",
        "TPE4": "remixer",
    }
    for frame_id, field in _ID3_RECORDING_MAP.items():
        if frame_id in tags:
            val = str(tags[frame_id])
            if val:
                recording[field] = val


def _extract_mp4_recording(tags: Any, recording: Dict[str, Any]) -> None:
    """Extract recording context from MP4/M4A atoms."""
    _MP4_MAP = {
        "©too": "encoder",
        "©cmt": "comment",
        "©day": "date",
        "©des": "description",
        "©lyr": "lyrics",
        "purd": "purchase_date",
    }
    for atom, field in _MP4_MAP.items():
        if atom in tags:
            val = tags[atom]
            if isinstance(val, list) and val:
                val = val[0]
            if val:
                recording[field] = str(val)

    # Custom/freeform atoms (----:com.apple.iTunes:*)
    for key in tags:
        if key.startswith("----:"):
            parts = key.split(":", 2)
            if len(parts) == 3:
                field_name = parts[2].lower()
                val = tags[key]
                if isinstance(val, list) and val:
                    val = val[0]
                if hasattr(val, "decode"):
                    val = val.decode("utf-8", errors="replace")
                _store_recording_field(field_name, str(val), recording)


def _extract_vorbis_recording(tags: Any, recording: Dict[str, Any]) -> None:
    """Extract recording context from Vorbis comments (FLAC, OGG)."""
    _VORBIS_RECORDING_KEYS = {
        "location", "gps", "latitude", "longitude",
        "device", "recording_device", "microphone",
        "recorder", "software", "hardware",
        "description", "notes", "session",
        "venue", "room", "studio",
        "producer", "engineer",
        "isrc", "barcode",
    }
    for key, values in tags.items():
        if key.lower() in _VORBIS_RECORDING_KEYS:
            val = values[0] if isinstance(values, list) and len(values) == 1 else values
            if isinstance(val, list):
                val = "; ".join(str(v) for v in val)
            recording[key.lower()] = str(val)


def _store_recording_field(name: str, value: str, recording: Dict[str, Any]) -> None:
    """Store a recording field, normalizing common names."""
    if not value or value.strip() == "":
        return

    # Normalize common extended tag names
    name = name.strip().lower().replace(" ", "_").replace("-", "_")
    _KNOWN_FIELDS = {
        "recording_device", "device", "microphone", "mic",
        "location", "gps", "gps_coordinates", "latitude", "longitude",
        "recording_location", "venue", "room", "studio",
        "recorder", "software", "hardware", "firmware",
        "recording_date", "recording_time", "session",
        "description", "notes", "producer", "engineer",
    }

    if name in _KNOWN_FIELDS or name.startswith(("recording_", "gps_", "device_")):
        recording[name] = value.strip()


def _normalize_tag_dict(tags: Dict[str, str]) -> Dict[str, str]:
    """Normalize a tag dict: lowercase keys, strip values, skip empty."""
    return {
        k.lower().replace(" ", "_"): v.strip()
        for k, v in tags.items()
        if v and v.strip()
    }


def _human_size(size_bytes: int) -> str:
    """Convert bytes to human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}" if unit != "B" else f"{size_bytes} B"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"
