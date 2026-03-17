"""Speaker diarization using pyannote-audio."""

import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
import torchaudio

logger = logging.getLogger(__name__)

from audioscript.config.settings import DEFAULT_DIARIZATION_MODEL, DEFAULT_VAD_MODEL


class SpeakerDatabase:
    """Persistent database of named speaker embeddings for identification.

    Stores speaker name -> embedding mappings in a JSON file.
    Used to replace anonymous labels (SPEAKER_00) with real names
    by matching embeddings via cosine similarity.
    """

    def __init__(self, db_path: Union[str, Path]):
        self.db_path = Path(db_path)
        self.data = self._load()

    def _load(self) -> Dict[str, Any]:
        if not self.db_path.exists():
            return {"version": "1.0", "speakers": {}}
        try:
            with open(self.db_path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load speaker database %s: %s", self.db_path, e)
            return {"version": "1.0", "speakers": {}}

    def save(self) -> None:
        """Save the speaker database to disk."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.db_path, "w") as f:
            json.dump(self.data, f, indent=2)

    @property
    def speaker_names(self) -> List[str]:
        return list(self.data["speakers"].keys())

    def add_speaker(self, name: str, embedding: np.ndarray) -> None:
        """Add or update a speaker in the database.

        If the speaker already exists, the embedding is averaged with
        the existing one (running mean) for better robustness.
        """
        emb_list = embedding.tolist()
        if name in self.data["speakers"]:
            existing = self.data["speakers"][name]
            old_emb = np.array(existing["embedding"])
            n = existing.get("num_samples", 1)
            # Running mean: new_mean = (old_mean * n + new) / (n + 1)
            averaged = ((old_emb * n) + embedding) / (n + 1)
            self.data["speakers"][name] = {
                "embedding": averaged.tolist(),
                "num_samples": n + 1,
            }
        else:
            self.data["speakers"][name] = {
                "embedding": emb_list,
                "num_samples": 1,
            }
        self.save()

    def remove_speaker(self, name: str) -> bool:
        """Remove a speaker from the database. Returns True if found."""
        if name in self.data["speakers"]:
            del self.data["speakers"][name]
            self.save()
            return True
        return False

    def identify(
        self,
        embeddings: Dict[str, np.ndarray],
        threshold: float = 0.7,
    ) -> Dict[str, str]:
        """Match anonymous speaker embeddings against the database.

        Args:
            embeddings: Map of anonymous label -> embedding vector.
            threshold: Minimum cosine similarity to consider a match.

        Returns:
            Map of anonymous label -> identified name (only for matches).
        """
        if not self.data["speakers"]:
            return {}

        db_names = []
        db_embeddings = []
        for name, info in self.data["speakers"].items():
            db_names.append(name)
            db_embeddings.append(np.array(info["embedding"]))

        db_matrix = np.stack(db_embeddings)
        mapping = {}
        used_names = set()

        # Compute all similarities, then greedily assign best matches
        scores = []
        for anon_label, anon_emb in embeddings.items():
            for i, db_name in enumerate(db_names):
                sim = _cosine_similarity(anon_emb, db_matrix[i])
                scores.append((sim, anon_label, db_name))

        # Sort by similarity descending, greedily assign
        scores.sort(key=lambda x: -x[0])
        used_anon = set()
        for sim, anon_label, db_name in scores:
            if sim < threshold:
                break
            if anon_label in used_anon or db_name in used_names:
                continue
            mapping[anon_label] = db_name
            used_anon.add(anon_label)
            used_names.add(db_name)

        return mapping


class SpeakerDiarizer:
    """Handles speaker diarization using pyannote-audio.

    Requires a HuggingFace access token with access to the pyannote
    models. Set via --hf-token, hf_token in config, or the HF_TOKEN
    environment variable.
    """

    def __init__(
        self,
        hf_token: Optional[str] = None,
        model: str = DEFAULT_DIARIZATION_MODEL,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
        segmentation_batch_size: Optional[int] = None,
        embedding_batch_size: Optional[int] = None,
    ):
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        if not self.hf_token:
            raise ValueError(
                "Speaker diarization requires a HuggingFace access token. "
                "Provide it via --hf-token, hf_token in .audioscript.yaml, "
                "or the HF_TOKEN environment variable. "
                "Get a token at https://hf.co/settings/tokens and accept "
                "the model conditions at https://hf.co/pyannote/speaker-diarization-3.1"
            )

        self.model_name = model
        self.cache_dir = cache_dir
        self.segmentation_batch_size = segmentation_batch_size
        self.embedding_batch_size = embedding_batch_size

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self._pipeline = None
        self._vad_pipeline = None

    def _load_pipeline(self) -> None:
        """Lazily load the diarization pipeline."""
        if self._pipeline is not None:
            return

        from pyannote.audio import Pipeline

        logger.info(
            "Loading diarization pipeline '%s' on %s...",
            self.model_name, self.device,
        )
        kwargs = {"checkpoint": self.model_name, "token": self.hf_token}
        if self.cache_dir:
            kwargs["cache_dir"] = self.cache_dir

        self._pipeline = Pipeline.from_pretrained(**kwargs)
        if self._pipeline is None:
            raise RuntimeError(
                f"Failed to load diarization pipeline '{self.model_name}'. "
                "Check your HuggingFace token and model access permissions."
            )

        # Apply batch size overrides
        if self.segmentation_batch_size is not None:
            self._pipeline.segmentation_batch_size = self.segmentation_batch_size
        if self.embedding_batch_size is not None:
            self._pipeline.embedding_batch_size = self.embedding_batch_size

        self._pipeline.to(torch.device(self.device))
        logger.info("Diarization pipeline loaded successfully.")

    def _load_audio(self, audio_path: Union[str, Path]) -> Dict[str, Any]:
        """Load audio via torchaudio (avoids torchcodec issues)."""
        waveform, sample_rate = torchaudio.load(str(audio_path))
        return {"waveform": waveform, "sample_rate": sample_rate}

    def diarize(
        self,
        audio_path: Union[str, Path],
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        allow_overlap: bool = False,
        progress_callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """Run speaker diarization on an audio file.

        Args:
            audio_path: Path to the audio file.
            num_speakers: Exact number of speakers (if known).
            min_speakers: Minimum expected speakers.
            max_speakers: Maximum expected speakers.
            allow_overlap: If True, use full diarization with overlapping speech.
                          If False (default), use exclusive (no-overlap) diarization.
            progress_callback: Optional callback(step_name, completed, total).

        Returns:
            Dict with segments, speakers, embeddings, and overlap stats.
        """
        self._load_pipeline()

        audio_path = Path(audio_path)
        audio_input = self._load_audio(audio_path)

        kwargs = {}
        if num_speakers is not None:
            kwargs["num_speakers"] = num_speakers
        if min_speakers is not None:
            kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            kwargs["max_speakers"] = max_speakers

        # Build hook for progress reporting
        hook = None
        if progress_callback:
            hook = _make_progress_hook(progress_callback)

        logger.info("Running diarization on %s...", audio_path.name)
        if hook:
            output = self._pipeline(audio_input, hook=hook, **kwargs)
        else:
            output = self._pipeline(audio_input, **kwargs)

        # Choose annotation based on overlap preference
        if allow_overlap:
            annotation = output.speaker_diarization
        else:
            annotation = output.exclusive_speaker_diarization

        # Extract segments
        segments = []
        speakers = set()
        for segment, _, speaker in annotation.itertracks(yield_label=True):
            segments.append({
                "start": round(segment.start, 3),
                "end": round(segment.end, 3),
                "speaker": speaker,
            })
            speakers.add(speaker)

        # Compute overlap statistics from the full (non-exclusive) annotation
        overlap_stats = self._compute_overlap_stats(output.speaker_diarization)

        # Extract speaker embeddings
        speaker_embeddings = {}
        sorted_speakers = sorted(speakers)
        if output.speaker_embeddings is not None:
            for i, spk in enumerate(sorted_speakers):
                if i < len(output.speaker_embeddings):
                    speaker_embeddings[spk] = output.speaker_embeddings[i]

        return {
            "segments": segments,
            "num_speakers": len(speakers),
            "speakers": sorted_speakers,
            "speaker_embeddings": speaker_embeddings,
            "overlap": overlap_stats,
        }

    @staticmethod
    def _compute_overlap_stats(annotation) -> Dict[str, Any]:
        """Compute overlap statistics from a diarization annotation."""
        try:
            overlap_timeline = annotation.get_overlap()
            total_speech = sum(
                seg.duration for seg, _ in annotation.itertracks()
            )
            overlap_duration = sum(
                seg.duration for seg in overlap_timeline
            )
            # Total audio extent
            extent = annotation.get_timeline().extent()
            total_duration = extent.duration if extent else 0.0

            return {
                "overlap_duration": round(overlap_duration, 3),
                "total_speech_duration": round(total_speech, 3),
                "total_audio_duration": round(total_duration, 3),
                "overlap_percentage": round(
                    (overlap_duration / total_duration * 100) if total_duration > 0 else 0, 1
                ),
            }
        except Exception as e:
            logger.warning("Failed to compute overlap stats: %s", e)
            return {}

    def assign_speakers(
        self,
        whisper_result: Dict[str, Any],
        diarization: Dict[str, Any],
        speaker_db: Optional[SpeakerDatabase] = None,
        similarity_threshold: float = 0.7,
        allow_overlap: bool = False,
    ) -> Dict[str, Any]:
        """Assign speaker labels to Whisper transcription segments.

        Uses time-overlap majority vote. Optionally matches against a
        speaker database to replace SPEAKER_XX with real names.

        If allow_overlap is True, segments spanning multiple speakers
        get a list of speakers and an is_overlap flag.
        """
        diar_segments = diarization["segments"]

        # Build name mapping from speaker database if provided
        name_mapping = {}
        if speaker_db and diarization.get("speaker_embeddings"):
            name_mapping = speaker_db.identify(
                diarization["speaker_embeddings"],
                threshold=similarity_threshold,
            )
            if name_mapping:
                logger.info("Speaker identification: %s", name_mapping)

        for seg in whisper_result.get("segments", []):
            seg_start = seg["start"]
            seg_end = seg["end"]

            if allow_overlap:
                speakers = self._speakers_in_range(seg_start, seg_end, diar_segments)
                speakers = [name_mapping.get(s, s) for s in speakers]
                if len(speakers) > 1:
                    seg["speaker"] = speakers
                    seg["is_overlap"] = True
                elif speakers:
                    seg["speaker"] = speakers[0]
                else:
                    seg["speaker"] = "UNKNOWN"
            else:
                speaker = self._majority_speaker(seg_start, seg_end, diar_segments)
                seg["speaker"] = name_mapping.get(speaker, speaker)

            # Word-level assignment
            if "words" in seg:
                for word in seg["words"]:
                    word_speaker = self._majority_speaker(
                        word["start"], word["end"], diar_segments
                    )
                    word["speaker"] = name_mapping.get(word_speaker, word_speaker)

        # Build metadata
        identified_speakers = [
            name_mapping.get(s, s) for s in diarization["speakers"]
        ]
        whisper_result["diarization"] = {
            "num_speakers": diarization["num_speakers"],
            "speakers": identified_speakers,
            "overlap": diarization.get("overlap", {}),
        }
        if name_mapping:
            whisper_result["diarization"]["identified"] = name_mapping

        return whisper_result

    @staticmethod
    def _majority_speaker(
        start: float, end: float, diar_segments: List[Dict[str, Any]],
    ) -> str:
        """Find the speaker with the most time overlap in a given range."""
        speaker_durations: Dict[str, float] = defaultdict(float)
        for d in diar_segments:
            overlap_start = max(start, d["start"])
            overlap_end = min(end, d["end"])
            overlap = overlap_end - overlap_start
            if overlap > 0:
                speaker_durations[d["speaker"]] += overlap
        if not speaker_durations:
            return "UNKNOWN"
        return max(speaker_durations, key=speaker_durations.get)

    @staticmethod
    def _speakers_in_range(
        start: float, end: float, diar_segments: List[Dict[str, Any]],
        min_overlap: float = 0.1,
    ) -> List[str]:
        """Find all speakers with significant overlap in a given range."""
        speaker_durations: Dict[str, float] = defaultdict(float)
        for d in diar_segments:
            overlap_start = max(start, d["start"])
            overlap_end = min(end, d["end"])
            overlap = overlap_end - overlap_start
            if overlap > 0:
                speaker_durations[d["speaker"]] += overlap
        # Return speakers with at least min_overlap seconds
        return sorted(
            [s for s, dur in speaker_durations.items() if dur >= min_overlap],
            key=lambda s: -speaker_durations[s],
        )

    def detect_speech(
        self, audio_path: Union[str, Path],
        onset: float = 0.5,
        offset: float = 0.5,
        min_duration_on: float = 0.0,
        min_duration_off: float = 0.0,
    ) -> Dict[str, Any]:
        """Run standalone Voice Activity Detection.

        Returns speech/non-speech timeline. Can be used to pre-filter
        audio before transcription to reduce hallucinations on silence.

        Args:
            audio_path: Path to the audio file.
            onset: Onset threshold for speech detection.
            offset: Offset threshold for speech detection.
            min_duration_on: Minimum duration of speech segments.
            min_duration_off: Minimum duration of silence segments.

        Returns:
            Dict with speech_segments, total_speech_duration, total_duration.
        """
        if self._vad_pipeline is None:
            from pyannote.audio import Model
            from pyannote.audio.pipelines import VoiceActivityDetection

            logger.info("Loading VAD model...")
            vad_model = Model.from_pretrained(
                DEFAULT_VAD_MODEL, token=self.hf_token,
            )
            self._vad_pipeline = VoiceActivityDetection(segmentation=vad_model)
            self._vad_pipeline.instantiate({
                "onset": onset,
                "offset": offset,
                "min_duration_on": min_duration_on,
                "min_duration_off": min_duration_off,
            })
            self._vad_pipeline.to(torch.device(self.device))

        audio_path = Path(audio_path)
        audio_input = self._load_audio(audio_path)

        vad_result = self._vad_pipeline(audio_input)

        speech_segments = []
        total_speech = 0.0
        for segment in vad_result.get_timeline():
            speech_segments.append({
                "start": round(segment.start, 3),
                "end": round(segment.end, 3),
            })
            total_speech += segment.duration

        # Calculate total audio duration
        duration = audio_input["waveform"].shape[1] / audio_input["sample_rate"]

        return {
            "speech_segments": speech_segments,
            "total_speech_duration": round(total_speech, 3),
            "total_duration": round(duration, 3),
            "speech_percentage": round(
                (total_speech / duration * 100) if duration > 0 else 0, 1
            ),
        }

    def evaluate(
        self,
        diarization_segments: List[Dict[str, Any]],
        reference_rttm_path: Union[str, Path],
        file_id: str = "audio",
        collar: float = 0.25,
        skip_overlap: bool = False,
    ) -> Dict[str, float]:
        """Evaluate diarization accuracy against a reference RTTM file.

        Args:
            diarization_segments: Hypothesis diarization segments.
            reference_rttm_path: Path to reference RTTM file.
            file_id: File identifier matching the RTTM entries.
            collar: Tolerance collar in seconds around reference boundaries.
            skip_overlap: Whether to skip overlapping speech regions.

        Returns:
            Dict with DER, missed speech, false alarm, and confusion rates.
        """
        from pyannote.core import Annotation, Segment
        from pyannote.metrics.diarization import DiarizationErrorRate

        # Build hypothesis annotation
        hypothesis = Annotation()
        for seg in diarization_segments:
            hypothesis[Segment(seg["start"], seg["end"])] = seg["speaker"]

        # Parse reference RTTM
        reference = _load_rttm(reference_rttm_path, file_id)

        # Compute DER
        metric = DiarizationErrorRate(collar=collar, skip_overlap=skip_overlap)
        der = metric(reference, hypothesis)
        detail = metric[reference, hypothesis]

        return {
            "diarization_error_rate": round(der, 4),
            "missed_speech": round(detail.get("missed detection", 0), 4),
            "false_alarm": round(detail.get("false alarm", 0), 4),
            "speaker_confusion": round(detail.get("confusion", 0), 4),
            "total_speech": round(detail.get("total", 0), 4),
        }

    def save_embeddings(
        self,
        speaker_embeddings: Dict[str, np.ndarray],
        output_path: Union[str, Path],
    ) -> None:
        """Save speaker embeddings to a JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {}
        for speaker, embedding in speaker_embeddings.items():
            data[speaker] = {
                "embedding": embedding.tolist() if isinstance(embedding, np.ndarray) else list(embedding),
                "dimension": len(embedding),
            }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

    def save_rttm(
        self,
        diarization_segments: List[Dict[str, Any]],
        output_path: Union[str, Path],
        file_id: str = "audio",
    ) -> None:
        """Save diarization results in RTTM format."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            for seg in diarization_segments:
                duration = seg["end"] - seg["start"]
                f.write(
                    f"SPEAKER {file_id} 1 {seg['start']:.3f} {duration:.3f} "
                    f"<NA> <NA> {seg['speaker']} <NA> <NA>\n"
                )


# --- Utility functions ---

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _make_progress_hook(callback: Callable):
    """Create a pyannote-compatible hook that reports to a callback."""
    def hook(step_name, step_artefact, file=None, completed=None, total=None):
        callback(step_name, completed, total)
    return hook


def _load_rttm(rttm_path: Union[str, Path], file_id: str = "audio"):
    """Parse an RTTM file into a pyannote Annotation."""
    from pyannote.core import Annotation, Segment

    annotation = Annotation()
    rttm_path = Path(rttm_path)

    with open(rttm_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 9 or parts[0] != "SPEAKER":
                continue
            rttm_file_id = parts[1]
            if file_id and rttm_file_id != file_id:
                continue
            start = float(parts[3])
            duration = float(parts[4])
            speaker = parts[7]
            annotation[Segment(start, start + duration)] = speaker

    return annotation
