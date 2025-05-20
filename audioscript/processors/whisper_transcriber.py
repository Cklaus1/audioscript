"""Whisper model based transcription implementation."""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import whisper


class WhisperTranscriber:
    """Handles transcription using OpenAI's Whisper model."""
    
    TIER_TO_MODEL = {
        "draft": "base",  # Faster but less accurate
        "high_quality": "large",  # Slower but more accurate
    }
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        tier: str = "draft",
        device: Optional[str] = None,
    ):
        """Initialize the WhisperTranscriber.
        
        Args:
            model_name: Specific model name to use (overrides tier-based selection)
            tier: Transcription quality tier ('draft' or 'high_quality')
            device: Device to run model on ('cpu', 'cuda', etc.). If None, auto-detect.
        """
        # Determine which model to use based on tier, unless specific model specified
        if model_name is None:
            model_name = self.TIER_TO_MODEL.get(tier, "base")
            
        # Auto-detect device if not specified
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
        self.model_name = model_name
        self.device = device
        self.tier = tier
        self.model = None
        
        # Load model lazily only when first needed
        
    def load_model(self) -> None:
        """Load the Whisper model if not already loaded."""
        if self.model is None:
            print(f"Loading Whisper model '{self.model_name}' on {self.device}...")
            self.model = whisper.load_model(self.model_name, device=self.device)
            print(f"Model loaded successfully.")
    
    def transcribe(
        self,
        audio_path: Union[str, Path],
        language: Optional[str] = None,
        task: str = "transcribe",
        verbose: bool = False,
        temperature: float = 0,
        compression_ratio_threshold: Optional[float] = 2.4,
        logprob_threshold: Optional[float] = -0.5,  # Adjusted from -1.0 to reduce false positives
        no_speech_threshold: Optional[float] = 0.7,  # Increased from 0.6 to reduce repetition issues
        condition_on_previous_text: bool = True,
        checkpoint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Transcribe audio using the Whisper model.
        
        Args:
            audio_path: Path to the audio file
            language: Language code (e.g., 'en', 'fr', etc.), or None for auto-detect
            task: Either 'transcribe' or 'translate'
            verbose: Whether to display progress
            temperature: Sampling temperature
            compression_ratio_threshold: If the gzip compression ratio is higher than this, treat as failed
            logprob_threshold: If average token log probability is lower than this, treat as failed
            no_speech_threshold: If no_speech probability is higher than this, treat as silence
            condition_on_previous_text: Whether to use previous text as prompt for next window
            checkpoint: Optional checkpoint info for resuming
            
        Returns:
            Dictionary containing transcription results
        """
        self.load_model()
        
        # Handle checkpointing if provided
        initial_prompt = None
        if checkpoint:
            try:
                checkpoint_data = json.loads(checkpoint)
                if "text" in checkpoint_data:
                    initial_prompt = checkpoint_data["text"]
            except Exception:
                # If checkpoint parsing fails, just proceed without it
                pass
                
        # Convert Path to string if needed
        if isinstance(audio_path, Path):
            audio_path = str(audio_path)
            
        # Run transcription
        result = self.model.transcribe(
            audio_path,
            language=language,
            task=task,
            verbose=verbose,
            initial_prompt=initial_prompt,
            temperature=temperature,
            compression_ratio_threshold=compression_ratio_threshold,
            logprob_threshold=logprob_threshold,
            no_speech_threshold=no_speech_threshold,
            condition_on_previous_text=condition_on_previous_text,
        )

        # Post-process the result to remove duplicate repeated segments
        if "segments" in result:
            # Filter out segments that repeat the same text as the previous segment
            filtered_segments = []
            prev_text = None

            for segment in result["segments"]:
                current_text = segment.get("text", "").strip()

                # Skip empty segments or segments repeating the exact same text
                if not current_text or current_text == prev_text:
                    continue

                filtered_segments.append(segment)
                prev_text = current_text

            # Update the result with filtered segments
            result["segments"] = filtered_segments

            # Also update the full text to match filtered segments
            if filtered_segments:
                result["text"] = " ".join(seg.get("text", "").strip() for seg in filtered_segments)
        
        return result
    
    def transcribe_with_progress(
        self,
        audio_path: Union[str, Path],
        callback=None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Transcribe audio with progress tracking.
        
        Args:
            audio_path: Path to the audio file
            callback: Optional callback function to report progress
            **kwargs: Additional arguments for transcribe()
            
        Returns:
            Dictionary containing transcription results
        """
        # This is a simplified implementation
        # For real progress tracking, you would need to modify the Whisper source
        # or use a wrapper that calls the model on chunks of audio
        
        if callback:
            callback(0, "Loading model...")
            
        self.load_model()
        
        if callback:
            callback(10, "Starting transcription...")
            
        result = self.transcribe(audio_path, **kwargs)
        
        if callback:
            callback(100, "Transcription complete.")
            
        return result
    
    def clean_audio(self, audio_path: Union[str, Path], output_path: Optional[Union[str, Path]] = None) -> Path:
        """Clean audio file to improve transcription quality.
        
        Args:
            audio_path: Path to the input audio file
            output_path: Path to save the cleaned audio file, or None to use default
            
        Returns:
            Path to the cleaned audio file
        """
        # This is a placeholder for audio cleaning logic
        # In a real implementation, you would use librosa, ffmpeg, or other audio processing tools
        
        if output_path is None:
            input_path = Path(audio_path)
            output_path = input_path.parent / f"{input_path.stem}_cleaned{input_path.suffix}"
            
        # For now, we just copy the file as a placeholder
        import shutil
        shutil.copy(audio_path, output_path)
        
        return Path(output_path)
    
    def generate_summary(self, transcription: Dict[str, Any]) -> str:
        """Generate a summary of the transcription.
        
        Args:
            transcription: The transcription dictionary from Whisper
            
        Returns:
            A summary of the transcription text
        """
        # This is a placeholder for summary generation
        # In a real implementation, you would use an LLM or other summarization technique
        
        text = transcription.get("text", "")
        words = text.split()
        total_words = len(words)
        
        if total_words <= 25:
            return text
            
        # Very simple extractive summary (first 25 words)
        summary = " ".join(words[:25]) + "..."
        
        return summary
    
    def save_results(
        self,
        transcription: Dict[str, Any],
        output_path: Union[str, Path],
        include_segments: bool = True,
    ) -> None:
        """Save transcription results to a file.
        
        Args:
            transcription: The transcription dictionary from Whisper
            output_path: Path to save the results
            include_segments: Whether to include detailed segment information
        """
        # Create a copy of the transcription to modify
        output = dict(transcription)
        
        # Remove segments if not wanted (can make large files)
        if not include_segments:
            output.pop("segments", None)
            
        # Ensure the output directory exists
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as JSON
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
    
    def save_summary(self, summary: str, output_path: Union[str, Path]) -> None:
        """Save summary to a file.
        
        Args:
            summary: The summary text
            output_path: Path to save the summary
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(summary)
    
    def create_checkpoint(self, transcription: Dict[str, Any]) -> str:
        """Create a checkpoint from the current transcription state.
        
        Args:
            transcription: The current transcription dictionary
            
        Returns:
            Checkpoint data as a JSON string
        """
        # Create a simple checkpoint with the text so far
        checkpoint = {
            "text": transcription.get("text", ""),
        }
        
        return json.dumps(checkpoint)