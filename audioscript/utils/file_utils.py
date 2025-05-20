"""File utility functions for AudioScript."""

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


def get_file_hash(file_path: Path) -> str:
    """Calculate a hash for the file based on path and modification time.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Hash string that uniquely identifies the file
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
        
    # Get file stats
    stats = file_path.stat()
    
    # Create a hash from the file path and modification time
    # This way, if the file is modified, the hash will change
    hash_input = f"{file_path.absolute()}_{stats.st_mtime}"
    return hashlib.md5(hash_input.encode()).hexdigest()


def get_output_path(input_file: Path, output_dir: Path, ext: str = "json") -> Path:
    """Generate an output path for a processed file.
    
    Args:
        input_file: Path to the input file
        output_dir: Directory for output files
        ext: Output file extension (default: json)
        
    Returns:
        Path to the output file
    """
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Use the same basename as the input file but change the extension
    output_filename = f"{input_file.stem}.{ext}"
    return output_dir / output_filename


class ProcessingManifest:
    """Manages the tracking of processed files and their status."""
    
    def __init__(self, manifest_path: Path):
        """Initialize the manifest.
        
        Args:
            manifest_path: Path to the manifest file
        """
        self.manifest_path = manifest_path
        self.data = self._load_manifest()
        
    def _load_manifest(self) -> Dict[str, Any]:
        """Load the manifest file or create a new one if it doesn't exist.
        
        Returns:
            Dict containing the manifest data
        """
        if not self.manifest_path.exists():
            # Create an empty manifest
            return {
                "version": "1.0",
                "files": {},
            }
            
        try:
            with open(self.manifest_path, "r") as f:
                return json.load(f)
        except Exception:
            # If there's an error reading the manifest, create a new one
            return {
                "version": "1.0",
                "files": {},
            }
    
    def save(self) -> None:
        """Save the manifest to disk."""
        # Ensure the directory exists
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.manifest_path, "w") as f:
            json.dump(self.data, f, indent=2)
    
    def is_processed(self, file_hash: str, tier: str, version: str) -> bool:
        """Check if a file has already been processed at the given tier and version.
        
        Args:
            file_hash: Hash of the file
            tier: Transcription tier (e.g., "draft", "high_quality")
            version: Version string
            
        Returns:
            True if the file has been processed, False otherwise
        """
        # Check if the file exists in the manifest
        if file_hash not in self.data["files"]:
            return False
            
        # Check if the file has been processed at the given tier and version
        file_data = self.data["files"][file_hash]
        
        # Check if the processing was completed successfully
        if file_data.get("status") != "completed":
            return False
            
        # Check tier and version
        return (
            file_data.get("tier") == tier and
            file_data.get("version") == version
        )
    
    def update_file_status(
        self,
        file_hash: str,
        status: str,
        tier: str,
        version: str,
        checkpoint: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        """Update the status of a file in the manifest.
        
        Args:
            file_hash: Hash of the file
            status: Status of the processing (e.g., "processing", "completed", "error")
            tier: Transcription tier
            version: Version string
            checkpoint: Optional checkpoint info for resuming
            error: Optional error message if status is "error"
        """
        # Initialize the file entry if it doesn't exist
        if file_hash not in self.data["files"]:
            self.data["files"][file_hash] = {}
            
        # Update the file status
        self.data["files"][file_hash].update({
            "status": status,
            "tier": tier,
            "version": version,
            "last_updated": os.path.getmtime(self.manifest_path) if self.manifest_path.exists() else 0,
        })
        
        # Add optional fields if provided
        if checkpoint is not None:
            self.data["files"][file_hash]["checkpoint"] = checkpoint
            
        if error is not None:
            self.data["files"][file_hash]["error"] = error
            
        # Save the manifest
        self.save()
    
    def get_checkpoint(self, file_hash: str) -> Optional[str]:
        """Get the checkpoint information for a file.
        
        Args:
            file_hash: Hash of the file
            
        Returns:
            Checkpoint information or None if not available
        """
        if file_hash not in self.data["files"]:
            return None
            
        return self.data["files"][file_hash].get("checkpoint")
    
    def get_status(self, file_hash: str) -> Optional[str]:
        """Get the processing status of a file.
        
        Args:
            file_hash: Hash of the file
            
        Returns:
            Status string or None if the file is not in the manifest
        """
        if file_hash not in self.data["files"]:
            return None
            
        return self.data["files"][file_hash].get("status")