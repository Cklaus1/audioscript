#!/usr/bin/env python3
"""
Simple demo script for AudioScript.
This script creates a few sample audio files and runs a mock version of AudioScript on them.
"""

import os
import shutil
import subprocess
import sys
import json
from pathlib import Path
import time

# Create some sample audio files
def create_samples():
    """Create dummy audio files for testing."""
    sample_dir = Path("samples")
    sample_dir.mkdir(exist_ok=True)
    
    # Create a few sample files
    for i in range(1, 4):
        filename = sample_dir / f"sample{i}.mp3"
        with open(filename, "w") as f:
            f.write(f"This is dummy audio content for file {i}")
    
    print(f"Created 3 sample files in {sample_dir.absolute()}")
    return sample_dir


def run_mock_audioscript(sample_dir):
    """Run a mock version of AudioScript on the sample files."""
    # Install the package in development mode
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], check=True)
    
    # Create an output directory
    output_dir = Path("output")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir()
    
    print("\nRunning mock AudioScript to process sample files")
    print("="*50)
    
    # Process each sample file
    for sample_file in sample_dir.glob("*.mp3"):
        print(f"Transcribing audio file: {sample_file.name} with tier: draft")
        
        # Create a mock transcription result
        mock_result = {
            "text": f"This is a mock transcription for {sample_file.name}.",
            "segments": [
                {
                    "id": 0,
                    "start": 0.0,
                    "end": 10.0,
                    "text": f"This is a mock transcription segment for {sample_file.name}."
                }
            ],
            "language": "en",
            "file_info": {
                "filename": str(sample_file.name),
                "duration_seconds": 10.0,
                "processed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "tier": "draft",
                "version": "1.0"
            }
        }
        
        # Save the mock transcription
        output_file = output_dir / f"{sample_file.stem}.json"
        with open(output_file, "w") as f:
            json.dump(mock_result, f, indent=2)
        
        # Create a mock summary
        summary_file = output_dir / f"{sample_file.stem}.summary.txt"
        with open(summary_file, "w") as f:
            f.write(f"Summary of {sample_file.name}: This is a mock audio transcription summary.")
        
        print(f"Successfully processed {sample_file.name}")
        print(f"Output saved to {output_file}")
        print("-"*50)
    
    print("="*50)
    
    # Show the generated files
    print("\nGenerated files:")
    for filepath in output_dir.glob("**/*"):
        if filepath.is_file():
            print(f"- {filepath.relative_to(output_dir)}")


if __name__ == "__main__":
    sample_dir = create_samples()
    run_mock_audioscript(sample_dir)
    print("\nDone! Mock AudioScript processed the sample files successfully.")
    print("Check the 'output' directory for the results.\n")