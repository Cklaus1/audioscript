# AudioScript Tasks

## Completed
- [x] CLI scaffolding with `typer`
- [x] Support CLI flags: `--input`, `--output-dir`, `--tier`, `--version`, `--clean-audio`, `--summarize`, `--force`, `--model`
- [x] Load and merge `.audioscript.yaml` config
- [x] Create file utility functions for managing files
- [x] Implement manifest system for tracking file processing state
- [x] Add skip logic for already processed files
- [x] Resume long transcription jobs from last checkpoint
- [x] Retry failed jobs unless suppressed
- [x] Implement basic project structure
- [x] Create comprehensive test suite
- [x] Add documentation
- [x] Integrate OpenAI's open-source Whisper model for local transcription
- [x] Map quality tiers to appropriate Whisper models
- [x] Add support for all Whisper model sizes
- [x] Implement checkpoint generation and loading
- [x] Create basic summarization functionality

## Future Enhancements
- [ ] Add real audio cleaning functionality
- [ ] Implement better error handling and reporting
- [ ] Add progress bars for long-running processes
- [ ] Support more output formats (text, SRT, VTT)
- [ ] Add speaker diarization
- [ ] Support batch processing with parallel execution
- [ ] Add more audio metadata extraction
- [ ] Implement custom vocabulary support
- [ ] Create a web interface
- [ ] Add more advanced summarization using LLMs
- [ ] Implement adjustable transcription parameters
- [ ] Add support for streaming transcription for real-time applications