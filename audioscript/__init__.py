"""AudioScript: CLI tool for audio transcription."""

__version__ = "0.2.0"

# Load .env file for API keys (ANTHROPIC_API_KEY, NVIDIA_API_KEY, HF_TOKEN, etc.)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Initialize structured logging on import
from audioscript.utils.logging import setup_logging
setup_logging()
