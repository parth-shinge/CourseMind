"""Application configuration and environment settings.

Loads environment variables from .env via python-dotenv and exposes
a centralized Settings object for paths, keys, and providers.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Attempt to load .env from project root
try:
    from dotenv import load_dotenv

    _DOTENV_AVAILABLE = True
except ImportError:
    _DOTENV_AVAILABLE = False


# Root directory of the repository
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load .env if dotenv is installed and the file exists
ENV_PATH = PROJECT_ROOT / ".env"
if _DOTENV_AVAILABLE and ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)
elif _DOTENV_AVAILABLE:
    load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Central settings and configuration values for the application."""

    # Project directories
    PROJECT_ROOT: Path = PROJECT_ROOT
    DATA_DIR: Path = PROJECT_ROOT / "data"
    RAW_VIDEOS_DIR: Path = PROJECT_ROOT / "data" / "raw_videos"
    TRANSCRIPTS_DIR: Path = PROJECT_ROOT / "data" / "transcripts"
    NOTES_DIR: Path = PROJECT_ROOT / "data" / "notes"
    VECTOR_DB_DIR: Path = PROJECT_ROOT / "data" / "vector_db"
    LOGS_DIR: Path = PROJECT_ROOT / "data" / "logs"

    # LLM Settings
    LLM_PROVIDER: str = field(
        default_factory=lambda: os.getenv("LLM_PROVIDER", "gemini").strip().lower()
    )
    ANTHROPIC_API_KEY: str = field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", "")
    )
    GEMINI_API_KEY: str = field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY", "")
    )
    GEMINI_MODEL: str = field(
        default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-3.5-flash").strip()
    )
    ANTHROPIC_MODEL: str = field(
        default_factory=lambda: os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6").strip()
    )

    # Transcription Settings
    WHISPER_MODEL_SIZE: str = field(
        default_factory=lambda: os.getenv("WHISPER_MODEL_SIZE", "small").strip()
    )
    WHISPER_DEVICE: str = field(
        default_factory=lambda: os.getenv("WHISPER_DEVICE", "cpu").strip().lower()
    )

    def ensure_dirs(self) -> None:
        """Ensure all required data directories exist on disk."""
        for directory in (
            self.DATA_DIR,
            self.RAW_VIDEOS_DIR,
            self.TRANSCRIPTS_DIR,
            self.NOTES_DIR,
            self.VECTOR_DB_DIR,
            self.LOGS_DIR,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def get_api_key(self, provider: str | None = None) -> str:
        """Get the API key for the active or specified provider."""
        active_provider = (provider or self.LLM_PROVIDER).lower()
        if active_provider == "gemini":
            return self.GEMINI_API_KEY
        elif active_provider == "anthropic":
            return self.ANTHROPIC_API_KEY
        return ""


# Default singleton instance
settings = Settings()
