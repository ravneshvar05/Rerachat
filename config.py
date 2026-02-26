"""
config.py — Single source of truth for all configuration.
Reads from .env file. Import `settings` anywhere in the project.
"""

from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # ── LLM ──────────────────────────────────────────────────────────────────
    GROQ_API_KEY: str = Field(..., description="Groq API key")
    GROQ_MODEL: str = Field("llama3-8b-8192", description="Groq model name")

    # ── Paths ─────────────────────────────────────────────────────────────────
    JSON_OUTPUT_DIR: Path = Field(
        Path("./output"),
        description="Folder containing brochure JSON files",
    )
    DB_DIR: Path = Field(
        Path("./db"),
        description="Folder where SQLite + ChromaDB are stored",
    )

    # ── ChromaDB ──────────────────────────────────────────────────────────────
    CHROMA_COLLECTION_NAME: str = Field("rera_units")

    # ── App ───────────────────────────────────────────────────────────────────
    APP_TITLE: str = Field("RERA Project Finder")
    LOG_LEVEL: str = Field("INFO")

    # ── Derived paths (computed, not from .env) ────────────────────────────
    @property
    def sqlite_path(self) -> Path:
        return self.DB_DIR / "projects.db"

    @property
    def chroma_path(self) -> Path:
        return self.DB_DIR / "chroma"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",          # ignore unknown env vars silently
    }


# Singleton — import this everywhere
settings = Settings()

# Ensure required directories exist
settings.DB_DIR.mkdir(parents=True, exist_ok=True)
(Path("logs")).mkdir(exist_ok=True)
