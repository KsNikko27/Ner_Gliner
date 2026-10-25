"""Централизованная конфигурация проекта Ner_Gliner.

Все пути и параметры берутся из переменных окружения (с разумными
значениями по умолчанию), что убирает жёстко зашитые Windows-пути
и делает проект переносимым между машинами.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Корень проекта: .../Gliner (или любой другой каталог, где лежит config.py)
PROJECT_ROOT = Path(__file__).resolve().parent


def _env_path(name: str, default: Path) -> Path:
    raw = os.getenv(name)
    return Path(raw).expanduser().resolve() if raw else default


@dataclass(frozen=True)
class PathsConfig:
    input_dir: Path = field(
        default_factory=lambda: _env_path(
            "NGLINER_INPUT_DIR", PROJECT_ROOT / "Text" / "input"
        )
    )
    archive_dir: Path = field(
        default_factory=lambda: _env_path(
            "NGLINER_ARCHIVE_DIR", PROJECT_ROOT / "Text" / "cached_text"
        )
    )
    model_dir: Path = field(
        default_factory=lambda: _env_path(
            "NGLINER_MODEL_DIR", PROJECT_ROOT / "models" / "gliner2.5-multi-v1"
        )
    )


@dataclass(frozen=True)
class NERConfig:
    confidence_threshold: float = field(
        default_factory=lambda: float(os.getenv("NGLINER_CONFIDENCE_THRESHOLD", "0.5"))
    )
    context_sentence_window: int = field(
        default_factory=lambda: int(os.getenv("NGLINER_CONTEXT_WINDOW", "1"))
    )
    device: str | None = field(default_factory=lambda: os.getenv("NGLINER_DEVICE") or None)
    text_pattern: str = field(
        default_factory=lambda: os.getenv("NGLINER_TEXT_PATTERN", "*.txt")
    )


@dataclass(frozen=True)
class AppConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    ner: NERConfig = field(default_factory=NERConfig)


def load_config() -> AppConfig:
    """Загружает конфигурацию из переменных окружения."""
    return AppConfig()
