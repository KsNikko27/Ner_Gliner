"""Точка входа: обработка самой старой папки во входном каталоге.

Пути и параметры берутся из config.py (переменные окружения NGLINER_*).
Логика пайплайна вынесена в NERFactory.process_oldest_input_folder().
"""
from __future__ import annotations

import logging

from config import load_config
from ner_pipeline import create_ner_factory


def main() -> int:
    cfg = load_config()
    factory = create_ner_factory(confidence_threshold=cfg.ner.confidence_threshold)
    processed = factory.process_oldest_input_folder(
        input_dir=cfg.paths.input_dir,
        archive_dir=cfg.paths.archive_dir,
        pattern=cfg.ner.text_pattern,
    )
    return 0 if processed else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    raise SystemExit(main())
