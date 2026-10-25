"""Фасад NER-пайплайна: связывает экстрактор, доменное ядро и хранилище.

``NERFactory`` принимает готовые зависимости (extractor, storage), поэтому
в тестах их легко подменить заглушками. ``create_ner_factory()`` собирает
реальную конфигурацию «модель + PostgreSQL» из config.py.

(Модуль назван ner_pipeline; старое имя ner_factory_database сохранено как
тонкая реэкспорт-обёртка для обратной совместимости.)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from config import load_config
from file_pipeline import find_html_snapshot, move_to_archive, pick_first_text_file
from gliner_extractor import GlinerExtractor
from name_parser import PersonName, PersonProfile, parse_name
from ner_core import (
    LABELS_SCHEMA,
    PersonProfile,
    filter_entities,
    group_entities_by_person,
)

logger = logging.getLogger(__name__)


class NERFactory:
    """Оркестратор: текст → сущности → группировка по людям → хранилище → архив."""

    def __init__(
        self,
        extractor,
        storage,
        confidence_threshold: float = 0.5,
        context_sentence_window: int = 1,
    ):
        if not 0 <= confidence_threshold <= 1:
            raise ValueError("confidence_threshold должен быть от 0 до 1")
        self.extractor = extractor
        self.storage = storage
        self.confidence_threshold = confidence_threshold
        self.context_sentence_window = max(0, int(context_sentence_window))

    # ------------------------------------------------------------------
    # NER
    # ------------------------------------------------------------------

    def analyze_text(self, text: str) -> list[dict[str, Any]]:
        """Извлекает и фильтрует сущности из текста."""
        if not isinstance(text, str):
            raise TypeError("text должен быть строкой")
        if not text.strip():
            return []
        raw = self.extractor.extract(text, LABELS_SCHEMA)
        return filter_entities(raw, text, self.confidence_threshold)

    def group_by_person(
        self, text: str, entities: list[dict[str, Any]], source_file: str = "<unknown>"
    ) -> dict[str, Any]:
        return group_entities_by_person(
            text,
            entities,
            parse_name_fn=parse_name,
            source_file=source_file,
            context_sentence_window=self.context_sentence_window,
        )

    # ------------------------------------------------------------------
    # Файловый уровень
    # ------------------------------------------------------------------

    def analyze_file(self, input_file: str | Path, archive_dir: str | Path | None = None) -> bool:
        """Обрабатывает один .txt: NER → сохранение в БД → перемещение папки в архив."""
        input_path = Path(input_file).resolve()
        if not input_path.is_file():
            raise FileNotFoundError(f"Файл не найден: {input_path}")

        source_folder = input_path.parent

        # 1. Извлечение сущностей и группировка
        text = input_path.read_text(encoding="utf-8")
        entities = self.analyze_text(text)
        grouped_result = self.group_by_person(text, entities, source_file=str(input_path))

        # 2. HTML-скриншот в той же папке
        html_snapshot = find_html_snapshot(source_folder)
        page_title = input_path.stem

        # 3. Сохранение в хранилище
        self.storage.save_grouped_result(
            grouped_result, page_title=page_title, html_snapshot=html_snapshot
        )
        logger.info(
            "Обработан: %s | Сущностей: %d | Страница: %s | Снимок: %s",
            input_path.name, len(entities), page_title, html_snapshot,
        )

        # 4. Перемещение родительской папки в архив
        if archive_dir is not None:
            move_to_archive(source_folder, Path(archive_dir))

        return True

    def analyze_directory(
        self,
        input_dir: str | Path,
        recursive: bool = True,
        pattern: str = "*.txt",
        archive_dir: str | Path | None = None,
    ) -> dict[str, list[Any]]:
        """Пакетная обработка всех файлов в папке; возвращает отчёт об ошибках."""
        input_dir = Path(input_dir)
        if not input_dir.is_dir():
            raise NotADirectoryError(f"Папка не найдена: {input_dir}")
        files = input_dir.rglob(pattern) if recursive else input_dir.glob(pattern)
        report: dict[str, list[Any]] = {"success": [], "errors": []}
        for input_path in sorted(files):
            try:
                self.analyze_file(input_path, archive_dir=archive_dir)
                report["success"].append({"input": str(input_path)})
            except Exception as exc:  # noqa: BLE001 — пакетный режим: пишем в отчёт
                report["errors"].append({"input": str(input_path), "error": str(exc)})
                logger.error("Ошибка обработки %s: %s", input_path, exc)
        logger.info(
            "Готово. Успешно: %d; ошибок: %d",
            len(report["success"]), len(report["errors"]),
        )
        return report

    def process_oldest_input_folder(
        self,
        input_dir: str | Path,
        archive_dir: str | Path,
        pattern: str = "*.txt",
    ) -> bool:
        """Обрабатывает самый старый подкаталог input_dir (сценарий main.py)."""
        from file_pipeline import find_oldest_folder  # локальный импорт против цикла в module graph

        input_dir = Path(input_dir)
        if not input_dir.is_dir():
            logger.error("Входная директория не найдена: %s", input_dir)
            return False

        oldest = find_oldest_folder(input_dir)
        if oldest is None:
            logger.info("В директории нет подпапок для обработки.")
            return False

        txt_file = pick_first_text_file(oldest, pattern=pattern)
        if txt_file is None:
            return False

        logger.info("Обрабатывается папка: %s (файл: %s)", oldest.name, txt_file.name)
        self.analyze_file(txt_file, archive_dir=archive_dir)
        return True


def create_ner_factory(
    confidence_threshold: float | None = None,
    model_name: str | Path | None = None,
    device: str | None = None,
    context_sentence_window: int | None = None,
    storage=None,
    extractor=None,
) -> NERFactory:
    """Собирает боевой NERFactory из конфигурации окружения.

    Args:
        storage: необязательная готовая реализация хранилища (для тестов);
            по умолчанию PostgresStorage.initialize().
        extractor: необязательная готовая реализация экстрактора; по умолчанию
            ленивая GlinerExtractor.
    """
    cfg = load_config()
    if extractor is None:
        extractor = GlinerExtractor(
            model_name=model_name or cfg.paths.model_dir,
            device=device if device is not None else cfg.ner.device,
        )
    if storage is None:
        from postgres_storage import PostgresStorage

        storage = PostgresStorage()
        storage.initialize()

    return NERFactory(
        extractor=extractor,
        storage=storage,
        confidence_threshold=(
            confidence_threshold
            if confidence_threshold is not None
            else cfg.ner.confidence_threshold
        ),
        context_sentence_window=(
            context_sentence_window
            if context_sentence_window is not None
            else cfg.ner.context_sentence_window
        ),
    )
