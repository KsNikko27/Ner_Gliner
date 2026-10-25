"""Файловый слой пайплайна: чтение .txt, поиск HTML-снимков, архивация папок."""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def find_oldest_folder(input_dir: Path) -> Path | None:
    """Находит подпапку с самым старым временем изменения."""
    folders = [p for p in input_dir.iterdir() if p.is_dir()]
    if not folders:
        return None
    return min(folders, key=lambda p: p.stat().st_mtime)


def pick_first_text_file(folder: Path, pattern: str = "*.txt") -> Path | None:
    """Выбирает первый (по имени) текстовый файл из папки; предупреждает о дублях."""
    txt_files = sorted(folder.glob(pattern))
    if not txt_files:
        logger.info("В папке %s нет файлов %s — пропускаем.", folder.name, pattern)
        return None
    if len(txt_files) > 1:
        logger.warning(
            "В папке %s найдено %d файлов %s. Используется первый: %s",
            folder.name, len(txt_files), pattern, txt_files[0].name,
        )
    return txt_files[0]


def find_html_snapshot(folder: Path) -> str | None:
    """Возвращает имя (без расширения) первого HTML-файла в папке, если он есть."""
    html_candidates = sorted(folder.glob("*.html"))
    if not html_candidates:
        logger.warning("HTML-файл не найден в папке: %s", folder)
        return None
    if len(html_candidates) > 1:
        logger.warning(
            "В папке найдено %d HTML-файлов. Используется первый: %s",
            len(html_candidates), html_candidates[0].name,
        )
    return html_candidates[0].stem


def unique_archive_destination(archive_dir: Path, folder_name: str) -> Path:
    """Формирует свободное имя в архиве: name, name_1, name_2 ..."""
    dest = archive_dir / folder_name
    if not dest.exists():
        return dest
    counter = 1
    while True:
        candidate = archive_dir / f"{folder_name}_{counter}"
        if not candidate.exists():
            return candidate
        counter += 1


def move_to_archive(source_folder: Path, archive_dir: Path) -> Path:
    """Перемещает папку-источник в архив, разрешая конфликты имён."""
    archive_path = Path(archive_dir).resolve()
    archive_path.mkdir(parents=True, exist_ok=True)
    dest = unique_archive_destination(archive_path, source_folder.name)
    shutil.move(str(source_folder), str(dest))
    logger.info("Папка перемещена в архив: %s", dest)
    return dest
