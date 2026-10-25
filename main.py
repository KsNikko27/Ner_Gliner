from pathlib import Path
from ner_factory_database import create_ner_factory


def find_oldest_folder(input_dir: Path) -> Path | None:
    """Находит подпапку с самым старым временем изменения."""
    folders = [p for p in input_dir.iterdir() if p.is_dir()]
    if not folders:
        return None
    return min(folders, key=lambda p: p.stat().st_mtime)


def main():
    input_dir = Path(r"C:\Users\rfhg3\Desktop\Project\Gliner\Text\input")
    archive_dir = Path(r"C:\Users\rfhg3\Desktop\Project\Gliner\Text\cached_text")

    if not input_dir.is_dir():
        print(f"Входная директория не найдена: {input_dir}")
        return

    oldest = find_oldest_folder(input_dir)
    if oldest is None:
        print("В директории нет подпапок для обработки.")
        return

    txt_files = sorted(oldest.glob("*.txt"))
    if not txt_files:
        print(f"В папке {oldest.name} нет .txt файлов — пропускаем.")
        return

    txt_file = txt_files[0]
    if len(txt_files) > 1:
        print(
            f"[Предупреждение] В папке {oldest.name} найдено {len(txt_files)} "
            f".txt файлов. Используется первый: {txt_file.name}"
        )

    print(f"Обрабатывается папка: {oldest.name} (файл: {txt_file.name})")

    ner = create_ner_factory(confidence_threshold=0.5)
    ner.analyze_file(
        input_file=str(txt_file),
        archive_dir=str(archive_dir),
    )


if __name__ == "__main__":
    main()