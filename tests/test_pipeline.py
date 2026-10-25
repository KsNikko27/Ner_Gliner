"""Интеграционные тесты пайплайна с заглушками extractor/storage и fs-утилит."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

import file_pipeline
from ner_pipeline import NERFactory


# ---------------------------------------------------------------------------
# Заглушки
# ---------------------------------------------------------------------------

class FakeExtractor:
    """Возвращает заранее заданные сырые сущности (как gliner-обёртка)."""

    def __init__(self, entities):
        self.entities = entities
        self.calls = []

    def extract(self, text, labels):
        self.calls.append(text)
        return self.entities


class RecordingStorage:
    def __init__(self):
        self.saved = []

    def save_grouped_result(self, grouped_result, page_title, *, html_snapshot=None):
        self.saved.append({
            "grouped_result": grouped_result,
            "page_title": page_title,
            "html_snapshot": html_snapshot,
        })


def person_entities(text):
    ivan = text.index("Иван Петров")
    acme = text.index("ACME")
    return [
        {"type": "Person", "text": "Иван Петров", "start": ivan, "end": ivan + 11,
         "confidence": 0.95},
        {"type": "Organization", "text": "ACME", "start": acme, "end": acme + 4,
         "confidence": 0.8},
    ]


# ---------------------------------------------------------------------------
# file_pipeline
# ---------------------------------------------------------------------------

class TestFilePipeline:
    def test_find_oldest_folder(self, tmp_path):
        a = tmp_path / "a"
        b = tmp_path / "b"
        a.mkdir()
        b.mkdir()
        old_ts = time.time() - 3600
        os.utime(a, (old_ts, old_ts))
        assert file_pipeline.find_oldest_folder(tmp_path) == a

    def test_find_oldest_folder_empty(self, tmp_path):
        assert file_pipeline.find_oldest_folder(tmp_path) is None

    def test_pick_first_text_file_warns_on_multiple(self, tmp_path, caplog):
        (tmp_path / "b.txt").write_text("second")
        (tmp_path / "a.txt").write_text("first")
        chosen = file_pipeline.pick_first_text_file(tmp_path)
        assert chosen.name == "a.txt"

    def test_pick_first_text_file_none(self, tmp_path):
        assert file_pipeline.pick_first_text_file(tmp_path) is None

    def test_find_html_snapshot(self, tmp_path):
        (tmp_path / "z.html").write_text("<html/>")
        (tmp_path / "a.html").write_text("<html/>")
        assert file_pipeline.find_html_snapshot(tmp_path) == "a"

    def test_find_html_snapshot_missing(self, tmp_path):
        assert file_pipeline.find_html_snapshot(tmp_path) is None

    def test_unique_archive_destination_conflict(self, tmp_path):
        archive = tmp_path / "archive"
        archive.mkdir()
        (archive / "folder").mkdir()
        dest = file_pipeline.unique_archive_destination(archive, "folder")
        assert dest.name == "folder_1"

    def test_move_to_archive(self, tmp_path):
        src_parent = tmp_path / "input"
        src = src_parent / "folder"
        src.mkdir(parents=True)
        (src / "f.txt").write_text("x")
        archive = tmp_path / "archive"
        dest = file_pipeline.move_to_archive(src, archive)
        assert dest.exists()
        assert not src.exists()
        assert (dest / "f.txt").read_text() == "x"


# ---------------------------------------------------------------------------
# NERFactory (фасад)
# ---------------------------------------------------------------------------

class TestNERFactoryAnalyzeText:
    def test_type_error_for_non_string(self):
        factory = NERFactory(FakeExtractor([]), RecordingStorage())
        with pytest.raises(TypeError):
            factory.analyze_text(None)

    def test_empty_text_returns_nothing_without_model_call(self):
        extractor = FakeExtractor([{"junk": True}])
        factory = NERFactory(extractor, RecordingStorage())
        assert factory.analyze_text("   ") == []
        assert extractor.calls == []

    def test_threshold_filtering(self):
        text = "Иван Петров из ACME."
        entities = person_entities(text) + [
            {"type": "Location", "text": "ACME", "start": text.index("ACME"),
             "end": text.index("ACME") + 4, "confidence": 0.2},
        ]
        factory = NERFactory(FakeExtractor(entities), RecordingStorage(),
                             confidence_threshold=0.5)
        result = factory.analyze_text(text)
        assert {e["type"] for e in result} == {"Person", "Organization"}

    def test_invalid_threshold_raises(self):
        with pytest.raises(ValueError):
            NERFactory(FakeExtractor([]), RecordingStorage(), confidence_threshold=1.5)


class TestNERFactoryAnalyzeFile:
    def _make_folder(self, tmp_path, text="Иван Петров работает в ACME.", with_html=True):
        folder = tmp_path / "input" / "article-1"
        folder.mkdir(parents=True)
        txt = folder / "article-1.txt"
        txt.write_text(text, encoding="utf-8")
        if with_html:
            (folder / "article-1.html").write_text("<html></html>")
        return folder, txt

    def test_full_flow_saves_and_archives(self, tmp_path):
        folder, txt = self._make_folder(tmp_path)
        storage = RecordingStorage()
        factory = NERFactory(FakeExtractor(person_entities(txt.read_text())), storage)
        archive = tmp_path / "archive"

        assert factory.analyze_file(txt, archive_dir=archive) is True

        saved = storage.saved[0]
        assert saved["page_title"] == "article-1"
        assert saved["html_snapshot"] == "article-1"
        people = saved["grouped_result"]["people"]
        assert people[0]["full_name"] == "Иван Петров"
        assert people[0]["organizations"][0]["value"] == "ACME"

        # Папка перемещена в архив
        assert not folder.exists()
        assert (archive / "article-1" / "article-1.txt").exists()

    def test_no_html_snapshot_is_none(self, tmp_path):
        folder, txt = self._make_folder(tmp_path, with_html=False)
        storage = RecordingStorage()
        factory = NERFactory(FakeExtractor(person_entities(txt.read_text())), storage)
        factory.analyze_file(txt)
        assert storage.saved[0]["html_snapshot"] is None
        assert folder.exists()  # без archive_dir папка не трогается

    def test_missing_file_raises(self, tmp_path):
        factory = NERFactory(FakeExtractor([]), RecordingStorage())
        with pytest.raises(FileNotFoundError):
            factory.analyze_file(tmp_path / "nope.txt")

    def test_archive_name_conflict_resolved(self, tmp_path):
        folder, txt = self._make_folder(tmp_path)
        archive = tmp_path / "archive"
        (archive / "article-1").mkdir(parents=True)  # конфликт
        factory = NERFactory(FakeExtractor(person_entities(txt.read_text())),
                             RecordingStorage())
        factory.analyze_file(txt, archive_dir=archive)
        assert (archive / "article-1_1" / "article-1.txt").exists()


class TestNERFactoryAnalyzeDirectory:
    def test_batch_report(self, tmp_path):
        good = tmp_path / "in" / "good"
        good.mkdir(parents=True)
        (good / "g.txt").write_text("Иван Петров из ACME.")
        bad = tmp_path / "in" / "bad"
        bad.mkdir(parents=True)
        (bad / "b.txt").write_text("текст")

        class BoomExtractor(FakeExtractor):
            def extract(self, text, labels):
                if "б" in text or text == "текст":
                    raise RuntimeError("model exploded")
                return super().extract(text, labels)

        factory = NERFactory(BoomExtractor(person_entities("Иван Петров из ACME.")),
                             RecordingStorage())
        report = factory.analyze_directory(tmp_path / "in")
        assert len(report["success"]) == 1
        assert len(report["errors"]) == 1
        assert "model exploded" in report["errors"][0]["error"]

    def test_not_a_directory(self, tmp_path):
        factory = NERFactory(FakeExtractor([]), RecordingStorage())
        with pytest.raises(NotADirectoryError):
            factory.analyze_directory(tmp_path / "missing")


class TestProcessOldestInputFolder:
    def test_processes_oldest(self, tmp_path):
        input_dir = tmp_path / "input"
        newer = input_dir / "newer"
        older = input_dir / "older"
        newer.mkdir(parents=True)
        older.mkdir(parents=True)
        ts = time.time() - 7200
        os.utime(older, (ts, ts))
        text = "Иван Петров из ACME."
        (older / "older.txt").write_text(text, encoding="utf-8")

        storage = RecordingStorage()
        factory = NERFactory(FakeExtractor(person_entities(text)), storage)
        archive = tmp_path / "archive"

        assert factory.process_oldest_input_folder(input_dir, archive) is True
        assert storage.saved[0]["page_title"] == "older"
        assert (archive / "older").exists()

    def test_missing_input_dir(self, tmp_path):
        factory = NERFactory(FakeExtractor([]), RecordingStorage())
        assert factory.process_oldest_input_folder(
            tmp_path / "missing", tmp_path / "archive") is False

    def test_no_subfolders(self, tmp_path):
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        factory = NERFactory(FakeExtractor([]), RecordingStorage())
        assert factory.process_oldest_input_folder(
            input_dir, tmp_path / "archive") is False
