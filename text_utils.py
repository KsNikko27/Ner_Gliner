"""Чистилка и базовые утилиты работы с текстом и именами.

Модуль не зависит от моделей, БД и сторонних библиотек — его легко
тестировать изолированно.
"""
from __future__ import annotations

import re
from collections.abc import Iterable

# Нормализация: ё -> е (встречается в словарях имён и проверках отчеств)
_YO_REPLACEMENTS = str.maketrans("ёЕ", "ее")


def normalize_key(value: object) -> str:
    """Нормализует значение в ключ сравнения: нижний регистр, одинарные пробелы."""
    return " ".join(str(value).casefold().split())


def canonicalize_name(full_name: str) -> str:
    """Приводит формат «Фамилия, Имя» к виду «Имя Фамилия».

    Лишние пробелы убираются; если запятой нет — возвращается очищенное имя.
    """
    clean = " ".join(full_name.strip().split())
    if "," in clean:
        left, right = (part.strip() for part in clean.split(",", 1))
        if left and right:
            return f"{right} {left}"
    return clean


def _lower(word: str) -> str:
    return word.casefold().translate(_YO_REPLACEMENTS)


def lower_yo(word: str) -> str:
    """Нижний регистр с заменой ё→е (публичная версия)."""
    return _lower(word)


def is_patronymic(word: str, suffixes: Iterable[str]) -> bool:
    """Проверяет, является ли слово отчеством по типовым суффиксам."""
    low = _lower(word)
    return any(low.endswith(suffix) for suffix in suffixes)


def split_sentences(text: str) -> list[tuple[int, int]]:
    """Возвращает spans предложений (start, end) для текста.

    Предложение заканчивается на знаке препинания (.!?) с опциональными
    закрывающими символами либо на пустой строке. Если разбиение не найдено
    — весь текст считается одним предложением.
    """
    spans: list[tuple[int, int]] = []
    start = 0
    for match in re.finditer(r"[.!?]+[\"''”)}\]]*\s+|\n+", text):
        end = match.end()
        if text[start:end].strip():
            spans.append((start, end))
        start = end
    if start < len(text) and text[start:].strip():
        spans.append((start, len(text)))
    return spans or [(0, len(text))]


def sentence_index(spans: list[tuple[int, int]], position: int) -> int:
    """Определяет номер предложения, содержащего позицию."""
    for idx, (start, end) in enumerate(spans):
        if start <= position < end:
            return idx
    return max(0, len(spans) - 1)
