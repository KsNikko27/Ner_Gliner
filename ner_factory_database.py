"""Обратная совместимость: старый модуль ``ner_factory_database``.

Весь код переехал в ``ner_pipeline`` (фасад), ``ner_core`` (домен),
``name_parser``/``text_utils`` (утилиты). Этот модуль лишь реэкспортирует
публичный API, чтобы существующие импорты продолжали работать.

Новый код должен импортировать ``ner_pipeline`` напрямую.
"""
from ner_pipeline import NERFactory, create_ner_factory  # noqa: F401

__all__ = ["NERFactory", "create_ner_factory"]
