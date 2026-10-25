"""Обёртка над моделью GLINER2 (единственное место, где импортируются torch/gliner).

Благодаря этому доменное ядро (ner_core) и хранилище (postgres_storage)
остаются независимыми от ML-зависимостей и тестируются без GPU.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class GlinerExtractor:
    """Ленивая обёртка над ``gliner2.AutoExtractor``.

    Реализует протокол ``ner_core.EntityExtractor``: метод ``extract()``
    возвращает плоский список сырых сущностей.
    """

    def __init__(self, model_name: str | Path, device: str | None = None):
        self.model_name = str(model_name)
        self._device = device
        self._model = None  # загружается при первом обращении

    @property
    def device(self) -> str:
        if self._device is not None:
            return self._device
        import torch  # локальный импорт: не тянуть torch без необходимости

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        return self._device

    @property
    def model(self):
        if self._model is None:
            from gliner2 import AutoExtractor

            logger.info("Загрузка модели NER: %s (device=%s)", self.model_name, self.device)
            self._model = AutoExtractor.from_pretrained(
                self.model_name, map_location=self.device
            )
        return self._model

    def extract(self, text: str, labels: dict[str, str]) -> list[dict[str, Any]]:
        """Извлекает сущности из текста, нормализуя ответ модели к плоскому списку."""
        result = self.model.extract_entities(
            text,
            labels,
            include_spans=True,
            include_confidence=True,
        )
        raw_entities: list[dict[str, Any]] = []
        entities_by_type = result.get("entities", {}) if isinstance(result, dict) else {}
        for entity_type, items in entities_by_type.items():
            for entity in items:
                item = dict(entity)
                item["type"] = entity_type
                raw_entities.append(item)
        return raw_entities
