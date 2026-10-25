"""Доменное ядро NER-пайплайна: извлечение сущностей и группировка по людям.

Модуль не импортирует torch/gliner/postgres — модель внедряется через
параметр ``extractor`` (dependency injection), хранилище — через параметр
``storage``. Это делает код тестируемым без GPU и базы данных.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

from text_utils import (
    canonicalize_name,
    normalize_key,
    sentence_index,
    split_sentences,
)

logger = logging.getLogger(__name__)

# Описание типов сущностей для GLINER-модели
LABELS_SCHEMA: dict[str, str] = {
    "Person": (
        "Names of people, including first and last names, stage names, "
        "aliases and pseudonyms. Combine adjacent first and last names "
        "into one entity. Exclude titles and honorifics."
    ),
    "Job Title": (
        "Professional roles, ranks and positions held by people, "
        "including roles in descriptive phrases."
    ),
    "Organization": (
        "Companies, government agencies, educational institutions, media "
        "outlets, digital platforms and other organizations."
    ),
    "Location": (
        "Countries, states, regions, cities, streets, postal addresses "
        "and specific buildings."
    ),
}

# Поле профиля человека для типа сущности (кроме Person)
ENTITY_FIELD_BY_TYPE: dict[str, str] = {
    "Job Title": "job_titles",
    "Organization": "organizations",
    "Location": "locations",
}
DEFAULT_ENTITY_FIELD = "other_entities"


class EntityExtractor(Protocol):
    """Интерфейс NER-экстрактора (реализуется обёрткой над GLINER)."""

    def extract(self, text: str, labels: dict[str, str]) -> list[dict[str, Any]]:
        """Возвращает список сырых сущностей: type/text/start/end/confidence."""
        ...


class GroupedResultStorage(Protocol):
    """Интерфейс хранилища сгруппированного результата."""

    def save_grouped_result(
        self,
        grouped_result: dict[str, Any],
        page_title: str,
        *,
        html_snapshot: str | None = None,
    ) -> None:
        ...


@dataclass
class PersonMention:
    """Упоминание человека в тексте."""

    key: str
    start: int
    end: int
    sentence: int
    text: str


@dataclass
class PersonProfile:
    """Профиль человека: имя + приписанные атрибуты."""

    full_name: str
    first_name: str | None = None
    last_name: str | None = None
    patronymic: str | None = None
    job_titles: list[dict[str, Any]] = field(default_factory=list)
    organizations: list[dict[str, Any]] = field(default_factory=list)
    locations: list[dict[str, Any]] = field(default_factory=list)
    other_entities: list[dict[str, Any]] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "full_name": self.full_name,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "patronymic": self.patronymic,
            "job_titles": self.job_titles,
            "organizations": self.organizations,
            "locations": self.locations,
            "other_entities": self.other_entities,
            "sources": self.sources,
        }


def validate_entity(
    entity_type: str, entity: dict[str, Any], text_len: int, confidence_threshold: float
) -> dict[str, Any] | None:
    """Проверяет сырую сущность от модели; возвращает cleaned dict или None."""
    confidence = entity.get("confidence")
    if confidence is None:
        return None
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        return None
    if confidence < confidence_threshold:
        return None

    start, end, entity_text = entity.get("start"), entity.get("end"), entity.get("text")
    if (
        not isinstance(start, int)
        or not isinstance(end, int)
        or not isinstance(entity_text, str)
        or not (0 <= start < end <= text_len)
    ):
        return None

    return {
        "type": entity_type,
        "text": entity_text,
        "start": start,
        "end": end,
        "confidence": round(confidence, 4),
    }


def filter_entities(
    raw_entities: list[dict[str, Any]], text: str, confidence_threshold: float
) -> list[dict[str, Any]]:
    """Фильтрует и сортирует валидные сущности по позиции в тексте."""
    entities_out = []
    for raw in raw_entities:
        entity = validate_entity(
            entity_type=str(raw.get("type", "")),
            entity=raw,
            text_len=len(text),
            confidence_threshold=confidence_threshold,
        )
        if entity is not None:
            entities_out.append(entity)
    entities_out.sort(key=lambda item: (item["start"], item["end"]))
    return entities_out


def make_attribute(entity: dict[str, Any], source_file: str) -> dict[str, Any]:
    """Формирует запись атрибута из валидированной сущности."""
    return {
        "value": entity["text"].strip(),
        "confidence": entity["confidence"],
        "sources": [source_file],
    }


def merge_attribute(
    profile: PersonProfile, field_name: str, attr: dict[str, Any]
) -> None:
    """Добавляет атрибут в профиль или объединяет с существующим (макс. confidence)."""
    target = getattr(profile, field_name)
    existing = next(
        (item for item in target if normalize_key(item["value"]) == normalize_key(attr["value"])),
        None,
    )
    if existing is None:
        target.append(attr)
    else:
        existing["confidence"] = max(existing["confidence"], attr["confidence"])
        for source in attr["sources"]:
            if source not in existing["sources"]:
                existing["sources"].append(source)


def find_closest_mention(
    mentions: list[PersonMention],
    entity_start: int,
    entity_end: int,
    entity_sentence: int,
    context_sentence_window: int,
) -> PersonMention | None:
    """Находит ближайшее упоминание человека в пределах окна предложений."""
    candidates: list[tuple[int, int, int, PersonMention]] = []
    for mention in mentions:
        sentence_distance = abs(mention.sentence - entity_sentence)
        if sentence_distance > context_sentence_window:
            continue
        char_distance = min(
            abs(entity_start - mention.end),
            abs(mention.start - entity_end),
        )
        candidates.append((sentence_distance, char_distance, mention.start, mention))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    return candidates[0][3]


def group_entities_by_person(
    text: str,
    entities: list[dict[str, Any]],
    parse_name_fn,
    source_file: str = "<unknown>",
    context_sentence_window: int = 1,
) -> dict[str, Any]:
    """Группирует сущности: каждый не-Person атрибут приписывается ближайшему человеку.

    Args:
        parse_name_fn: функция (str) -> PersonName (из name_parser), используется
            для заполнения first/last/patronymic профиля.
    """
    spans = split_sentences(text)
    people: dict[str, PersonProfile] = {}
    person_mentions: list[PersonMention] = []
    unassigned: list[dict[str, Any]] = []

    # 1. Собираем людей и их упоминания
    for entity in entities:
        if entity.get("type") != "Person":
            continue
        name = " ".join(entity["text"].strip().split())
        if not name:
            continue
        canonical = canonicalize_name(name)
        key = normalize_key(canonical)
        if key not in people:
            parsed = parse_name_fn(canonical)
            people[key] = PersonProfile(
                full_name=canonical,
                first_name=parsed.first_name,
                last_name=parsed.last_name,
                patronymic=parsed.patronymic,
            )
        if source_file not in people[key].sources:
            people[key].sources.append(source_file)
        person_mentions.append(
            PersonMention(
                key=key,
                start=entity["start"],
                end=entity["end"],
                sentence=sentence_index(spans, entity["start"]),
                text=name,
            )
        )

    # 2. Приписываем остальные сущности ближайшим людям
    for entity in entities:
        entity_type = entity.get("type")
        if entity_type == "Person":
            continue
        if not person_mentions:
            unassigned.append({**entity, "reason": "no_person_in_text"})
            continue
        entity_sentence = sentence_index(spans, entity["start"])
        closest = find_closest_mention(
            person_mentions,
            entity["start"],
            entity["end"],
            entity_sentence,
            context_sentence_window,
        )
        if closest is None:
            unassigned.append({**entity, "reason": "no_person_in_local_context"})
            continue
        field_name = ENTITY_FIELD_BY_TYPE.get(entity_type, DEFAULT_ENTITY_FIELD)
        merge_attribute(
            people[closest.key], field_name, make_attribute(entity, source_file)
        )

    return {
        "people": [profile.to_dict() for profile in people.values()],
        "unassigned_entities": unassigned,
    }
