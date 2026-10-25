"""Юнит-тесты чистого домена (без модели и БД): text_utils, name_parser, ner_core."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from name_parser import PersonName, parse_name
from ner_core import (
    filter_entities,
    group_entities_by_person,
    make_attribute,
    merge_attribute,
    validate_entity,
)
from text_utils import (
    canonicalize_name,
    normalize_key,
    sentence_index,
    split_sentences,
)


# ---------------------------------------------------------------------------
# text_utils
# ---------------------------------------------------------------------------

class TestNormalizeKey:
    def test_casefold_and_spaces(self):
        assert normalize_key("  Иван   Петров ") == "иван петров"

    def test_non_string(self):
        assert normalize_key(42) == "42"


class TestCanonicalizeName:
    def test_comma_reorder(self):
        assert canonicalize_name("Петров, Иван") == "Иван Петров"

    def test_plain(self):
        assert canonicalize_name("  Иван   Петров ") == "Иван Петров"

    def test_empty_right_part(self):
        assert canonicalize_name("Петров,") == "Петров,"


class TestSentences:
    def test_split_basic(self):
        text = "Первое предложение. Второе предложение! Третье?"
        spans = split_sentences(text)
        assert len(spans) == 3
        assert spans[0][0] == 0
        assert spans[-1][1] == len(text)

    def test_no_punctuation_returns_whole(self):
        text = "просто текст без знаков"
        assert split_sentences(text) == [(0, len(text))]

    def test_paragraph_breaks(self):
        text = "Абзац один.\n\nАбзац два."
        spans = split_sentences(text)
        assert len(spans) == 2

    def test_sentence_index(self):
        text = "Раз. Два. Три."
        spans = split_sentences(text)
        assert sentence_index(spans, 0) == 0
        assert sentence_index(spans, len(text)) == len(spans) - 1


# ---------------------------------------------------------------------------
# name_parser
# ---------------------------------------------------------------------------

class TestParseName:
    def test_single_word_is_last_name(self):
        assert parse_name("Сталин") == PersonName(None, "Сталин", None)

    def test_first_last_russian(self):
        assert parse_name("Иван Петров") == PersonName("Иван", "Петров", None)

    def test_last_first_via_dictionary(self):
        assert parse_name("Петров Иван") == PersonName("Иван", "Петров", None)

    def test_full_name_patronymic_order(self):
        assert parse_name("Сергей Иванов Иванович") == PersonName(
            "Сергей", "Иванов", "Иванович"
        )

    def test_last_first_patronymic_order(self):
        assert parse_name("Иванов Сергей Иванович") == PersonName(
            "Сергей", "Иванов", "Иванович"
        )

    def test_yo_normalization(self):
        assert parse_name("Петрова Анна Фёдоровна") == PersonName(
            "Анна", "Петрова", "Фёдоровна"
        )

    def test_comma_format(self):
        assert parse_name("Фамилия, Иван") == PersonName("Иван", "Фамилия", None)

    def test_unknown_two_words_defaults_first_as_name(self):
        parsed = parse_name("Хоббит Гэндальф")
        assert parsed.first_name == "Хоббит"
        assert parsed.last_name == "Гэндальф"

    def test_full_name_property(self):
        assert parse_name("Иванов Сергей Иванович").full_name == "Сергей Иванов Иванович"


# ---------------------------------------------------------------------------
# ner_core: фильтрация сущностей
# ---------------------------------------------------------------------------

def raw(entity_type, text, start, end, confidence):
    return {"type": entity_type, "text": text, "start": start, "end": end,
            "confidence": confidence}


class TestFilterEntities:
    TEXT = "Иван Петров работает в ACME."

    def test_drops_low_confidence(self):
        entities = filter_entities(
            [raw("Person", "Иван Петров", 0, 11, 0.4)], self.TEXT, confidence_threshold=0.5
        )
        assert entities == []

    def test_drops_invalid_offsets(self):
        entities = filter_entities(
            [raw("Person", "bad", 5, 2, 0.9), raw("Person", "bad", -1, 3, 0.9),
             raw("Person", "bad", 0, 999, 0.9)],
            self.TEXT, confidence_threshold=0.5,
        )
        assert entities == []

    def test_keeps_and_sorts_valid(self):
        entities = filter_entities(
            [raw("Organization", "ACME", 21, 25, 0.8),
             raw("Person", "Иван Петров", 0, 11, 0.9)],
            self.TEXT, confidence_threshold=0.5,
        )
        assert [e["type"] for e in entities] == ["Person", "Organization"]
        assert entities[0]["start"] == 0

    def test_none_confidence_dropped(self):
        entities = filter_entities(
            [{"type": "Person", "text": "x", "start": 0, "end": 1, "confidence": None}],
            self.TEXT, confidence_threshold=0.5,
        )
        assert entities == []


# ---------------------------------------------------------------------------
# ner_core: группировка по людям
# ---------------------------------------------------------------------------

TEXT_TWO_PEOPLE = (
    "Иван Петров — инженер из Москвы. "
    "Мария Сидорова работает директором в компании ACME."
)


def entities_two_people():
    ivan = TEXT_TWO_PEOPLE.index("Иван Петров")
    moscow = TEXT_TWO_PEOPLE.index("Москвы")
    maria = TEXT_TWO_PEOPLE.index("Мария Сидорова")
    director = TEXT_TWO_PEOPLE.index("директором")
    acme = TEXT_TWO_PEOPLE.index("ACME")
    return [
        {"type": "Person", "text": "Иван Петров", "start": ivan, "end": ivan + 11, "confidence": 0.9},
        {"type": "Location", "text": "Москвы", "start": moscow, "end": moscow + 6, "confidence": 0.8},
        {"type": "Person", "text": "Мария Сидорова", "start": maria, "end": maria + 14, "confidence": 0.9},
        {"type": "Job Title", "text": "директором", "start": director, "end": director + 10, "confidence": 0.7},
        {"type": "Organization", "text": "ACME", "start": acme, "end": acme + 4, "confidence": 0.8},
    ]


def _group(text, entities, window=1):
    return group_entities_by_person(
        text, entities, parse_name_fn=parse_name, source_file="f.txt",
        context_sentence_window=window,
    )


class TestGroupByPerson:
    def test_attributes_assigned_to_closest_person(self):
        result = _group(TEXT_TWO_PEOPLE, entities_two_people())
        people = {p["full_name"]: p for p in result["people"]}
        ivan = people["Иван Петров"]
        maria = people["Мария Сидорова"]
        assert [a["value"] for a in ivan["locations"]] == ["Москвы"]
        assert [a["value"] for a in maria["job_titles"]] == ["директором"]
        assert [a["value"] for a in maria["organizations"]] == ["ACME"]
        assert ivan["first_name"] == "Иван" and ivan["last_name"] == "Петров"

    def test_no_person_all_unassigned(self):
        entities = [{"type": "Location", "text": "Париж", "start": 0, "end": 5, "confidence": 0.9}]
        result = _group("Париж красивый.", entities)
        assert result["people"] == []
        assert result["unassigned_entities"][0]["reason"] == "no_person_in_text"

    def test_outside_window_unassigned(self):
        entities = entities_two_people()
        result = _group(TEXT_TWO_PEOPLE, entities, window=0)
        unassigned_values = {e["text"] for e in result["unassigned_entities"]}
        assert all(v in {"Москвы", "директором", "ACME"} for v in unassigned_values)

    def test_duplicate_attribute_merges_confidence(self):
        text = "Иван Петров из Москвы. Иван Петров снова из Москвы."
        i1 = text.index("Иван Петров")
        m1 = text.index("Москвы")
        i2 = text.index("Иван Петров", i1 + 1)
        m2 = text.index("Москвы", m1 + 1)
        entities = [
            {"type": "Person", "text": "Иван Петров", "start": i1, "end": i1 + 11, "confidence": 0.9},
            {"type": "Location", "text": "Москвы", "start": m1, "end": m1 + 6, "confidence": 0.6},
            {"type": "Person", "text": "Иван Петров", "start": i2, "end": i2 + 11, "confidence": 0.9},
            {"type": "Location", "text": "Москвы", "start": m2, "end": m2 + 6, "confidence": 0.85},
        ]
        result = _group(text, entities)
        assert len(result["people"]) == 1
        loc = result["people"][0]["locations"][0]
        assert loc["confidence"] == pytest.approx(0.85)
        assert loc["sources"] == ["f.txt"]

    def test_unknown_type_goes_to_other_entities(self):
        text = "Иван Петров видел НЛО."
        p = text.index("Иван Петров")
        u = text.index("НЛО")
        entities = [
            {"type": "Person", "text": "Иван Петров", "start": p, "end": p + 11, "confidence": 0.9},
            {"type": "UFO", "text": "НЛО", "start": u, "end": u + 3, "confidence": 0.9},
        ]
        result = _group(text, entities)
        assert [a["value"] for a in result["people"][0]["other_entities"]] == ["НЛО"]


class TestMergeAttribute:
    def test_merge_updates_max_confidence(self):
        from ner_core import PersonProfile

        profile = PersonProfile(full_name="X")
        merge_attribute(profile, "locations", make_attribute(
            {"text": "Москва", "confidence": 0.5}, "a.txt"))
        merge_attribute(profile, "locations", make_attribute(
            {"text": " москва ", "confidence": 0.9}, "b.txt"))
        assert len(profile.locations) == 1
        assert profile.locations[0]["confidence"] == 0.9
        assert profile.locations[0]["sources"] == ["a.txt", "b.txt"]


class TestValidateEntity:
    def test_rounds_confidence(self):
        e = validate_entity("Person", raw("Person", "X", 0, 1, 0.9123456), 10, 0.5)
        assert e["confidence"] == 0.9123

    def test_bad_confidence_type(self):
        assert validate_entity("Person", raw("Person", "X", 0, 1, "abc"), 10, 0.5) is None
