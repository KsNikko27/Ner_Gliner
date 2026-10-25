"""Разбор ФИО: определение порядка «Имя Фамилия» / «Фамилия Имя» и отчеств."""
from __future__ import annotations

from dataclasses import dataclass

from name_dictionaries import GIVEN_NAMES, PATRONYMIC_SUFFIXES
from text_utils import canonicalize_name, is_patronymic, lower_yo


@dataclass(frozen=True)
class PersonName:
    """Разобранное имя человека."""

    first_name: str | None
    last_name: str | None
    patronymic: str | None

    @property
    def full_name(self) -> str:
        parts = [p for p in (self.first_name, self.last_name, self.patronymic) if p]
        return " ".join(parts)


def parse_name(raw_name: str) -> PersonName:
    """Пытается разложить строку с именем на (first, last, patronymic).

    Эвристики:
      * «Фамилия, Имя» сначала канонизируется в «Имя Фамилия».
      * Одиночное слово трактуется как фамилия/псевдоним (last_name).
      * Если третье (или последнее) слово похоже на отчество, оно им и считается;
        порядок «Имя Фамилия Отчество» vs «Фамилия Имя Отчество» определяется
        по словарю распространённых имён.
      * Для двух слов — то из них, что найдено в словаре имён, считается именем.
    """
    clean = canonicalize_name(raw_name)
    parts = clean.split()

    if len(parts) == 1:
        return PersonName(first_name=None, last_name=parts[0], patronymic=None)

    # Три и более слова: проверяем отчество на третьей позиции или в конце.
    if len(parts) >= 3:
        for idx in (2, len(parts) - 1):
            if not is_patronymic(parts[idx], PATRONYMIC_SUFFIXES):
                continue
            second = lower_yo(parts[1])
            if second in GIVEN_NAMES:
                # «Фамилия Имя Отчество»
                return PersonName(
                    first_name=parts[1], last_name=parts[0], patronymic=parts[idx]
                )
            # «Имя Фамилия Отчество»
            return PersonName(
                first_name=parts[0], last_name=parts[1], patronymic=parts[idx]
            )

    first_norm = lower_yo(parts[0])
    second_norm = lower_yo(parts[1])
    if first_norm in GIVEN_NAMES:
        # «Имя Фамилия ...»
        return PersonName(
            first_name=parts[0],
            last_name=" ".join(parts[1:]),
            patronymic=None,
        )
    if second_norm in GIVEN_NAMES:
        # «Фамилия Имя [Отчество]»
        patronymic = parts[2] if len(parts) > 2 else None
        return PersonName(first_name=parts[1], last_name=parts[0], patronymic=patronymic)

    # Неопознанное: считаем первое слово именем (как в исходной реализации).
    return PersonName(
        first_name=parts[0],
        last_name=" ".join(parts[1:]),
        patronymic=None,
    )
