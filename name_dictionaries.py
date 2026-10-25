"""Словари имён и суффиксы отчеств, используемые парсером ФИО."""
from __future__ import annotations

# Типичные русские суффиксы отчеств (в нижнем регистре, ё уже заменено на е)
PATRONYMIC_SUFFIXES: tuple[str, ...] = (
    "ович", "евич", "ич", "овна", "евна", "ична", "инична",
)

# Распространённые имена (русские и английские), нормализованные:
# нижний регистр, ё -> е. Нужны, чтобы определять порядок «Имя Фамилия»
# vs «Фамилия Имя» при разборе NER-упоминаний.
GIVEN_NAMES: frozenset[str] = frozenset({
    # Русские
    "александр", "александра", "алексей", "алена", "анастасия",
    "андрей", "анна", "артем", "борис", "вадим", "валентина",
    "валерий", "василий", "виктор", "виктория", "владимир", "владислав",
    "дарья", "денис", "дмитрий", "евгений", "екатерина", "елена",
    "иван", "игорь", "илья", "ирина", "кирилл", "константин", "максим",
    "мария", "михаил", "никита", "николай", "ольга", "павел", "петр",
    "полина", "роман", "светлана", "сергей", "станислав",
    "татьяна", "тимур", "юлия", "юрий",
    # Английские
    "aaron", "adam", "alex", "alexander", "alexandra", "alice", "amanda",
    "andrew", "anna", "anthony", "benjamin", "brandon", "brian",
    "charles", "charlotte", "christopher", "daniel", "david", "elizabeth",
    "emily", "emma", "ethan", "george", "grace", "henry", "isabella", "jack",
    "jacob", "james", "jennifer", "john", "joseph", "julia", "karen",
    "kevin", "laura", "liam", "linda", "lucas", "madison", "margaret", "mark",
    "mary", "matthew", "michael", "nicholas", "oliver", "patrick", "paul", "peter",
    "richard", "robert", "sarah", "scott", "stephen", "steven", "thomas", "william",
})
