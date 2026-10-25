"""PostgreSQL-хранилище результатов NER (psycopg2).

Схема:
    article_archive     — статьи (уникальный page_title, имя HTML-снимка)
    people              — люди (ФИО в разложенном виде)
    person_attributes   — атрибуты людей (должность/организация/локация/прочее)
    person_mentions     — связь статья ↔ человек
    attribute_mentions  — связь статья ↔ атрибут

Модуль содержит также миграции legacy-таблиц (articles, people с UUID).
"""
from __future__ import annotations

import logging
import os
from typing import Any

import psycopg2

logger = logging.getLogger(__name__)

ATTRIBUTE_FIELDS: dict[str, str] = {
    "job_titles": "Job Title",
    "organizations": "Organization",
    "locations": "Location",
    "other_entities": "Other",
}


def _norm(value: object) -> str:
    return " ".join(str(value).casefold().split())


def _attribute_parts(attribute: Any) -> tuple[str, float]:
    """Разворачивает атрибут (dict или строку) в пару (value, confidence)."""
    if isinstance(attribute, dict):
        value = str(attribute.get("value", "")).strip()
        try:
            confidence = float(attribute.get("confidence", 0) or 0)
        except (TypeError, ValueError):
            confidence = 0.0
    else:
        value, confidence = str(attribute).strip(), 0.0
    return value, confidence


# ---------------------------------------------------------------------------
# SQL-операторы (DML) — вынесены в константы для читаемости и переиспользования
# ---------------------------------------------------------------------------

SQL_UPSERT_ARTICLE = """
INSERT INTO article_archive(page_title, html_snapshot, saved_at)
VALUES (%s, %s, now())
ON CONFLICT (page_title) DO UPDATE SET
    html_snapshot = COALESCE(EXCLUDED.html_snapshot, article_archive.html_snapshot),
    saved_at = CASE
        WHEN EXCLUDED.html_snapshot IS NOT NULL THEN now()
        ELSE article_archive.saved_at
    END
RETURNING article_id
"""

SQL_SELECT_PERSON_BY_ID = "SELECT person_id FROM people WHERE person_id = %s"

SQL_SELECT_PERSON_BY_NAME = """
SELECT person_id FROM people
WHERE lower(full_name) = lower(%s)
ORDER BY person_id
LIMIT 1
"""

SQL_UPDATE_PERSON_BY_ID = """
UPDATE people SET
    full_name = CASE
        WHEN full_name = '' OR full_name = 'Unknown' THEN %s
        ELSE full_name END,
    first_name = COALESCE(first_name, %s),
    last_name = COALESCE(last_name, %s),
    patronymic = COALESCE(patronymic, %s),
    updated_at = now()
WHERE person_id = %s
"""

SQL_UPDATE_PERSON_FIELDS = """
UPDATE people SET
    first_name = COALESCE(first_name, %s),
    last_name = COALESCE(last_name, %s),
    patronymic = COALESCE(patronymic, %s),
    updated_at = now()
WHERE person_id = %s
"""

SQL_INSERT_PERSON = """
INSERT INTO people(full_name, first_name, last_name, patronymic)
VALUES (%s, %s, %s, %s)
RETURNING person_id
"""

SQL_INSERT_PERSON_MENTION = """
INSERT INTO person_mentions(article_id, person_id, mentioned_name)
VALUES (%s, %s, %s) ON CONFLICT DO NOTHING
"""

SQL_INSERT_ATTRIBUTE = """
INSERT INTO person_attributes
    (person_id, attribute_type, value, normalized_value, confidence)
VALUES (%s, %s, %s, %s, %s)
ON CONFLICT (person_id, attribute_type, normalized_value)
DO UPDATE SET confidence = GREATEST(
    person_attributes.confidence, EXCLUDED.confidence
)
RETURNING attribute_id
"""

SQL_INSERT_ATTRIBUTE_MENTION = """
INSERT INTO attribute_mentions(article_id, attribute_id)
VALUES (%s, %s) ON CONFLICT DO NOTHING
"""


class PostgresStorage:
    """Хранилище NER-результатов в PostgreSQL."""

    def __init__(self, dsn: str | None = None):
        self.dsn = dsn or os.getenv("DATABASE_URL")
        self.conninfo = None
        if not self.dsn:
            self.conninfo = {
                "host": os.getenv("PGHOST", "localhost"),
                "port": int(os.getenv("PGPORT", "5432")),
                "dbname": os.getenv("PGDATABASE", "ner_db"),
                "user": os.getenv("PGUSER", "postgres"),
                "password": os.getenv("PGPASSWORD", ""),
            }

    def _connect(self):
        if self.dsn:
            return psycopg2.connect(self.dsn)
        return psycopg2.connect(**self.conninfo)

    # ------------------------------------------------------------------
    # Схема и миграции
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Создаёт схему; при необходимости мигрирует legacy-таблицы."""
        conn = self._connect()
        try:
            with conn, conn.cursor() as cur:
                legacy_people_uuid, legacy_articles = self._detect_legacy_state(cur)
                self._create_article_archive(cur)
                if legacy_articles:
                    self._migrate_legacy_articles(cur)
                self._ensure_people_table(cur, legacy_people_uuid)
                self._create_dependent_tables(cur)
                if legacy_articles:
                    cur.execute("DROP TABLE articles")
                self._create_indexes(cur)
        finally:
            conn.close()

    @staticmethod
    def _detect_legacy_state(cur) -> tuple[bool, bool]:
        """Определяет наличие legacy-таблиц: people с UUID и articles."""
        cur.execute("""
            SELECT data_type
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = 'people'
              AND column_name = 'person_id'
        """)
        person_id_info = cur.fetchone()
        legacy_people_uuid = person_id_info is not None and person_id_info[0] != "bigint"

        cur.execute("SELECT to_regclass('public.articles')")
        legacy_articles = cur.fetchone()[0] is not None
        return legacy_people_uuid, legacy_articles

    @staticmethod
    def _create_article_archive(cur) -> None:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS article_archive (
                article_id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
                page_title TEXT NOT NULL UNIQUE,
                html_snapshot TEXT,
                saved_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
        """)

    @staticmethod
    def _migrate_legacy_articles(cur) -> None:
        cur.execute("""
            INSERT INTO article_archive(article_id, page_title)
            OVERRIDING SYSTEM VALUE
            SELECT article_id, source
            FROM articles
            ON CONFLICT (article_id) DO NOTHING
        """)
        cur.execute("""
            SELECT setval(
                pg_get_serial_sequence('article_archive', 'article_id'),
                GREATEST(COALESCE((SELECT MAX(article_id) FROM article_archive), 1), 1),
                EXISTS (SELECT 1 FROM article_archive)
            )
        """)

    @staticmethod
    def _ensure_people_table(cur, legacy_people_uuid: bool) -> None:
        create_people = """
            CREATE TABLE IF NOT EXISTS people (
                person_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                full_name TEXT NOT NULL,
                first_name TEXT,
                last_name TEXT,
                patronymic TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
        """
        if not legacy_people_uuid:
            cur.execute(create_people)
            return

        # Миграция: old UUID-based people -> bigint identity
        cur.execute("DROP TABLE IF EXISTS attribute_mentions CASCADE")
        cur.execute("DROP TABLE IF EXISTS person_mentions CASCADE")
        cur.execute("DROP TABLE IF EXISTS person_attributes CASCADE")
        cur.execute("ALTER TABLE people RENAME TO people_old_uuid")
        cur.execute(create_people.replace("IF NOT EXISTS ", ""))
        cur.execute("""
            INSERT INTO people(full_name, first_name, last_name, patronymic,
                               created_at, updated_at)
            SELECT full_name, first_name, last_name, patronymic, created_at, updated_at
            FROM people_old_uuid
            ORDER BY created_at, full_name
        """)
        cur.execute("DROP TABLE people_old_uuid")

    @staticmethod
    def _create_dependent_tables(cur) -> None:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS person_attributes (
                attribute_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                person_id BIGINT NOT NULL REFERENCES people(person_id) ON DELETE CASCADE,
                attribute_type TEXT NOT NULL,
                value TEXT NOT NULL,
                normalized_value TEXT NOT NULL,
                confidence DOUBLE PRECISION NOT NULL DEFAULT 0,
                UNIQUE (person_id, attribute_type, normalized_value)
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS person_mentions (
                article_id BIGINT NOT NULL REFERENCES article_archive(article_id) ON DELETE CASCADE,
                person_id BIGINT NOT NULL REFERENCES people(person_id) ON DELETE CASCADE,
                mentioned_name TEXT NOT NULL,
                PRIMARY KEY (article_id, person_id, mentioned_name)
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS attribute_mentions (
                article_id BIGINT NOT NULL REFERENCES article_archive(article_id) ON DELETE CASCADE,
                attribute_id BIGINT NOT NULL REFERENCES person_attributes(attribute_id) ON DELETE CASCADE,
                PRIMARY KEY (article_id, attribute_id)
            )
        """)

    @staticmethod
    def _create_indexes(cur) -> None:
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_person_attributes_type "
            "ON person_attributes(attribute_type)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_person_attributes_value "
            "ON person_attributes(normalized_value)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_person_mentions_person "
            "ON person_mentions(person_id)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_article_archive_page_title "
            "ON article_archive(page_title)"
        )

    # ------------------------------------------------------------------
    # Запись результатов
    # ------------------------------------------------------------------

    def save_grouped_result(
        self,
        grouped_result: dict[str, Any],
        page_title: str,
        *,
        html_snapshot: str | None = None,
    ) -> None:
        """Сохраняет результат группировки: статью, людей и их атрибуты."""
        people = grouped_result.get("people", [])
        if not isinstance(people, list):
            raise TypeError("grouped_result['people'] должен быть списком")
        if not page_title or not page_title.strip():
            raise ValueError("Нужно передать page_title (имя файла без расширения).")

        conn = self._connect()
        try:
            with conn, conn.cursor() as cur:
                article_id = self._upsert_article(cur, page_title, html_snapshot)
                for person in people:
                    if not isinstance(person, dict):
                        continue
                    person_id = self._upsert_person(cur, person)
                    self._link_person_mention(cur, article_id, person_id, person)
                    self._save_person_attributes(cur, article_id, person_id, person)
        finally:
            conn.close()

    @staticmethod
    def _upsert_article(cur, page_title: str, html_snapshot: str | None) -> int:
        cur.execute(SQL_UPSERT_ARTICLE, (page_title, html_snapshot))
        return cur.fetchone()[0]

    @staticmethod
    def _upsert_person(cur, person: dict[str, Any]) -> int:
        """Возвращает person_id: по явному db-id, по полному имени, либо создаёт нового."""
        full_name = str(person.get("full_name") or "Unknown").strip()
        name_fields = (
            person.get("first_name"),
            person.get("last_name"),
            person.get("patronymic"),
        )

        requested_db_id = person.get("person_db_id")
        if requested_db_id is None and isinstance(person.get("person_id"), int):
            requested_db_id = person["person_id"]

        if requested_db_id is not None:
            cur.execute(SQL_SELECT_PERSON_BY_ID, (requested_db_id,))
            row = cur.fetchone()
            if row is None:
                raise ValueError(f"person_db_id={requested_db_id} не найден в таблице people")
            person_id = row[0]
            cur.execute(SQL_UPDATE_PERSON_BY_ID, (full_name, *name_fields, person_id))
            return person_id

        cur.execute(SQL_SELECT_PERSON_BY_NAME, (full_name,))
        row = cur.fetchone()
        if row:
            person_id = row[0]
            cur.execute(SQL_UPDATE_PERSON_FIELDS, (*name_fields, person_id))
            return person_id

        cur.execute(SQL_INSERT_PERSON, (full_name, *name_fields))
        return cur.fetchone()[0]

    @staticmethod
    def _link_person_mention(cur, article_id: int, person_id: int, person: dict) -> None:
        full_name = str(person.get("full_name") or "Unknown").strip()
        cur.execute(SQL_INSERT_PERSON_MENTION, (article_id, person_id, full_name))

    @staticmethod
    def _save_person_attributes(
        cur, article_id: int, person_id: int, person: dict[str, Any]
    ) -> None:
        for field, attribute_type in ATTRIBUTE_FIELDS.items():
            attributes = person.get(field, []) or []
            if not isinstance(attributes, (list, tuple, set)):
                attributes = [attributes]
            for attribute in attributes:
                value, confidence = _attribute_parts(attribute)
                if not value:
                    continue
                cur.execute(
                    SQL_INSERT_ATTRIBUTE,
                    (person_id, attribute_type, value, _norm(value), confidence),
                )
                attribute_id = cur.fetchone()[0]
                cur.execute(SQL_INSERT_ATTRIBUTE_MENTION, (article_id, attribute_id))

    # ------------------------------------------------------------------
    # Чтение / диагностика
    # ------------------------------------------------------------------

    def get_article_snapshot(self, page_title: str) -> dict[str, Any] | None:
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT article_id, page_title, html_snapshot, saved_at "
                    "FROM article_archive WHERE page_title = %s",
                    (page_title,),
                )
                row = cur.fetchone()
                if row is None:
                    return None
                return {
                    "article_id": row[0],
                    "page_title": row[1],
                    "html_snapshot": row[2],
                    "saved_at": row[3],
                }
        finally:
            conn.close()

    def test_connection(self) -> bool:
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                return cur.fetchone()[0] == 1
        finally:
            conn.close()
