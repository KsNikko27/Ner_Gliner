import hashlib
import json
import re
import uuid
from pathlib import Path
from postgres_storage import PostgresStorage

import torch
from gliner2 import AutoExtractor


class NERFactory:
    """
    Извлекает NER-сущности, группирует их по ближайшему человеку и ведёт
    общий JSON-справочник людей, объединяя атрибуты из разных текстов.

    Формат базы:
      {
        "people": [
          {
            "person_id": "...",
            "first_name": "... или None",
            "last_name": "... или None",
            "patronymic": "... или None",
            "full_name": "...",
            "job_titles": [{"value": "...", "confidence": 0.9, "sources": [...]}],
            "organizations": [...],
            "locations": [...],
            "other_entities": [...]
          }
        ]
      }

    Важно: автоматическое сопоставление по имени не различает тёзок.
    Если в корпусе встречаются люди с одинаковым именем, для надёжного
    разрешения идентичности нужны дополнительные данные или ручная проверка.
    """

    DATABASE_FILENAME = "people_database.json"
    ID_NAMESPACE = uuid.UUID("c0e9d728-47de-4bdb-bb3b-22a9d876a211")

    def __init__(
        self,
        output_dir,
        confidence_threshold=0.5,
        model_name=r"C:\Users\rfhg3\Desktop\Project\Gliner\models\gliner2.5-multi-v1",
        device=None,
        context_sentence_window=1,
    ):
        self.output_dir = Path(output_dir)
        self.confidence_threshold = confidence_threshold
        self.context_sentence_window = max(0, int(context_sentence_window))

        if not 0 <= confidence_threshold <= 1:
            raise ValueError("confidence_threshold должен быть от 0 до 1")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.database_path = self.output_dir / self.DATABASE_FILENAME
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = AutoExtractor.from_pretrained(
            str(model_name),
            map_location=self.device,
        )

        self.labels_schema = {
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

        # Подсказки для русского и английского порядка имён. Это эвристика:
        # неизвестные имена всё равно обрабатываются по обычному порядку First Last.
        self._given_names = {
            "александр", "александра", "алексей", "алёна", "алена", "анастасия",
            "андрей", "анна", "артём", "артем", "борис", "вадим", "валентина",
            "валерий", "василий", "виктор", "виктория", "владимир", "владислав",
            "дарья", "денис", "дмитрий", "евгений", "екатерина", "елена",
            "иван", "игорь", "илья", "ирина", "кирилл", "константин", "максим",
            "мария", "михаил", "никита", "николай", "ольга", "павел", "пётр",
            "петр", "полина", "роман", "светлана", "сергей", "станислав",
            "татьяна", "тимур", "юлия", "юрий",
            "aaron", "adam", "alex", "alexander", "alexandra", "alice", "amanda",
            "andrew", "anna", "anthony", "benjamin", "benjamin", "brandon", "brian",
            "charles", "charlotte", "christopher", "daniel", "david", "elizabeth",
            "emily", "emma", "ethan", "george", "grace", "henry", "isabella", "jack",
            "jacob", "james", "jennifer", "john", "joseph", "julia", "julia", "karen",
            "kevin", "laura", "liam", "linda", "lucas", "madison", "margaret", "mark",
            "mary", "matthew", "michael", "nicholas", "oliver", "patrick", "paul", "peter",
            "richard", "robert", "sarah", "scott", "stephen", "steven", "thomas", "william",
        }
        self._patronymic_suffixes = (
            "ович", "евич", "ич", "овна", "евна", "ична", "инична",
        )
        self.pg_storage = PostgresStorage()
        self.pg_storage.initialize()
        print(f"NER initialized on {self.device}")

    @staticmethod
    def _norm(value):
        return " ".join(str(value).casefold().split())

    @classmethod
    def _person_id(cls, full_name):
        normalized = cls._norm(full_name)
        return str(uuid.uuid5(cls.ID_NAMESPACE, normalized))

    @staticmethod
    def _canonicalize_name(full_name):
        """Приводит формат «Фамилия, Имя» к «Имя Фамилия» для ключа профиля."""
        clean = " ".join(full_name.strip().split())
        if "," in clean:
            left, right = [part.strip() for part in clean.split(",", 1)]
            if left and right:
                return f"{right} {left}"
        return clean

    def _parse_name(self, full_name):
        """
        Возвращает first_name, last_name, patronymic.
        Поддерживает русские ФИО, английский порядок «First Last» и формат
        «Last, First». Неизвестные имена разбираются эвристически.
        """
        clean = self._canonicalize_name(full_name)
        parts = clean.split()
        first_name = last_name = patronymic = None

        if len(parts) == 1:
            # Нельзя достоверно определить, имя это или фамилия.
            return first_name, parts[0], patronymic

        def is_patronymic(word):
            low = word.casefold().replace("ё", "е")
            return any(low.endswith(suffix) for suffix in self._patronymic_suffixes)

        if len(parts) >= 3 and is_patronymic(parts[2]):
            if parts[1].casefold().replace("ё", "е") in self._given_names:
                last_name, first_name, patronymic = parts[0], parts[1], parts[2]
            else:
                first_name, last_name, patronymic = parts[0], parts[1], parts[2]
            return first_name, last_name, patronymic

        if len(parts) >= 3 and is_patronymic(parts[-1]):
            if parts[1].casefold().replace("ё", "е") in self._given_names:
                last_name, first_name, patronymic = parts[0], parts[1], parts[-1]
            else:
                first_name, last_name, patronymic = parts[0], parts[1], parts[-1]
            return first_name, last_name, patronymic

        first_norm = parts[0].casefold().replace("ё", "е")
        second_norm = parts[1].casefold().replace("ё", "е")
        if first_norm in self._given_names:
            first_name, last_name = parts[0], " ".join(parts[1:])
        elif second_norm in self._given_names:
            #
            last_name, first_name = parts[0], parts[1]
            if len(parts) > 2:
                patronymic = parts[2]
        else:
            # Для большинства англоязычных имён: Firstname Lastname.
            first_name, last_name = parts[0], " ".join(parts[1:])
        return first_name, last_name, patronymic

    def analyze_text(self, text):
        if not isinstance(text, str):
            raise TypeError("text должен быть строкой")
        if not text.strip():
            return []

        result = self.model.extract_entities(
            text,
            self.labels_schema,
            include_spans=True,
            include_confidence=True,
        )
        entities_out = []
        for entity_type, items in result.get("entities", {}).items():
            for entity in items:
                confidence = entity.get("confidence", 0.0)
                if confidence is None:
                    continue
                confidence = float(confidence)
                if confidence < self.confidence_threshold:
                    continue

                start, end = entity.get("start"), entity.get("end")
                entity_text = entity.get("text")
                if (
                    not isinstance(start, int)
                    or not isinstance(end, int)
                    or not isinstance(entity_text, str)
                    or not (0 <= start < end <= len(text))
                ):
                    continue

                entities_out.append({
                    "type": entity_type,
                    "text": entity_text,
                    "start": start,
                    "end": end,
                    "confidence": round(confidence, 4),
                })
        entities_out.sort(key=lambda item: (item["start"], item["end"]))
        return entities_out

    @staticmethod
    def _sentence_spans(text):
        spans = []
        start = 0
        for match in re.finditer(r"[.!?]+[\"'’”)}\]]*\s+|\n+", text):
            end = match.end()
            if text[start:end].strip():
                spans.append((start, end))
            start = end
        if start < len(text) and text[start:].strip():
            spans.append((start, len(text)))
        return spans or [(0, len(text))]

    @staticmethod
    def _sentence_index(spans, position):
        for idx, (start, end) in enumerate(spans):
            if start <= position < end:
                return idx
        return max(0, len(spans) - 1)

    @staticmethod
    def _attribute(entity, source_file):
        return {
            "value": entity["text"].strip(),
            "confidence": entity["confidence"],
            "sources": [source_file],
        }

    def group_by_person(self, text, entities, source_file="<unknown>"):
        """
        Привязка к ближайшему упоминанию персоны.
        Приоритет: то же предложение > число предложений > расстояние в символах.
        При равенстве берётся первое по тексту упоминание; сущность не теряется.
        """
        spans = self._sentence_spans(text)
        people = {}
        person_mentions = []
        unassigned = []

        for entity in entities:
            if entity.get("type") != "Person":
                continue
            name = " ".join(entity["text"].strip().split())
            if not name:
                continue
            canonical_name = self._canonicalize_name(name)
            key = self._norm(canonical_name)
            if key not in people:
                first_name, last_name, patronymic = self._parse_name(canonical_name)
                people[key] = {
                    "person_id": self._person_id(canonical_name),
                    "first_name": first_name,
                    "last_name": last_name,
                    "patronymic": patronymic,
                    "full_name": name,
                    "job_titles": [],
                    "organizations": [],
                    "locations": [],
                    "other_entities": [],
                    "sources": [],
                }
            mention = {
                "key": key,
                "start": entity["start"],
                "end": entity["end"],
                "sentence": self._sentence_index(spans, entity["start"]),
                "text": name,
            }
            person_mentions.append(mention)
            if source_file not in people[key]["sources"]:
                people[key]["sources"].append(source_file)

        for entity in entities:
            entity_type = entity.get("type")
            if entity_type == "Person":
                continue
            if not person_mentions:
                unassigned.append({**entity, "reason": "no_person_in_text"})
                continue

            entity_sentence = self._sentence_index(spans, entity["start"])
            candidates = []
            for mention in person_mentions:
                sentence_distance = abs(mention["sentence"] - entity_sentence)
                # Сохраняем локальность: атрибут не будет прикреплён к человеку
                # из совершенно другой части большой статьи.
                if sentence_distance > self.context_sentence_window:
                    continue
                char_distance = min(
                    abs(entity["start"] - mention["end"]),
                    abs(mention["start"] - entity["end"]),
                )
                candidates.append((
                    sentence_distance,
                    char_distance,
                    mention["start"],
                    mention["key"],
                ))

            if not candidates:
                unassigned.append({**entity, "reason": "no_person_in_local_context"})
                continue

            # «Ближе — значит больше подходит»: всегда выбираем ближайшее упоминание.
            candidates.sort(key=lambda item: (item[0], item[1], item[2]))
            best_key = candidates[0][3]
            field = {
                "Job Title": "job_titles",
                "Organization": "organizations",
                "Location": "locations",
            }.get(entity_type, "other_entities")
            attr = self._attribute(entity, source_file)
            profile = people[best_key]

            existing = next(
                (item for item in profile[field]
                 if self._norm(item["value"]) == self._norm(attr["value"])),
                None,
            )
            if existing is None:
                profile[field].append(attr)
            else:
                existing["confidence"] = max(existing["confidence"], attr["confidence"])
                if source_file not in existing["sources"]:
                    existing["sources"].append(source_file)

        return {
            "people": list(people.values()),
            "unassigned_entities": unassigned,
        }

    def _load_database(self):
        if not self.database_path.exists():
            return {"people": []}
        try:
            data = json.loads(self.database_path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("people"), list):
                return data
        except (json.JSONDecodeError, OSError):
            pass
        raise ValueError(
            f"Не удалось прочитать базу {self.database_path}. "
            "Сделайте резервную копию и проверьте JSON."
        )

    @staticmethod
    def _merge_values(target, incoming):
        for item in incoming:
            key = NERFactory._norm(item["value"])
            existing = next(
                (old for old in target if NERFactory._norm(old["value"]) == key),
                None,
            )
            if existing is None:
                target.append({
                    "value": item["value"],
                    "confidence": item["confidence"],
                    "sources": list(item.get("sources", [])),
                })
            else:
                existing["confidence"] = max(existing.get("confidence", 0), item["confidence"])
                for source in item.get("sources", []):
                    if source not in existing.setdefault("sources", []):
                        existing["sources"].append(source)

    def update_database(self, grouped_result):
        """Объединяет найденные профили с постоянной базой people_database.json."""
        database = self._load_database()
        by_id = {p["person_id"]: p for p in database["people"] if "person_id" in p}

        for incoming in grouped_result.get("people", []):
            person_id = incoming["person_id"]
            if person_id not in by_id:
                database["people"].append(incoming)
                by_id[person_id] = incoming
                continue

            current = by_id[person_id]
            # Не перетираем уже известные компоненты ФИО значением None.
            for field in ("first_name", "last_name", "patronymic"):
                if current.get(field) in (None, "") and incoming.get(field) not in (None, ""):
                    current[field] = incoming[field]
            for field in ("job_titles", "organizations", "locations", "other_entities"):
                self._merge_values(current.setdefault(field, []), incoming.get(field, []))
            for source in incoming.get("sources", []):
                if source not in current.setdefault("sources", []):
                    current["sources"].append(source)

        database["people"].sort(key=lambda p: (self._norm(p.get("last_name") or ""), self._norm(p.get("first_name") or "")))
        temp_path = self.database_path.with_name(self.database_path.name + ".tmp")
        try:
            temp_path.write_text(
                json.dumps(database, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            temp_path.replace(self.database_path)
        finally:
            if temp_path.exists():
                temp_path.unlink()
        return self.database_path

    def analyze_file(self, input_file, output_file=None):
        input_path = Path(input_file)
        if not input_path.is_file():
            raise FileNotFoundError(f"Файл не найден: {input_path}")

        text = input_path.read_text(encoding="utf-8")
        entities = self.analyze_text(text)
        grouped_result = self.group_by_person(
            text,
            entities,
            source_file=str(input_path),
        )
        self.pg_storage.save_grouped_result(
            grouped_result,
            source_file=input_path,
        )

        output_path = (
            Path(output_file)
            if output_file is not None
            else self.output_dir / f"{input_path.stem}_NER.json"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = output_path.with_name(output_path.name + ".tmp")
        try:
            temp_path.write_text(
                json.dumps(grouped_result, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            temp_path.replace(output_path)
        finally:
            if temp_path.exists():
                temp_path.unlink()

        database_path = self.update_database(grouped_result)
        print(
            f"Обработан: {input_path.name} | Сущностей: {len(entities)} | "
            f"JSON: {output_path} | База: {database_path}"
        )
        return output_path

    def analyze_directory(self, input_dir, recursive=True, pattern="*.txt"):
        input_dir = Path(input_dir)
        if not input_dir.is_dir():
            raise NotADirectoryError(f"Папка не найдена: {input_dir}")

        files = input_dir.rglob(pattern) if recursive else input_dir.glob(pattern)
        report = {"success": [], "errors": []}
        for input_path in sorted(files):
            relative_path = input_path.relative_to(input_dir)
            output_path = self.output_dir / relative_path.with_suffix(".json")
            try:
                self.analyze_file(input_path, output_file=output_path)
                report["success"].append({
                    "input": str(input_path),
                    "output": str(output_path),
                    "database": str(self.database_path),
                })
            except Exception as exc:
                report["errors"].append({"input": str(input_path), "error": str(exc)})
                print(f"Ошибка обработки {input_path}: {exc}")

        print(
            f"\nГотово. Успешно: {len(report['success'])}; "
            f"ошибок: {len(report['errors'])}"
        )
        return report


def create_ner_factory(
    output_dir,
    confidence_threshold=0.5,
    model_name=r"C:\Users\rfhg3\Desktop\Project\Gliner\models\gliner2.5-multi-v1",
    device=None,
    context_sentence_window=1,
):
    return NERFactory(
        output_dir=output_dir,
        confidence_threshold=confidence_threshold,
        model_name=model_name,
        device=device,
        context_sentence_window=context_sentence_window,
    )
