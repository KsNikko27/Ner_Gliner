import re
import shutil
from pathlib import Path
from postgres_storage import PostgresStorage
import torch
from gliner2 import AutoExtractor


class NERFactory:
    """
    Извлекает NER-сущности, группирует их по ближайшему человеку и сохраняет
    результаты в PostgreSQL. После обработки исходная папка перемещается в архив.
    """

    def __init__(
        self,
        confidence_threshold=0.5,
        model_name=r"C:\Users\rfhg3\Desktop\Project\Gliner\models\gliner2.5-multi-v1",
        device=None,
        context_sentence_window=1,
    ):
        self.confidence_threshold = confidence_threshold
        self.context_sentence_window = max(0, int(context_sentence_window))
        if not 0 <= confidence_threshold <= 1:
            raise ValueError("confidence_threshold должен быть от 0 до 1")
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
            "andrew", "anna", "anthony", "benjamin", "brandon", "brian",
            "charles", "charlotte", "christopher", "daniel", "david", "elizabeth",
            "emily", "emma", "ethan", "george", "grace", "henry", "isabella", "jack",
            "jacob", "james", "jennifer", "john", "joseph", "julia", "karen",
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


    @staticmethod
    def _canonicalize_name(full_name):
        clean = " ".join(full_name.strip().split())
        if "," in clean:
            left, right = [part.strip() for part in clean.split(",", 1)]
            if left and right:
                return f"{right} {left}"
        return clean

    def _parse_name(self, full_name):
        clean = self._canonicalize_name(full_name)
        parts = clean.split()
        first_name = last_name = patronymic = None
        if len(parts) == 1:
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
            last_name, first_name = parts[0], parts[1]
            if len(parts) > 2:
                patronymic = parts[2]
        else:
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
        for match in re.finditer(r"[.!?]+[\"''”)}\]]*\s+|\n+", text):
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

    def analyze_file(self, input_file, archive_dir=None):
        input_path = Path(input_file).resolve()
        if not input_path.is_file():
            raise FileNotFoundError(f"Файл не найден: {input_path}")

        source_folder = input_path.parent

        # 1. Извлечение сущностей
        text = input_path.read_text(encoding="utf-8")
        entities = self.analyze_text(text)
        grouped_result = self.group_by_person(
            text,
            entities,
            source_file=str(input_path),
        )

        # 2. Поиск HTML-скриншота в той же папке
        html_snapshot = None
        page_title = input_path.stem
        html_candidates = sorted(source_folder.glob("*.html"))
        if html_candidates:
            if len(html_candidates) > 1:
                print(
                    f"[Предупреждение] В папке найдено {len(html_candidates)} HTML-файлов. "
                    f"Используется первый: {html_candidates[0].name}"
                )
            html_snapshot = html_candidates[0].stem
        else:
            print(f"[Предупреждение] HTML-файл не найден в папке: {source_folder}")

        # 3. Сохранение в PostgreSQL
        self.pg_storage.save_grouped_result(
            grouped_result,
            page_title=page_title,
            html_snapshot=html_snapshot,
        )

        print(
            f"Обработан: {input_path.name} | Сущностей: {len(entities)} | "
            f"Страница: {page_title} | Снимок: {html_snapshot}"
        )

        # 4. Перемещение всей родительской папки в архив
        if archive_dir is not None:
            archive_path = Path(archive_dir).resolve()
            archive_path.mkdir(parents=True, exist_ok=True)
            dest = archive_path / source_folder.name
            if dest.exists():
                counter = 1
                while True:
                    candidate = archive_path / f"{source_folder.name}_{counter}"
                    if not candidate.exists():
                        dest = candidate
                        break
                    counter += 1
            shutil.move(str(source_folder), str(dest))
            print(f"Папка перемещена в архив: {dest}")

        return True

    def analyze_directory(self, input_dir, recursive=True, pattern="*.txt", archive_dir=None):
        input_dir = Path(input_dir)
        if not input_dir.is_dir():
            raise NotADirectoryError(f"Папка не найдена: {input_dir}")
        files = input_dir.rglob(pattern) if recursive else input_dir.glob(pattern)
        report = {"success": [], "errors": []}
        for input_path in sorted(files):
            try:
                self.analyze_file(input_path, archive_dir=archive_dir)
                report["success"].append({"input": str(input_path)})
            except Exception as exc:
                report["errors"].append({"input": str(input_path), "error": str(exc)})
                print(f"Ошибка обработки {input_path}: {exc}")
        print(
            f"\nГотово. Успешно: {len(report['success'])}; "
            f"ошибок: {len(report['errors'])}"
        )
        return report


def create_ner_factory(
    confidence_threshold=0.5,
    model_name=r"C:\Users\rfhg3\Desktop\Project\Gliner\models\gliner2.5-multi-v1",
    device=None,
    context_sentence_window=1,
):
    return NERFactory(
        confidence_threshold=confidence_threshold,
        model_name=model_name,
        device=device,
        context_sentence_window=context_sentence_window,
    )