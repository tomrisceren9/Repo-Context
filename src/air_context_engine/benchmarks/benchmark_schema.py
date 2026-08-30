"""Benchmark question-set schema and validation helpers."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, NoReturn

from jsonschema import Draft202012Validator


QUESTION_SCHEMA_VERSION = "1.1.0"
QUESTION_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://air-context-engine.dev/schemas/benchmark-questions-v1.json",
    "title": "Air Context Engine benchmark questions",
    "description": "A versioned set of questions used by the LLM A/B benchmark.",
    "type": "array",
    "minItems": 1,
    "items": {
        "type": "object",
        "additionalProperties": False,
        "required": ["id", "category", "question", "reference_answer"],
        "properties": {
            "id": {
                "oneOf": [
                    {"type": "integer", "minimum": 1},
                    {"type": "string", "minLength": 1, "pattern": "\\S"},
                ]
            },
            "category": {"type": "string", "enum": ["NRA", "AAS", "ZHR"]},
            "question": {"type": "string", "minLength": 1, "pattern": "\\S"},
            "reference_answer": {"type": "string", "minLength": 1, "pattern": "\\S"},
            "expected_keywords": {
                "type": "array",
                "items": {"type": "string", "minLength": 1, "pattern": "\\S"},
                "uniqueItems": True,
                "default": [],
            },
            "forbidden_keywords": {
                "type": "array",
                "items": {"type": "string", "minLength": 1, "pattern": "\\S"},
                "uniqueItems": True,
                "default": [],
            },
            "expected_symbols": {
                "type": "array",
                "items": {"type": "string", "minLength": 1, "pattern": "\\S"},
                "uniqueItems": True,
                "default": [],
            },
            "symbol_aliases": {
                "type": "object",
                "propertyNames": {"minLength": 1, "pattern": "\\S"},
                "additionalProperties": {
                    "type": "array",
                    "minItems": 1,
                    "items": {"type": "string", "minLength": 1, "pattern": "\\S"},
                    "uniqueItems": True,
                },
                "default": {},
            },
            "match_mode": {"type": "string", "enum": ["any", "all"], "default": "any"},
            "symbol_match_mode": {
                "type": "string",
                "enum": ["any", "all"],
                "default": "all",
            },
            "expected_regex": {"type": "string", "minLength": 1},
            "semantic_threshold": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "default": 0.75,
            },
        },
    },
}


def _format_path(path: list[Any]) -> str:
    if not path:
        return "$"
    return "$" + "".join(
        f"[{item}]" if isinstance(item, int) else f".{item}" for item in path
    )


def validate_questions(questions: Any) -> list[dict[str, Any]]:
    """Validate a decoded question set and return it with a narrow type."""
    validator = Draft202012Validator(QUESTION_SCHEMA)
    errors = sorted(
        validator.iter_errors(questions), key=lambda error: list(error.absolute_path)
    )
    if errors:
        first = errors[0]
        raise ValueError(f"{_format_path(list(first.absolute_path))}: {first.message}")

    validated = list(questions)
    seen_ids: set[int | str] = set()
    for index, question in enumerate(validated):
        question_id = question["id"]
        if question_id in seen_ids:
            raise ValueError(f"$[{index}].id: tekrarlanan soru id'si: {question_id}")
        seen_ids.add(question_id)
        expected_symbols = set(question.get("expected_symbols", []))
        unknown_alias_targets = (
            set(question.get("symbol_aliases", {})) - expected_symbols
        )
        if unknown_alias_targets:
            unknown = sorted(unknown_alias_targets)[0]
            raise ValueError(
                f"$[{index}].symbol_aliases.{unknown}: anahtar expected_symbols içinde olmalıdır."
            )
        normalized_expected = {
            " ".join(value.casefold().split())
            for value in question.get("expected_keywords", [])
        }
        normalized_forbidden = {
            " ".join(value.casefold().split())
            for value in question.get("forbidden_keywords", [])
        }
        contradictions = normalized_expected & normalized_forbidden
        if contradictions:
            keyword = sorted(contradictions)[0]
            raise ValueError(
                f"$[{index}]: aynı ifade hem expected_keywords hem "
                f"forbidden_keywords içinde: {keyword}"
            )
        expected_regex = question.get("expected_regex")
        if expected_regex:
            try:
                re.compile(expected_regex)
            except re.error as exc:
                raise ValueError(
                    f"$[{index}].expected_regex: geçersiz regex: {exc}"
                ) from exc
    return validated


def load_questions(path: Path) -> list[dict[str, Any]]:
    """Load and validate a UTF-8 JSON question file."""
    try:
        content = path.read_text(encoding="utf-8")

        def reject_non_json_constant(value: str) -> NoReturn:
            raise ValueError(f"Standart dışı JSON sayısı: {value}")

        decoded = json.loads(content, parse_constant=reject_non_json_constant)
    except UnicodeError as exc:
        raise ValueError("Soru dosyası UTF-8 olmalıdır.") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Geçersiz JSON: satır {exc.lineno}, sütun {exc.colno}: {exc.msg}"
        ) from exc
    return validate_questions(decoded)


def question_set_sha256(questions: list[dict[str, Any]]) -> str:
    """Hash semantic JSON content independently from whitespace and key ordering."""
    canonical = json.dumps(
        questions,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()
