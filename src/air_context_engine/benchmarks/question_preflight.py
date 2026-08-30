"""Project-aware validation for benchmark question anchors."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ..analysis.scanner import RepositoryScanner


_QUALIFIER_SEPARATOR = re.compile(r"(?:\.|::|#)")
_SOURCE_EXTENSIONS = {
    ".c",
    ".cc",
    ".cpp",
    ".cs",
    ".cts",
    ".cxx",
    ".go",
    ".h",
    ".hpp",
    ".java",
    ".js",
    ".jsx",
    ".kt",
    ".kts",
    ".mjs",
    ".mts",
    ".php",
    ".py",
    ".rb",
    ".rs",
    ".svelte",
    ".swift",
    ".ts",
    ".tsx",
    ".vue",
}
_SEMANTIC_STOPWORDS = {
    "a",
    "bir",
    "bu",
    "da",
    "de",
    "ile",
    "mi",
    "mı",
    "mu",
    "mü",
    "the",
    "ve",
    "veya",
}


def _contains_identifier(content: str, identifier: str) -> bool:
    pattern = rf"(?<![A-Za-z0-9_]){re.escape(identifier)}(?![A-Za-z0-9_])"
    return re.search(pattern, content) is not None


def _symbol_exists(source_contents: list[str], symbol: str) -> bool:
    """Return whether all qualified symbol parts coexist in one source file."""
    parts = [part for part in _QUALIFIER_SEPARATOR.split(symbol) if part]
    if not parts:
        return False
    return any(
        all(_contains_identifier(content, part) for part in parts)
        for content in source_contents
    )


def _normalized_text(value: str) -> str:
    return " ".join(value.casefold().split())


def _semantic_terms(value: str) -> set[str]:
    return {
        term
        for term in re.findall(r"[^\W\d_]\w+", value.casefold(), flags=re.UNICODE)
        if len(term) > 2 and term not in _SEMANTIC_STOPWORDS
    }


def validate_questions_for_project(
    root_path: Path,
    questions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Check positive and zero-hallucination symbol anchors against a repository."""
    root_path = root_path.absolute()
    scanner = RepositoryScanner(root_path)
    source_contents: list[str] = []
    for file_path in scanner.scan_files():
        if file_path.suffix.lower() not in _SOURCE_EXTENSIONS:
            continue
        try:
            source_contents.append(
                file_path.read_text(encoding="utf-8", errors="ignore")
            )
        except OSError:
            continue

    issues: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    checked_symbols = 0
    for question in questions:
        category = str(question["category"]).upper()
        aliases = question.get("symbol_aliases", {})
        reference_answer = str(question["reference_answer"])
        normalized_reference = _normalized_text(reference_answer)
        expected_keywords = list(question.get("expected_keywords", []))
        forbidden_keywords = list(question.get("forbidden_keywords", []))
        expected_regex = question.get("expected_regex")
        if category in {"NRA", "AAS"} and not (
            expected_keywords or question.get("expected_symbols") or expected_regex
        ):
            warnings.append(
                {
                    "question_id": question["id"],
                    "problem": "no_deterministic_positive_anchor",
                    "message": (
                        "Judge kapalıyken bu soru boş deterministik beklentiyle "
                        "otomatik geçebilir."
                    ),
                }
            )
        keyword_matches = [
            _normalized_text(keyword) in normalized_reference
            for keyword in expected_keywords
        ]
        if keyword_matches and not (
            all(keyword_matches)
            if question.get("match_mode", "any") == "all"
            else any(keyword_matches)
        ):
            warnings.append(
                {
                    "question_id": question["id"],
                    "problem": "reference_answer_misses_expected_keywords",
                    "message": (
                        "reference_answer, seçilen match_mode için expected_keywords "
                        "beklentisini karşılamıyor."
                    ),
                }
            )
        conflicting_reference = [
            keyword
            for keyword in forbidden_keywords
            if _normalized_text(keyword) in normalized_reference
        ]
        if conflicting_reference:
            issues.append(
                {
                    "question_id": question["id"],
                    "problem": "reference_answer_contains_forbidden_keyword",
                    "keywords": conflicting_reference,
                }
            )
        threshold = float(question.get("semantic_threshold", 0.75))
        if threshold < 0.5 or threshold > 0.95:
            warnings.append(
                {
                    "question_id": question["id"],
                    "problem": "extreme_semantic_threshold",
                    "value": threshold,
                    "message": "Önerilen semantic_threshold aralığı 0.50-0.95'tir.",
                }
            )
        question_terms = _semantic_terms(str(question["question"]))
        reference_terms = _semantic_terms(reference_answer)
        if question_terms and reference_terms and not question_terms & reference_terms:
            warnings.append(
                {
                    "question_id": question["id"],
                    "problem": "question_reference_low_lexical_alignment",
                    "message": (
                        "Soru ile reference_answer ortak anlamlı terim içermiyor; "
                        "anlamsal uyum insan veya local judge ile gözden geçirilmeli."
                    ),
                }
            )
        for expected_symbol in question.get("expected_symbols", []):
            checked_symbols += 1
            candidates = [expected_symbol]
            if isinstance(aliases, dict):
                candidates.extend(aliases.get(expected_symbol, []))
            matched_by = next(
                (
                    candidate
                    for candidate in candidates
                    if _symbol_exists(source_contents, candidate)
                ),
                None,
            )
            if category in {"NRA", "AAS"} and matched_by is None:
                issues.append(
                    {
                        "question_id": question["id"],
                        "category": category,
                        "symbol": expected_symbol,
                        "candidates": candidates,
                        "problem": "expected_symbol_not_found",
                    }
                )
            elif category == "ZHR" and matched_by is not None:
                issues.append(
                    {
                        "question_id": question["id"],
                        "category": category,
                        "symbol": expected_symbol,
                        "matched_by": matched_by,
                        "problem": "zhr_symbol_present_in_project",
                    }
                )
            reference_candidates = [expected_symbol]
            if isinstance(aliases, dict):
                reference_candidates.extend(aliases.get(expected_symbol, []))
            if category in {"NRA", "AAS"} and not any(
                _symbol_exists([reference_answer], candidate)
                for candidate in reference_candidates
            ):
                warnings.append(
                    {
                        "question_id": question["id"],
                        "symbol": expected_symbol,
                        "problem": "reference_answer_misses_expected_symbol",
                        "message": (
                            "Deterministik olarak zorunlu sembol reference_answer "
                            "içinde görünmüyor."
                        ),
                    }
                )

    return {
        "valid": not issues,
        "project_path": str(root_path),
        "source_file_count": len(source_contents),
        "question_count": len(questions),
        "checked_symbol_count": checked_symbols,
        "issues": issues,
        "warning_count": len(warnings),
        "warnings": warnings,
    }
