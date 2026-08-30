from __future__ import annotations

import ast
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ..analysis.scanner import RepositoryScanner
from ..analysis.tree_sitter_support import extract_symbol_ranges, language_for_extension
from .index_control import IndexControl, ProgressCallback
from .search_index import (
    IndexStats,
    SQLiteSearchIndex,
    chunker_language,
    interpret_query,
    split_identifier,
    tokenize_code,
)
from .semantic_search import EmbeddingProvider, LocalSemanticIndex


class ASTAwareCodeChunker:
    """Kod dosyalarını sembol sınırlarında metadata'lı parçalara böler."""

    def __init__(self, fallback_lines: int = 35) -> None:
        if fallback_lines <= 0:
            raise ValueError("fallback_lines pozitif olmalıdır")
        self.fallback_lines = fallback_lines

    @staticmethod
    def _chunk(
        rel_path: str,
        language: str,
        symbol_name: str,
        symbol_type: str,
        start_line: int,
        end_line: int,
        lines: list[str],
    ) -> dict[str, Any]:
        content = "\n".join(lines[start_line - 1 : end_line])
        metadata = (
            f"--- [METADATA] Dosya: {rel_path} | Dil: {language} | "
            f"Tür: {symbol_type} | Sembol: {symbol_name or '-'} | "
            f"Satır: {start_line}-{end_line} ---\n"
        )
        return {
            "file_path": rel_path,
            "language": language,
            "symbol_name": symbol_name,
            "symbol_type": symbol_type,
            "start_line": start_line,
            "end_line": end_line,
            "metadata_header": metadata,
            "content": content,
        }

    def _chunk_python_with_ast(
        self,
        code_text: str,
        rel_path: str,
        lines: list[str],
    ) -> list[dict[str, Any]]:
        try:
            tree = ast.parse(code_text)
        except (SyntaxError, ValueError):
            return []
        chunks: list[dict[str, Any]] = []

        def visit(body: list[ast.stmt], scope: tuple[str, ...] = ()) -> None:
            for node in body:
                if isinstance(node, ast.ClassDef):
                    name = ".".join((*scope, node.name))
                    chunks.append(
                        self._chunk(
                            rel_path,
                            "python",
                            name,
                            "class",
                            node.lineno,
                            node.end_lineno or node.lineno,
                            lines,
                        )
                    )
                    visit(node.body, (*scope, node.name))
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    name = ".".join((*scope, node.name))
                    chunks.append(
                        self._chunk(
                            rel_path,
                            "python",
                            name,
                            "method" if scope else "function",
                            node.lineno,
                            node.end_lineno or node.lineno,
                            lines,
                        )
                    )
                    visit(node.body, (*scope, node.name))

        visit(tree.body)
        return chunks

    def chunk_file(self, file_path: Path, root_path: Path) -> list[dict[str, Any]]:
        try:
            code_text = file_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return []
        lines = code_text.splitlines()
        if not lines:
            return []

        rel_path = file_path.relative_to(root_path).as_posix()
        extension = file_path.suffix.lower()
        language = chunker_language(extension)
        chunks: list[dict[str, Any]] = []
        if extension == ".py":
            chunks = self._chunk_python_with_ast(code_text, rel_path, lines)
        elif (tree_language := language_for_extension(extension)) is not None:
            chunks = [
                self._chunk(
                    rel_path,
                    tree_language,
                    symbol.name,
                    symbol.symbol_type,
                    symbol.start_line,
                    symbol.end_line,
                    lines,
                )
                for symbol in extract_symbol_ranges(tree_language, code_text)
            ]

        if not chunks:
            start = 0
            while start < len(lines):
                end = min(start + self.fallback_lines, len(lines))
                chunks.append(
                    self._chunk(rel_path, language, "", "text", start + 1, end, lines)
                )
                if end == len(lines):
                    break
                start += max(1, self.fallback_lines - 10)
        for ordinal, chunk in enumerate(chunks):
            chunk["ordinal"] = ordinal
        return chunks


class RAGEngine:
    """Lexical aramayı koruyan, opsiyonel local semantic hibrit motor."""

    def __init__(
        self,
        root_path: Path,
        scanner: RepositoryScanner,
        *,
        sync_index: bool = True,
        control: IndexControl | None = None,
        progress: ProgressCallback | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        search_mode: str | None = None,
    ) -> None:
        resolved_mode = search_mode or ("hybrid" if embedding_provider else "lexical")
        if resolved_mode not in {"lexical", "semantic", "hybrid"}:
            raise ValueError("search_mode lexical, semantic veya hybrid olmalıdır")
        if resolved_mode in {"semantic", "hybrid"} and embedding_provider is None:
            raise ValueError(
                "Semantic/hybrid arama için açıkça yerel bir embedding modeli gereklidir."
            )
        self.root_path = root_path.absolute()
        self.scanner = scanner
        self.chunker = ASTAwareCodeChunker()
        self.index = SQLiteSearchIndex(self.root_path)
        self.index_stats: IndexStats = (
            self.index.sync(scanner, self.chunker, control=control, progress=progress)
            if sync_index
            else self.index.current_stats()
        )
        self.chunks: list[dict[str, Any]] = []
        self.last_query_interpretation: dict[str, Any] | None = None
        self.search_mode = resolved_mode
        self.semantic_index = (
            LocalSemanticIndex(self.index, embedding_provider)
            if embedding_provider is not None
            else None
        )

    @staticmethod
    def _exact_score(query: str, symbol_name: str) -> float:
        normalized = query.strip().casefold().replace("::", ".")
        symbol = symbol_name.casefold()
        query_leaf = normalized.rsplit(".", 1)[-1]
        leaf = symbol.rsplit(".", 1)[-1]
        query_canonical = "".join(split_identifier(query_leaf))
        symbol_canonical = "".join(split_identifier(leaf))
        return (
            1.0
            if normalized
            and (
                normalized == symbol
                or query_leaf == leaf
                or query_canonical == symbol_canonical
            )
            else 0.0
        )

    @staticmethod
    def _source_role(file_path: str) -> str:
        path = Path(file_path)
        lowered_parts = {part.casefold() for part in path.parts}
        stem = path.stem.casefold()
        if (
            "tests" in lowered_parts
            or "test" in lowered_parts
            or stem.startswith("test_")
            or stem.endswith("_test")
        ):
            return "test"
        if (
            path.suffix.casefold() in {".md", ".mdx", ".rst", ".txt"}
            or "docs" in lowered_parts
            or "documentation" in lowered_parts
            or "architectural_docs" in lowered_parts
        ):
            return "documentation"
        return "production"

    @staticmethod
    def _candidate_similarity(left: dict[str, Any], right: dict[str, Any]) -> float:
        same_file = left["file_path"] == right["file_path"]
        left_symbol = str(left["symbol_name"]).rsplit(".", 1)[-1].casefold()
        right_symbol = str(right["symbol_name"]).rsplit(".", 1)[-1].casefold()
        similarity = 0.25 if left_symbol and left_symbol == right_symbol else 0.0
        if same_file:
            intersection = max(
                0,
                min(int(left["end_line"]), int(right["end_line"]))
                - max(int(left["start_line"]), int(right["start_line"]))
                + 1,
            )
            minimum_span = min(
                int(left["end_line"]) - int(left["start_line"]) + 1,
                int(right["end_line"]) - int(right["start_line"]) + 1,
            )
            overlap = intersection / minimum_span if minimum_span else 0.0
            if intersection:
                similarity = max(similarity, 0.15 + 0.85 * overlap)
        return min(similarity, 1.0)

    def _diversify(
        self,
        candidates: list[dict[str, Any]],
        top_k: int,
        strength: float,
    ) -> list[dict[str, Any]]:
        selected: list[dict[str, Any]] = []
        remaining = candidates[:]
        while remaining and len(selected) < top_k:
            best = max(
                remaining,
                key=lambda candidate: (
                    float(candidate["score"])
                    - strength
                    * max(
                        (
                            self._candidate_similarity(candidate, item)
                            for item in selected
                        ),
                        default=0.0,
                    ),
                    float(candidate["score"]),
                    str(candidate["file_path"]),
                    -int(candidate["start_line"]),
                ),
            )
            selected.append(best)
            remaining.remove(best)
        return selected

    def _lexical_search(
        self,
        query: str,
        top_k: int = 3,
        *,
        file_paths: Iterable[str] | None = None,
        languages: Iterable[str] | None = None,
        symbol_types: Iterable[str] | None = None,
        start_line: int | None = None,
        end_line: int | None = None,
        diversity_strength: float = 0.25,
    ) -> list[dict[str, Any]]:
        if top_k <= 0 or not query.strip():
            return []
        if start_line is not None and start_line <= 0:
            raise ValueError("start_line pozitif olmalıdır")
        if end_line is not None and end_line <= 0:
            raise ValueError("end_line pozitif olmalıdır")
        if start_line is not None and end_line is not None and start_line > end_line:
            raise ValueError("start_line end_line değerinden büyük olamaz")

        interpretation = interpret_query(query)
        self.last_query_interpretation = interpretation.to_dict()
        rows, term_rows, document_count, average_length = self.index.search_rows(
            query,
            query_terms=interpretation.term_weights,
            file_paths=file_paths,
            languages=languages,
            symbol_types=symbol_types,
            start_line=start_line,
            end_line=end_line,
        )
        if not rows or document_count == 0:
            return []

        query_counts = interpretation.term_weights
        document_frequencies = self.index.document_frequencies(query_counts)
        query_weights = {
            term: semantic_weight
            * (
                math.log((document_count + 1) / (document_frequencies.get(term, 0) + 1))
                + 1.0
            )
            for term, semantic_weight in query_counts.items()
        }
        query_norm = (
            math.sqrt(sum(weight * weight for weight in query_weights.values())) or 1.0
        )
        raw: list[dict[str, Any]] = []
        max_bm25 = 0.0
        max_metadata_match = 0.0
        for row in rows:
            matches = term_rows.get(str(row["id"]), {})
            bm25 = 0.0
            dot_product = 0.0
            for term, query_frequency in query_counts.items():
                term_frequency, tfidf_weight = matches.get(term, (0, 0.0))
                if term_frequency:
                    document_frequency = document_frequencies.get(term, 0)
                    idf = math.log(
                        1.0
                        + (document_count - document_frequency + 0.5)
                        / (document_frequency + 0.5)
                    )
                    denominator = term_frequency + 1.5 * (
                        1.0
                        - 0.75
                        + 0.75 * int(row["doc_length"]) / (average_length or 1.0)
                    )
                    bm25 += query_frequency * idf * term_frequency * 2.5 / denominator
                    dot_product += tfidf_weight * query_weights[term]
            tfidf = dot_product / (float(row["vector_norm"]) * query_norm or 1.0)
            exact = self._exact_score(query, str(row["symbol_name"]))
            metadata_terms = set(
                tokenize_code(
                    " ".join(
                        (
                            str(row["file_path"]),
                            str(row["symbol_name"]),
                            str(row["symbol_type"]),
                        )
                    )
                )
            )
            metadata_match = sum(
                query_weights[term]
                for term in metadata_terms.intersection(query_weights)
            )
            callable_score = (
                1.0
                if interpretation.callable_intent
                and str(row["symbol_type"]) in {"function", "method"}
                else 0.0
            )
            source_role = self._source_role(str(row["file_path"]))
            production_symbol_score = (
                1.0
                if interpretation.callable_intent
                and source_role == "production"
                and str(row["symbol_type"])
                in {"class", "function", "method", "interface", "struct"}
                else 0.0
            )
            path_parts = set(tokenize_code(str(row["file_path"])))
            scope_score = (
                1.0
                if interpretation.scope_hints
                and any(hint in path_parts for hint in interpretation.scope_hints)
                else 0.0
            )
            row.update(
                {
                    "exact_score": exact,
                    "bm25_raw": bm25,
                    "tfidf_score": tfidf,
                    "metadata_match_raw": metadata_match,
                    "callable_score": callable_score,
                    "source_role": source_role,
                    "production_symbol_score": production_symbol_score,
                    "scope_score": scope_score,
                }
            )
            max_bm25 = max(max_bm25, bm25)
            max_metadata_match = max(max_metadata_match, metadata_match)
            raw.append(row)

        candidates: list[dict[str, Any]] = []
        for row in raw:
            normalized_bm25 = float(row.pop("bm25_raw")) / max_bm25 if max_bm25 else 0.0
            metadata_match = float(row.pop("metadata_match_raw"))
            normalized_metadata = (
                metadata_match / max_metadata_match if max_metadata_match else 0.0
            )
            score = (
                0.45 * float(row["exact_score"])
                + 0.25 * normalized_bm25
                + 0.15 * float(row["tfidf_score"])
                + 0.15 * normalized_metadata
                + 0.05 * float(row["callable_score"])
                + 0.25 * float(row["production_symbol_score"])
                + 0.30 * float(row["scope_score"])
            )
            if score <= 0:
                continue
            row["bm25_score"] = round(normalized_bm25, 6)
            row["score"] = round(score, 6)
            row["exact_score"] = round(float(row["exact_score"]), 6)
            row["tfidf_score"] = round(float(row["tfidf_score"]), 6)
            row["metadata_score"] = round(normalized_metadata, 6)
            candidates.append(row)
        candidates.sort(
            key=lambda item: (
                -float(item["score"]),
                str(item["file_path"]),
                int(item["start_line"]),
            )
        )
        selected = self._diversify(
            candidates[: max(50, top_k * 10)], top_k, diversity_strength
        )
        self.chunks = selected
        return selected

    @staticmethod
    def _hybrid_fusion(
        lexical: list[dict[str, Any]],
        semantic: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Fuse independent rankings with weighted reciprocal-rank fusion."""
        by_id: dict[str, dict[str, Any]] = {}
        fusion: dict[str, float] = {}
        for weight, results in ((0.65, lexical), (0.35, semantic)):
            for rank, result in enumerate(results, 1):
                chunk_id = str(result["id"])
                by_id.setdefault(chunk_id, {}).update(result)
                fusion[chunk_id] = fusion.get(chunk_id, 0.0) + weight / (60 + rank)
        maximum = max(fusion.values(), default=1.0)
        lexical_scores = {
            str(result["id"]): float(result["score"]) for result in lexical
        }
        merged: list[dict[str, Any]] = []
        for chunk_id, result in by_id.items():
            result["lexical_score"] = lexical_scores.get(chunk_id, 0.0)
            result.setdefault("semantic_score", 0.0)
            result["fusion_score"] = round(fusion[chunk_id] / maximum, 6)
            result["score"] = result["fusion_score"]
            merged.append(result)
        merged.sort(
            key=lambda item: (
                -float(item["fusion_score"]),
                str(item["file_path"]),
                int(item["start_line"]),
            )
        )
        return merged[:top_k]

    def search(
        self,
        query: str,
        top_k: int = 3,
        *,
        file_paths: Iterable[str] | None = None,
        languages: Iterable[str] | None = None,
        symbol_types: Iterable[str] | None = None,
        start_line: int | None = None,
        end_line: int | None = None,
        diversity_strength: float = 0.25,
        search_mode: str | None = None,
    ) -> list[dict[str, Any]]:
        mode = search_mode or self.search_mode
        if mode not in {"lexical", "semantic", "hybrid"}:
            raise ValueError("search_mode lexical, semantic veya hybrid olmalıdır")
        if mode in {"semantic", "hybrid"} and self.semantic_index is None:
            raise ValueError(
                "Semantic/hybrid arama için açıkça yerel bir embedding modeli gereklidir."
            )
        candidate_count = max(top_k * 5, 20) if mode == "hybrid" else top_k
        lexical = (
            self._lexical_search(
                query,
                top_k=candidate_count,
                file_paths=file_paths,
                languages=languages,
                symbol_types=symbol_types,
                start_line=start_line,
                end_line=end_line,
                diversity_strength=diversity_strength,
            )
            if mode in {"lexical", "hybrid"}
            else []
        )
        semantic = (
            self.semantic_index.search(
                query,
                top_k=candidate_count,
                file_paths=file_paths,
                languages=languages,
                symbol_types=symbol_types,
                start_line=start_line,
                end_line=end_line,
            )
            if mode in {"semantic", "hybrid"} and self.semantic_index is not None
            else []
        )
        if mode == "lexical":
            results = lexical[:top_k]
        elif mode == "semantic":
            results = self._diversify(semantic, top_k, diversity_strength)
        else:
            results = self._diversify(
                self._hybrid_fusion(lexical, semantic, max(top_k * 3, top_k)),
                top_k,
                diversity_strength,
            )
        self.chunks = results
        if self.last_query_interpretation is None:
            self.last_query_interpretation = interpret_query(query).to_dict()
        self.last_query_interpretation["search_mode"] = mode
        self.last_query_interpretation["semantic_model"] = (
            self.semantic_index.provider.model_id if self.semantic_index else None
        )
        return results
