"""Optional, fully local embedding support for semantic code retrieval."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Protocol, Sequence

from .search_index import SQLiteSearchIndex


class EmbeddingProvider(Protocol):
    """Provider-independent contract for offline embedding implementations."""

    @property
    def model_id(self) -> str: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def _normalize(vector: Sequence[float]) -> list[float]:
    values = [float(value) for value in vector]
    norm = math.sqrt(sum(value * value for value in values))
    if not values or norm == 0:
        raise ValueError("Embedding vektörü boş veya sıfır normlu olamaz.")
    return [value / norm for value in values]


class SentenceTransformerEmbeddingProvider:
    """Load a sentence-transformers model strictly from a local filesystem path."""

    def __init__(self, model_path: Path, *, batch_size: int = 32) -> None:
        resolved = model_path.expanduser().absolute()
        if not resolved.exists():
            raise ValueError(
                "Semantic model yerel dosya sisteminde bulunamadı: "
                f"{resolved}. Otomatik model indirme ve cloud fallback kapalıdır."
            )
        if batch_size < 1:
            raise ValueError("Embedding batch_size en az 1 olmalıdır.")
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Yerel semantic search için semantic extra grubunu kurun: "
                "pip install 'air-context-engine[semantic]'"
            ) from exc
        try:
            self._model = SentenceTransformer(
                str(resolved),
                local_files_only=True,
            )
        except TypeError as exc:
            raise RuntimeError(
                "Kurulu sentence-transformers sürümü local_files_only desteği "
                "sunmuyor; ağ erişimi riski nedeniyle model yüklenmedi."
            ) from exc
        self._model_path = resolved
        self._batch_size = batch_size

    @property
    def model_id(self) -> str:
        return f"sentence-transformers:{self._model_path}"

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        encoded = self._model.encode(
            list(texts),
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [_normalize(vector) for vector in encoded]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class LocalSemanticIndex:
    """Persist and rank embeddings produced by an explicitly local provider."""

    def __init__(
        self,
        index: SQLiteSearchIndex,
        provider: EmbeddingProvider,
        *,
        batch_size: int = 32,
    ) -> None:
        if batch_size < 1:
            raise ValueError("Semantic indeks batch_size en az 1 olmalıdır.")
        self.index = index
        self.provider = provider
        self.batch_size = batch_size

    @staticmethod
    def _embedding_text(chunk: dict[str, Any]) -> str:
        return "\n".join(
            (
                str(chunk["file_path"]),
                str(chunk["symbol_name"]),
                str(chunk["symbol_type"]),
                str(chunk["metadata_header"]),
                str(chunk["content"]),
            )
        )

    def sync(self) -> dict[str, int]:
        chunks = self.index.all_chunks()
        existing = self.index.semantic_vectors(self.provider.model_id)
        missing = [chunk for chunk in chunks if str(chunk["id"]) not in existing]
        embedded = 0
        for start in range(0, len(missing), self.batch_size):
            batch = missing[start : start + self.batch_size]
            vectors = self.provider.embed_documents(
                [self._embedding_text(chunk) for chunk in batch]
            )
            if len(vectors) != len(batch):
                raise ValueError(
                    "Embedding sağlayıcısı istenen belge sayısıyla uyumsuz sonuç döndürdü."
                )
            normalized = [
                (str(chunk["id"]), _normalize(vector))
                for chunk, vector in zip(batch, vectors, strict=True)
            ]
            self.index.store_semantic_vectors(self.provider.model_id, normalized)
            embedded += len(normalized)
        return {
            "chunks": len(chunks),
            "embedded": embedded,
            "reused": len(chunks) - embedded,
        }

    def search(
        self,
        query: str,
        top_k: int,
        **filters: Any,
    ) -> list[dict[str, Any]]:
        if top_k <= 0 or not query.strip():
            return []
        self.sync()
        query_vector = _normalize(self.provider.embed_query(query))
        vectors = self.index.semantic_vectors(self.provider.model_id)
        candidates = self.index.all_chunks(**filters)
        scored: list[dict[str, Any]] = []
        for chunk in candidates:
            vector = vectors.get(str(chunk["id"]))
            if vector is None or len(vector) != len(query_vector):
                continue
            similarity = sum(
                left * right for left, right in zip(query_vector, vector, strict=True)
            )
            if similarity <= 0:
                continue
            chunk["semantic_score"] = round(similarity, 6)
            chunk["score"] = round(similarity, 6)
            scored.append(chunk)
        scored.sort(
            key=lambda item: (
                -float(item["semantic_score"]),
                str(item["file_path"]),
                int(item["start_line"]),
            )
        )
        return scored[:top_k]
