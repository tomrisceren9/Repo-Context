from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import time
import unicodedata
from types import TracebackType
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from filelock import FileLock, Timeout as FileLockTimeout

from ..analysis.cache_manager import CacheManager
from ..analysis.scanner import RepositoryScanner
from .index_control import IndexControl, IndexProgress, ProgressCallback


INDEX_SCHEMA_VERSION = 5
DEFAULT_INDEX_FILENAME = ".air_index.sqlite3"

QUERY_STOPWORDS = frozenset(
    {
        "a",
        "acaba",
        "and",
        "bir",
        "bu",
        "da",
        "de",
        "the",
        "ve",
        "veya",
        "hangi",
        "how",
        "için",
        "icin",
        "ile",
        "ise",
        "kod",
        "kodu",
        "kodunu",
        "kodun",
        "nerede",
        "nasıl",
        "nasil",
        "ne",
        "olan",
        "olarak",
        "olursa",
        "to",
        "when",
    }
)

TURKISH_ASCII_TRANSLATION = str.maketrans(
    {
        "ç": "c",
        "ğ": "g",
        "ı": "i",
        "ö": "o",
        "ş": "s",
        "ü": "u",
    }
)

CALLABLE_QUERY_TERMS = frozenset(
    {
        "akış",
        "akis",
        "akışı",
        "akisi",
        "belirtin",
        "bul",
        "çağrı",
        "cagri",
        "dönen",
        "donen",
        "döndüren",
        "donduren",
        "eden",
        "endpoint",
        "fonksiyon",
        "işlem",
        "islem",
        "ilerler",
        "kontrol",
        "kontrolü",
        "kontrolu",
        "metot",
        "nerede",
        "oluşturan",
        "olusturan",
        "sembol",
        "semboller",
        "sınıf",
        "sinif",
        "yapan",
    }
)
CALLABLE_QUERY_PREFIXES = (
    "akış",
    "akis",
    "çağr",
    "cagr",
    "fonksiyon",
    "işle",
    "isle",
    "metot",
    "oluştur",
    "olustur",
    "sembol",
)

# Dil modeli çağırmadan doğal Türkçe ürün/yazılım ifadelerini kod terimlerine taşır.
# Tetikleyicilerdeki `*` son eki prefix eşleşmesini belirtir.
QUERY_CONCEPTS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (
        ("öner*", "oner*", "tavsi*", "recommend*"),
        ("recommendation", "recommendations", "proposal", "generate"),
    ),
    (("stok*", "envanter*", "inventory", "stock"), ("inventory", "stock", "quantity")),
    (("eşik*", "esik*", "threshold*"), ("threshold", "minimum", "maximum")),
    (("tedarik*", "restock*"), ("restock", "inventory", "stock")),
    (("indirim*", "discount*"), ("discount", "campaign", "pricing")),
    (("kullanıcı*", "kullanici*", "user*"), ("user", "current_user")),
    (("yönetici*", "yonetici*", "manager*", "admin*"), ("manager", "admin", "role")),
    (
        ("yetki*", "izin*", "authoriz*"),
        ("authorization", "permission", "role", "dependency"),
    ),
    (
        ("doğrula*", "dogrula*", "validate*", "verify*"),
        ("validate", "verify", "authentication"),
    ),
    (("giriş*", "giris*", "login*"), ("login", "auth", "authentication")),
    (("parola*", "şifre*", "sifre*", "password*"), ("password", "verify_password")),
    (
        ("belirteç*", "belirtec*", "token*", "jwt*"),
        ("token", "jwt", "access_token", "bearer"),
    ),
    (("arayüz*", "arayuz*", "frontend*"), ("frontend", "api", "client")),
    (("çağrı*", "cagri*", "istek*", "request*"), ("request", "endpoint", "api")),
    (
        ("belge*", "doküman*", "dokuman*", "document*", "pdf*"),
        ("document", "documents", "text"),
    ),
    (("parça*", "parca*", "chunk*"), ("chunk", "chunks", "chunk_text")),
    (("embedding*", "vektör*", "vektor*"), ("embedding", "vector", "embed", "index")),
    (("veritaban*", "database*"), ("database", "db", "repository")),
    (("tekrar*", "aynı*", "ayni*", "duplicate*"), ("duplicate", "recent", "repeat")),
    (("aday*", "candidate*", "seç*", "sec*"), ("candidate", "select", "selection")),
    (("öncelik*", "oncelik*", "priority*"), ("priority", "score")),
    (("güven*", "guven*", "confidence*"), ("confidence", "score")),
    (("ajan*", "agent*", "uzman*"), ("agent", "agents", "output")),
    (("oluştur*", "olustur*", "create*"), ("create", "build", "new")),
    (
        ("kuyruk*", "kuyru*", "kuyrug*", "queue*"),
        ("queue", "enqueue", "dispatch", "submit"),
    ),
    (("kredi*", "credit*"), ("credit", "credits", "charge", "refund")),
    (("kota*", "quota*"), ("quota", "limit", "reservation", "reserve")),
    (("hata*", "başarısız*", "basarisiz*", "error*"), ("error", "fail", "failure")),
    (("sil*", "delete*"), ("delete", "remove", "cleanup")),
    (("görsel*", "gorsel*", "image*"), ("image", "images", "media", "vision")),
    (("paralel*", "parallel*"), ("parallel", "gather", "concurrent")),
    (("işle*", "isle*", "process*"), ("process", "handle", "execute")),
    (("sağlayıcı*", "saglayici*", "provider*"), ("provider", "service", "client")),
    (("durum*", "status*"), ("status", "state")),
    (
        ("birleştir*", "birlestir*", "birleşik*", "birlesik*", "orchestrat*"),
        ("orchestrate", "aggregate", "combine", "result"),
    ),
    (
        ("ara", "arama*", "sorgu*", "ilgili*", "retriev*"),
        ("search", "query", "retriever", "relevance"),
    ),
)


class ClosingConnection(sqlite3.Connection):
    """Context manager çıkışında transaction ile birlikte bağlantıyı da kapatır."""

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        try:
            super().__exit__(exc_type, exc_value, traceback)
        finally:
            self.close()
        return False


class Chunker(Protocol):
    def chunk_file(self, file_path: Path, root_path: Path) -> list[dict[str, Any]]: ...


@dataclass(frozen=True)
class IndexStats:
    elapsed_seconds: float
    scanned_files: int
    parsed_files: int
    reused_files: int
    deleted_files: int
    chunk_count: int
    term_count: int

    @property
    def warm(self) -> bool:
        return self.parsed_files == 0 and self.deleted_files == 0

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "warm": self.warm}


def split_identifier(value: str) -> list[str]:
    """snake_case, camelCase, acronym ve nitelikli adları ayrı terimlere böler."""
    pieces: list[str] = []
    for qualified_part in re.split(r"[./\\:]+", value):
        for snake_part in qualified_part.split("_"):
            pieces.extend(
                match.group(0).casefold()
                for match in re.finditer(
                    r"[A-Z]+(?=[A-Z][a-z]|\d|$)|[A-Z]?[a-z]+|\d+",
                    snake_part,
                )
            )
    return [piece for piece in pieces if piece]


def tokenize_code(text: str) -> list[str]:
    """Kod, yol ve nitelikli sembol metninden arama terimleri üretir."""
    terms: list[str] = []
    pattern = r"[^\W\d]\w*(?:[.:/\\][^\W\d]\w*)*"
    for raw in re.findall(
        pattern, unicodedata.normalize("NFKC", text), flags=re.UNICODE
    ):
        normalized = raw.casefold().replace("\\", "/")
        terms.append(normalized)
        ascii_folded = (
            unicodedata.normalize(
                "NFKD",
                normalized.translate(TURKISH_ASCII_TRANSLATION),
            )
            .encode("ascii", "ignore")
            .decode("ascii")
        )
        if ascii_folded and ascii_folded != normalized:
            terms.append(ascii_folded)
        if raw.isascii():
            terms.extend(split_identifier(raw))
    return terms


@dataclass(frozen=True)
class QueryInterpretation:
    """Açıklanabilir doğal dil sorgu normalizasyonu ve kavram genişletmesi."""

    original_terms: tuple[str, ...]
    retained_terms: tuple[str, ...]
    expanded_terms: tuple[str, ...]
    term_weights: dict[str, float]
    natural_language: bool
    callable_intent: bool
    scope_hints: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _concept_matches(term: str, trigger: str) -> bool:
    return term.startswith(trigger[:-1]) if trigger.endswith("*") else term == trigger


def interpret_query(text: str) -> QueryInterpretation:
    """Doğal dil sorgusunu düşük ağırlıklı kod kavramlarıyla genişlet."""
    original = tuple(dict.fromkeys(tokenize_code(text)))
    retained = tuple(
        term for term in original if term not in QUERY_STOPWORDS and len(term) > 1
    )
    surface_terms = re.findall(
        r"[^\W\d]\w*",
        unicodedata.normalize("NFKC", text),
        flags=re.UNICODE,
    )
    natural_language = len(surface_terms) >= 4
    callable_intent = natural_language and any(
        term in CALLABLE_QUERY_TERMS
        or any(term.startswith(prefix) for prefix in CALLABLE_QUERY_PREFIXES)
        for term in original
    )
    normalized_text = " ".join(original)
    scope_hints: list[str] = []
    if any(
        phrase in normalized_text
        for phrase in ("arayüzde", "arayuzde", "arayüz tarafında", "arayuz tarafinda")
    ):
        scope_hints.append("frontend")
    if any(
        phrase in normalized_text
        for phrase in ("sunucuda", "sunucu tarafında", "sunucu tarafinda")
    ):
        scope_hints.append("backend")
    weights = {term: 1.0 for term in retained}
    expanded: list[str] = []
    if natural_language:
        for triggers, concepts in QUERY_CONCEPTS:
            if not any(
                _concept_matches(term, trigger)
                for term in retained
                for trigger in triggers
            ):
                continue
            for concept in concepts:
                if concept not in weights:
                    weights[concept] = 0.35
                    expanded.append(concept)
    return QueryInterpretation(
        original_terms=original,
        retained_terms=retained,
        expanded_terms=tuple(expanded),
        term_weights=weights,
        natural_language=natural_language,
        callable_intent=callable_intent,
        scope_hints=tuple(scope_hints),
    )


def _file_hash(path: Path, control: IndexControl | None = None) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(65_536):
            if control:
                control.checkpoint()
            digest.update(block)
    return digest.hexdigest()


class SQLiteSearchIndex:
    """Dosya bazlı artımlı, kalıcı sparse kod arama indeksi."""

    def __init__(
        self,
        root_path: Path,
        index_filename: str = DEFAULT_INDEX_FILENAME,
    ) -> None:
        self.root_path = root_path.absolute()
        self.index_path = self.root_path / index_filename
        self.lock = FileLock(f"{self.index_path}.lock", timeout=60)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.index_path,
            timeout=60,
            factory=ClosingConnection,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = NORMAL")
        return connection

    def current_stats(self) -> IndexStats:
        """Return persisted index size without scanning or mutating repository files."""
        if not self.index_path.exists():
            return IndexStats(0.0, 0, 0, 0, 0, 0, 0)
        with self._connect() as connection:
            self._ensure_schema(connection)
            file_count = int(
                connection.execute("SELECT COUNT(*) FROM files").fetchone()[0]
            )
            chunk_count = int(
                connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            )
            term_count = int(
                connection.execute("SELECT COUNT(*) FROM terms").fetchone()[0]
            )
        return IndexStats(0.0, file_count, 0, file_count, 0, chunk_count, term_count)

    @staticmethod
    def _create_schema(connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS files (
                file_path TEXT PRIMARY KEY,
                content_hash TEXT NOT NULL,
                language TEXT NOT NULL,
                size INTEGER NOT NULL,
                mtime_ns INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                file_path TEXT NOT NULL REFERENCES files(file_path) ON DELETE CASCADE,
                language TEXT NOT NULL,
                symbol_name TEXT NOT NULL,
                symbol_type TEXT NOT NULL,
                start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL,
                content TEXT NOT NULL,
                metadata_header TEXT NOT NULL,
                doc_length INTEGER NOT NULL,
                vector_norm REAL NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS terms (
                chunk_id TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
                term TEXT NOT NULL,
                term_frequency INTEGER NOT NULL,
                tfidf_weight REAL NOT NULL DEFAULT 0,
                PRIMARY KEY (chunk_id, term)
            );
            CREATE TABLE IF NOT EXISTS semantic_vectors (
                chunk_id TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
                model_id TEXT NOT NULL,
                dimensions INTEGER NOT NULL,
                vector_json TEXT NOT NULL,
                PRIMARY KEY (chunk_id, model_id)
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_path ON chunks(file_path);
            CREATE INDEX IF NOT EXISTS idx_chunks_language ON chunks(language);
            CREATE INDEX IF NOT EXISTS idx_chunks_symbol_type ON chunks(symbol_type);
            CREATE INDEX IF NOT EXISTS idx_chunks_symbol_name ON chunks(symbol_name);
            CREATE INDEX IF NOT EXISTS idx_terms_term ON terms(term);
            CREATE INDEX IF NOT EXISTS idx_semantic_model
                ON semantic_vectors(model_id);
            """
        )

    def _ensure_schema(self, connection: sqlite3.Connection) -> None:
        self._create_schema(connection)
        row = connection.execute(
            "SELECT value FROM meta WHERE key = 'schema_version'"
        ).fetchone()
        if row is not None and int(row["value"]) != INDEX_SCHEMA_VERSION:
            connection.executescript(
                "DROP TABLE IF EXISTS semantic_vectors; DROP TABLE IF EXISTS terms; "
                "DROP TABLE IF EXISTS chunks; "
                "DROP TABLE IF EXISTS files; DROP TABLE IF EXISTS meta;"
            )
            self._create_schema(connection)
        connection.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('schema_version', ?)",
            (str(INDEX_SCHEMA_VERSION),),
        )

    @staticmethod
    def _chunk_id(chunk: dict[str, Any]) -> str:
        identity = "\0".join(
            (
                str(chunk["file_path"]),
                str(chunk["start_line"]),
                str(chunk["end_line"]),
                str(chunk.get("symbol_name", "")),
                str(chunk.get("symbol_type", "")),
                hashlib.sha256(str(chunk["content"]).encode("utf-8")).hexdigest(),
                str(chunk.get("ordinal", 0)),
            )
        )
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()

    def _insert_chunk(
        self, connection: sqlite3.Connection, chunk: dict[str, Any]
    ) -> int:
        search_text = "\n".join(
            (
                str(chunk["file_path"]),
                str(chunk.get("symbol_name", "")),
                str(chunk.get("symbol_type", "")),
                str(chunk.get("metadata_header", "")),
                str(chunk["content"]),
            )
        )
        frequencies = Counter(tokenize_code(search_text))
        chunk_id = self._chunk_id(chunk)
        connection.execute(
            """
            INSERT INTO chunks(
                id, file_path, language, symbol_name, symbol_type, start_line,
                end_line, content, metadata_header, doc_length
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                chunk_id,
                chunk["file_path"],
                chunk["language"],
                chunk.get("symbol_name", ""),
                chunk.get("symbol_type", "text"),
                chunk["start_line"],
                chunk["end_line"],
                chunk["content"],
                chunk.get("metadata_header", ""),
                sum(frequencies.values()),
            ),
        )
        connection.executemany(
            "INSERT INTO terms(chunk_id, term, term_frequency) VALUES (?, ?, ?)",
            ((chunk_id, term, frequency) for term, frequency in frequencies.items()),
        )
        return len(frequencies)

    @staticmethod
    def _recompute_vectors(
        connection: sqlite3.Connection,
        control: IndexControl | None = None,
        progress: ProgressCallback | None = None,
    ) -> None:
        document_count = int(
            connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        )
        if document_count == 0:
            return
        document_frequencies = {
            row["term"]: int(row["document_frequency"])
            for row in connection.execute(
                "SELECT term, COUNT(*) AS document_frequency FROM terms GROUP BY term"
            )
        }
        norms: defaultdict[str, float] = defaultdict(float)
        updates: list[tuple[float, str, str]] = []
        for index, row in enumerate(
            connection.execute("SELECT chunk_id, term, term_frequency FROM terms"),
            start=1,
        ):
            if control and index % 250 == 0:
                control.checkpoint()
            if progress and index % 1000 == 0:
                progress(
                    IndexProgress(
                        "vectors", index, None, "TF-IDF ağırlıkları hesaplanıyor"
                    )
                )
            tf = 1.0 + math.log(int(row["term_frequency"]))
            idf = (
                math.log((document_count + 1) / (document_frequencies[row["term"]] + 1))
                + 1.0
            )
            weight = tf * idf
            updates.append((weight, row["chunk_id"], row["term"]))
            norms[row["chunk_id"]] += weight * weight
        connection.executemany(
            "UPDATE terms SET tfidf_weight = ? WHERE chunk_id = ? AND term = ?",
            updates,
        )
        connection.executemany(
            "UPDATE chunks SET vector_norm = ? WHERE id = ?",
            ((math.sqrt(value), chunk_id) for chunk_id, value in norms.items()),
        )

    def sync(
        self,
        scanner: RepositoryScanner,
        chunker: Chunker,
        *,
        control: IndexControl | None = None,
        progress: ProgressCallback | None = None,
    ) -> IndexStats:
        started = time.perf_counter()
        effective_control = control or IndexControl()

        def report(phase: str, completed: int, total: int | None, message: str) -> None:
            effective_control.checkpoint()
            if progress:
                progress(IndexProgress(phase, completed, total, message))

        files: list[Path] = []
        for file_path in scanner.scan_files(effective_control.checkpoint):
            files.append(file_path)
            if len(files) % 100 == 0:
                report(
                    "scan",
                    len(files),
                    None,
                    "Dosyalar güvenlik politikasıyla taranıyor",
                )
        report("scan", len(files), len(files), "Dosya taraması tamamlandı")
        current: dict[str, tuple[Path, str, int, int]] = {}
        for index, file_path in enumerate(files, start=1):
            report("metadata", index, len(files), "Dosya metadata'sı hazırlanıyor")
            relative = file_path.relative_to(self.root_path).as_posix()
            language = str(chunker_language(file_path.suffix))
            stat = file_path.stat()
            current[relative] = (
                file_path,
                language,
                stat.st_size,
                stat.st_mtime_ns,
            )

        parsed_files = 0
        deleted_files = 0
        current_hashes: dict[str, str] = {}
        while True:
            effective_control.checkpoint()
            try:
                self.lock.acquire(timeout=0.2)
                break
            except FileLockTimeout:
                report("lock", 0, None, "İndeks kilidi bekleniyor")
        try:
            with self._connect() as connection:
                parsed_files, deleted_files, metadata_changed_count = (
                    self._sync_transaction(
                        connection,
                        current,
                        chunker,
                        effective_control,
                        progress,
                        current_hashes,
                    )
                )
                chunk_count = int(
                    connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
                )
                stored_terms = int(
                    connection.execute("SELECT COUNT(*) FROM terms").fetchone()[0]
                )
        finally:
            self.lock.release()

        if parsed_files or deleted_files or metadata_changed_count:
            cache = CacheManager(self.root_path)
            with cache.transaction():
                cache.cache_data["rag_index"] = {
                    "schema_version": INDEX_SCHEMA_VERSION,
                    "database": self.index_path.name,
                    "files": {
                        relative: {
                            "hash": current_hashes[relative],
                            "language": language,
                            "size": size,
                            "mtime_ns": mtime_ns,
                        }
                        for relative, (_, language, size, mtime_ns) in current.items()
                    },
                }

        if progress:
            progress(
                IndexProgress(
                    "complete", len(files), len(files), "İndeksleme tamamlandı"
                )
            )
        return IndexStats(
            elapsed_seconds=time.perf_counter() - started,
            scanned_files=len(files),
            parsed_files=parsed_files,
            reused_files=len(files) - parsed_files,
            deleted_files=deleted_files,
            chunk_count=chunk_count,
            term_count=stored_terms,
        )

    def _sync_transaction(
        self,
        connection: sqlite3.Connection,
        current: dict[str, tuple[Path, str, int, int]],
        chunker: Chunker,
        control: IndexControl,
        progress: ProgressCallback | None,
        current_hashes: dict[str, str],
    ) -> tuple[int, int, int]:
        def report(phase: str, completed: int, total: int | None, message: str) -> None:
            control.checkpoint()
            if progress:
                progress(IndexProgress(phase, completed, total, message))

        self._ensure_schema(connection)
        existing = {
            str(row["file_path"]): dict(row)
            for row in connection.execute(
                "SELECT file_path, content_hash, language, size, mtime_ns FROM files"
            )
        }
        deleted = set(existing) - set(current)
        changed: set[str] = set()
        metadata_changed: set[str] = set()
        for index, (relative, (file_path, language, size, mtime_ns)) in enumerate(
            current.items(), start=1
        ):
            report("hash", index, len(current), f"Hash doğrulanıyor: {relative}")
            previous = existing.get(relative)
            if (
                previous is not None
                and previous["language"] == language
                and previous["size"] == size
                and previous["mtime_ns"] == mtime_ns
            ):
                current_hashes[relative] = str(previous["content_hash"])
                continue
            content_hash = _file_hash(file_path, control)
            current_hashes[relative] = content_hash
            if (
                previous is None
                or previous["content_hash"] != content_hash
                or previous["language"] != language
            ):
                changed.add(relative)
            else:
                metadata_changed.add(relative)
        for relative in deleted:
            control.checkpoint()
            connection.execute("DELETE FROM files WHERE file_path = ?", (relative,))
        for index, relative in enumerate(sorted(changed), start=1):
            report("parse", index, len(changed), f"Kod parçalanıyor: {relative}")
            file_path, language, size, mtime_ns = current[relative]
            content_hash = current_hashes[relative]
            connection.execute("DELETE FROM files WHERE file_path = ?", (relative,))
            connection.execute(
                """
                INSERT INTO files(
                    file_path, content_hash, language, size, mtime_ns
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (relative, content_hash, language, size, mtime_ns),
            )
            for chunk in chunker.chunk_file(file_path, self.root_path):
                control.checkpoint()
                self._insert_chunk(connection, chunk)
        for relative in metadata_changed:
            control.checkpoint()
            _, _, size, mtime_ns = current[relative]
            connection.execute(
                "UPDATE files SET size = ?, mtime_ns = ? WHERE file_path = ?",
                (size, mtime_ns, relative),
            )
        if changed or deleted:
            self._recompute_vectors(connection, control, progress)
        if changed or deleted or metadata_changed:
            connection.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES('last_sync', ?)",
                (str(time.time()),),
            )
        control.checkpoint()
        return len(changed), len(deleted), len(metadata_changed)

    @staticmethod
    def _filter_sql(
        file_paths: Iterable[str] | None,
        languages: Iterable[str] | None,
        symbol_types: Iterable[str] | None,
        start_line: int | None,
        end_line: int | None,
        table_alias: str = "",
    ) -> tuple[str, list[Any]]:
        clauses: list[str] = []
        parameters: list[Any] = []
        prefix = f"{table_alias}." if table_alias else ""
        if file_paths:
            patterns = list(file_paths)
            clauses.append(
                "(" + " OR ".join(f"{prefix}file_path GLOB ?" for _ in patterns) + ")"
            )
            parameters.extend(patterns)
        for column, values in (("language", languages), ("symbol_type", symbol_types)):
            if values:
                selected = list(values)
                clauses.append(
                    f"{prefix}{column} IN ({','.join('?' for _ in selected)})"
                )
                parameters.extend(selected)
        if start_line is not None:
            clauses.append(f"{prefix}end_line >= ?")
            parameters.append(start_line)
        if end_line is not None:
            clauses.append(f"{prefix}start_line <= ?")
            parameters.append(end_line)
        return (" WHERE " + " AND ".join(clauses) if clauses else "", parameters)

    def search_rows(
        self,
        query: str,
        *,
        query_terms: Iterable[str] | None = None,
        file_paths: Iterable[str] | None = None,
        languages: Iterable[str] | None = None,
        symbol_types: Iterable[str] | None = None,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> tuple[
        list[dict[str, Any]], dict[str, dict[str, tuple[int, float]]], int, float
    ]:
        selected_query_terms = sorted(set(query_terms or tokenize_code(query)))
        if not selected_query_terms:
            return [], {}, 0, 0.0
        where, parameters = self._filter_sql(
            file_paths,
            languages,
            symbol_types,
            start_line,
            end_line,
            table_alias="c",
        )
        term_placeholders = ",".join("?" for _ in selected_query_terms)
        candidate_clause = f"t.term IN ({term_placeholders})"
        where = (
            f" WHERE {candidate_clause} AND {where.removeprefix(' WHERE ')}"
            if where
            else f" WHERE {candidate_clause}"
        )
        with self._connect() as connection:
            self._ensure_schema(connection)
            rows = [
                dict(row)
                for row in connection.execute(
                    "SELECT DISTINCT c.* FROM chunks c "
                    "JOIN terms t ON t.chunk_id = c.id" + where,
                    [*selected_query_terms, *parameters],
                )
            ]
            if not rows:
                return [], {}, 0, 0.0
            scores: dict[str, dict[str, tuple[int, float]]] = {}
            candidate_ids = {str(row["id"]) for row in rows}
            placeholders = ",".join("?" for _ in selected_query_terms)
            for term_row in connection.execute(
                f"SELECT chunk_id, term, term_frequency, tfidf_weight FROM terms "
                f"WHERE term IN ({placeholders})",
                selected_query_terms,
            ):
                chunk_id = str(term_row["chunk_id"])
                if chunk_id in candidate_ids:
                    scores.setdefault(chunk_id, {})[str(term_row["term"])] = (
                        int(term_row["term_frequency"]),
                        float(term_row["tfidf_weight"]),
                    )
            document_count = int(
                connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            )
            average_length = float(
                connection.execute(
                    "SELECT COALESCE(AVG(doc_length), 0) FROM chunks"
                ).fetchone()[0]
            )
        return rows, scores, document_count, average_length

    def document_frequencies(self, terms: Iterable[str]) -> dict[str, int]:
        selected = sorted(set(terms))
        if not selected:
            return {}
        placeholders = ",".join("?" for _ in selected)
        with self._connect() as connection:
            return {
                str(row["term"]): int(row["frequency"])
                for row in connection.execute(
                    f"SELECT term, COUNT(*) AS frequency FROM terms "
                    f"WHERE term IN ({placeholders}) GROUP BY term",
                    selected,
                )
            }

    def symbol_locations(
        self,
        file_path: str,
        symbol_name: str,
    ) -> list[dict[str, Any]]:
        """Bir relevance hedefinin indeks içindeki kesin satır aralıklarını döndürür."""
        with self._connect() as connection:
            self._ensure_schema(connection)
            return [
                dict(row)
                for row in connection.execute(
                    "SELECT file_path, symbol_name, symbol_type, start_line, end_line "
                    "FROM chunks WHERE file_path = ? AND symbol_name = ? "
                    "ORDER BY start_line, end_line",
                    (file_path, symbol_name),
                )
            ]

    def all_chunks(
        self,
        *,
        file_paths: Iterable[str] | None = None,
        languages: Iterable[str] | None = None,
        symbol_types: Iterable[str] | None = None,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return indexed chunks for local semantic ranking and maintenance."""
        where, parameters = self._filter_sql(
            file_paths,
            languages,
            symbol_types,
            start_line,
            end_line,
        )
        with self._connect() as connection:
            self._ensure_schema(connection)
            return [
                dict(row)
                for row in connection.execute(
                    "SELECT * FROM chunks" + where + " ORDER BY file_path, start_line",
                    parameters,
                )
            ]

    def semantic_vectors(self, model_id: str) -> dict[str, list[float]]:
        """Load persisted local embedding vectors for one exact model identity."""
        with self._connect() as connection:
            self._ensure_schema(connection)
            rows = connection.execute(
                "SELECT chunk_id, dimensions, vector_json FROM semantic_vectors "
                "WHERE model_id = ?",
                (model_id,),
            )
            vectors: dict[str, list[float]] = {}
            for row in rows:
                decoded = json.loads(str(row["vector_json"]))
                if not isinstance(decoded, list) or len(decoded) != int(
                    row["dimensions"]
                ):
                    continue
                vectors[str(row["chunk_id"])] = [float(value) for value in decoded]
            return vectors

    def store_semantic_vectors(
        self,
        model_id: str,
        vectors: Iterable[tuple[str, list[float]]],
    ) -> None:
        """Persist normalized local embeddings without changing lexical state."""
        rows = [
            (
                chunk_id,
                model_id,
                len(vector),
                json.dumps(vector, separators=(",", ":")),
            )
            for chunk_id, vector in vectors
        ]
        if not rows:
            return
        with self.lock:
            with self._connect() as connection:
                self._ensure_schema(connection)
                connection.executemany(
                    "INSERT OR REPLACE INTO semantic_vectors"
                    "(chunk_id, model_id, dimensions, vector_json) VALUES (?, ?, ?, ?)",
                    rows,
                )


def chunker_language(extension: str) -> str:
    from ..analysis.tree_sitter_support import language_for_extension

    return language_for_extension(extension) or extension.lower().lstrip(".") or "text"
