import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from filelock import FileLock


CACHE_SCHEMA_VERSION = 4


def _empty_cache() -> dict[str, Any]:
    return {
        "version": CACHE_SCHEMA_VERSION,
        "files": {},
        "symbols": {},
        "dependencies": {},
        "rag_index": {"schema_version": 2, "files": {}},
    }


class CacheManager:
    """SHA-256 tabanlı, atomik ve süreçler arası kilitli depo önbelleği."""

    def __init__(self, root_path: Path, cache_filename: str = ".air_cache.json"):
        self.root_path = root_path.absolute()
        self.cache_file = self.root_path / cache_filename
        self.lock_file = self.root_path / f"{cache_filename}.lock"
        self._lock = FileLock(str(self.lock_file), timeout=30)
        self.cache_data: dict[str, Any] = self._load_cache()

    def _load_cache(self) -> dict[str, Any]:
        """Var olan cache dosyasını yükler; bozuk/uyumsuz veride boş cache döndürür."""
        if not self.cache_file.exists():
            return _empty_cache()

        try:
            with open(self.cache_file, "r", encoding="utf-8") as cache_handle:
                data = json.load(cache_handle)
            if (
                not isinstance(data, dict)
                or data.get("version") != CACHE_SCHEMA_VERSION
            ):
                return _empty_cache()
            normalized = _empty_cache()
            for key in ("files", "symbols", "dependencies"):
                value = data.get(key, {})
                normalized[key] = value if isinstance(value, dict) else {}
            rag_index = data.get("rag_index", {})
            if isinstance(rag_index, dict):
                normalized["rag_index"] = rag_index
            return normalized
        except (OSError, json.JSONDecodeError):
            return _empty_cache()

    def _save_cache_atomic(self) -> None:
        """Cache'i aynı dosya sisteminde geçici dosya + os.replace ile atomik yazar."""
        self.root_path.mkdir(parents=True, exist_ok=True)
        temp_path: Path | None = None
        try:
            file_descriptor, temp_name = tempfile.mkstemp(
                prefix=f".{self.cache_file.name}.",
                suffix=".tmp",
                dir=self.root_path,
            )
            temp_path = Path(temp_name)
            with os.fdopen(file_descriptor, "w", encoding="utf-8") as cache_handle:
                json.dump(self.cache_data, cache_handle, indent=2, ensure_ascii=False)
                cache_handle.flush()
                os.fsync(cache_handle.fileno())
            os.replace(temp_path, self.cache_file)
            temp_path = None
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

    @contextmanager
    def transaction(self) -> Iterator["CacheManager"]:
        """Güncel cache'i kilit altında yükler ve başarılı çıkışta atomik kaydeder."""
        with self._lock:
            self.cache_data = self._load_cache()
            yield self
            self._save_cache_atomic()

    def save_cache(self) -> None:
        """Mevcut cache verisini süreç kilidi altında atomik olarak kaydeder."""
        with self._lock:
            self._save_cache_atomic()

    def _calculate_hash(self, file_path: Path) -> str:
        """Dosyanın SHA-256 içerik hash'ini çıkarır."""
        hasher = hashlib.sha256()
        try:
            with open(file_path, "rb") as file_handle:
                while chunk := file_handle.read(8192):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except OSError:
            return ""

    def get_changed_and_deleted_files(
        self,
        current_files: list[Path],
    ) -> tuple[set[Path], set[str]]:
        """Değişen/yeni dosyaları ve silinen bağıntılı yolları döndürür."""
        changed_files: set[Path] = set()
        current_rel_paths: set[str] = set()
        cached_files = self.cache_data.setdefault("files", {})

        for file_path in current_files:
            rel_path = str(file_path.relative_to(self.root_path)).replace("\\", "/")
            current_rel_paths.add(rel_path)
            file_hash = self._calculate_hash(file_path)

            if (
                rel_path not in cached_files
                or cached_files[rel_path].get("hash") != file_hash
            ):
                changed_files.add(file_path)
                cached_files[rel_path] = {
                    "hash": file_hash,
                    "mtime": file_path.stat().st_mtime,
                }

        deleted_rel_paths = set(cached_files) - current_rel_paths
        cached_symbols = self.cache_data.setdefault("symbols", {})
        cached_dependencies = self.cache_data.setdefault("dependencies", {})
        rag_files = self.cache_data.setdefault("rag_index", {}).setdefault("files", {})
        for deleted in deleted_rel_paths:
            del cached_files[deleted]
            cached_symbols.pop(deleted, None)
            cached_dependencies.pop(deleted, None)
            rag_files.pop(deleted, None)

        return changed_files, deleted_rel_paths
