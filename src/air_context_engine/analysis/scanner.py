from __future__ import annotations

import os
from collections.abc import Callable, Generator, Iterable, Mapping
from pathlib import Path
from typing import Any

import pathspec

try:
    import tomllib
except ImportError:  # pragma: no cover - yalnızca Python 3.10
    import tomli as tomllib

from ..core.fs import safe_resolve_path


DEFAULT_CONFIG_FILENAME = ".air-context.toml"
DEFAULT_MAX_FILE_SIZE_BYTES = 1_000_000
DEFAULT_BINARY_SAMPLE_BYTES = 8_192
DEFAULT_DENY_PATTERNS = [
    ".env",
    ".env.*",
    "**/.env",
    "**/.env.*",
    "*.pem",
    "*.key",
    "id_rsa*",
    "id_ed25519*",
    "credentials.json",
    "**/credentials.json",
    "service-account*.json",
    "**/service-account*.json",
    "secrets.*",
    "**/secrets.*",
    ".npmrc",
    ".pypirc",
    ".netrc",
    ".aws/",
    "**/.aws/",
    ".ssh/",
    "**/.ssh/",
]

DEFAULT_ALLOWED_EXTENSIONS = frozenset(
    {
        ".c",
        ".cc",
        ".cjs",
        ".cpp",
        ".cs",
        ".css",
        ".cts",
        ".cxx",
        ".go",
        ".h",
        ".hpp",
        ".html",
        ".ini",
        ".java",
        ".js",
        ".json",
        ".jsx",
        ".md",
        ".php",
        ".mjs",
        ".mts",
        ".ps1",
        ".py",
        ".rb",
        ".rs",
        ".scss",
        ".sh",
        ".sql",
        ".swift",
        ".toml",
        ".ts",
        ".tsx",
        ".txt",
        ".xml",
        ".yaml",
        ".yml",
    }
)

DEFAULT_ALLOWED_FILENAMES = frozenset(
    {
        ".dockerignore",
        ".editorconfig",
        ".gitignore",
        "Dockerfile",
        "Makefile",
        "Pipfile",
        "Procfile",
        "requirements.txt",
    }
)

DEFAULT_IGNORE_PATTERNS = [
    ".git/",
    ".vscode/",
    ".idea/",
    "__pycache__/",
    ".ruff_cache/",
    ".pytest_cache/",
    ".mypy_cache/",
    ".tox/",
    ".nox/",
    "*.pyc",
    "venv/",
    ".venv/",
    "node_modules/",
    "dist/",
    "build/",
    "htmlcov/",
    ".coverage",
    "*.egg-info/",
    "*.png",
    "*.jpg",
    "*.jpeg",
    "*.gif",
    "*.ico",
    "*.pdf",
    "*.zip",
    "*.exe",
    "CLAUDE.md",
    "repo_map.md",
    "dependency_graph.md",
    "BENCHMARK_REPORT.md",
    ".air_cache.json",
    ".air_cache.json.lock",
    ".air-benchmark/",
    ".air_index.sqlite3",
    ".air_index.sqlite3-*",
    ".air_index.sqlite3.lock",
    DEFAULT_CONFIG_FILENAME,
]


class RepositoryScanner:
    """Ignore, tür, boyut, binary ve kullanıcı yapılandırması politikalarını uygular."""

    def __init__(
        self,
        root_path: str | Path,
        *,
        allowed_extensions: Iterable[str] | None = None,
        allowed_filenames: Iterable[str] | None = None,
        max_file_size: int | None = None,
        detect_binary: bool | None = None,
        include_hidden: bool | None = None,
        deny_patterns: Iterable[str] | None = None,
        config_path: str | Path | None = None,
    ) -> None:
        self.root_path = Path(root_path).expanduser().absolute()
        try:
            self.canonical_root = safe_resolve_path(self.root_path, strict=True)
        except OSError as exc:
            raise ValueError(f"Tarama kökü çözümlenemedi: {root_path}: {exc}") from exc
        if not self.canonical_root.is_dir():
            raise ValueError(f"Tarama kökü dizin olmalıdır: {self.root_path}")
        config = self._load_user_config(config_path)

        configured_extensions = self._string_list(config, "allowed_extensions")
        configured_filenames = self._string_list(config, "allowed_filenames")
        configured_ignores = self._string_list(config, "ignore_patterns") or []
        configured_denies = self._string_list(config, "deny_patterns") or []

        extensions = (
            allowed_extensions
            if allowed_extensions is not None
            else (
                configured_extensions
                if configured_extensions is not None
                else DEFAULT_ALLOWED_EXTENSIONS
            )
        )
        filenames = (
            allowed_filenames
            if allowed_filenames is not None
            else (
                configured_filenames
                if configured_filenames is not None
                else DEFAULT_ALLOWED_FILENAMES
            )
        )
        configured_size = config.get("max_file_size", DEFAULT_MAX_FILE_SIZE_BYTES)
        effective_size = configured_size if max_file_size is None else max_file_size
        if not isinstance(effective_size, int) or isinstance(effective_size, bool):
            raise ValueError("scanner.max_file_size tam sayı olmalıdır")
        if effective_size < 0:
            raise ValueError("max_file_size negatif olamaz")

        configured_binary = config.get("detect_binary", True)
        effective_binary = configured_binary if detect_binary is None else detect_binary
        if not isinstance(effective_binary, bool):
            raise ValueError("scanner.detect_binary boolean olmalıdır")

        configured_hidden = config.get("include_hidden", False)
        effective_hidden = (
            configured_hidden if include_hidden is None else include_hidden
        )
        if not isinstance(effective_hidden, bool):
            raise ValueError("scanner.include_hidden boolean olmalıdır")
        effective_denies = (
            list(deny_patterns) if deny_patterns is not None else configured_denies
        )
        if any(
            not pattern.strip() or pattern.lstrip().startswith("!")
            for pattern in effective_denies
        ):
            raise ValueError("scanner.deny_patterns boş veya negation kalıbı içeremez")

        self.allowed_extensions = frozenset(
            self._normalize_extension(extension) for extension in extensions
        )
        self.allowed_filenames = frozenset(filenames)
        self.max_file_size = effective_size
        self.detect_binary = effective_binary
        self.include_hidden = effective_hidden
        self.ignore_spec = self._load_ignore_patterns(configured_ignores)
        self.deny_patterns = tuple([*DEFAULT_DENY_PATTERNS, *effective_denies])
        self.deny_spec = pathspec.PathSpec.from_lines(
            "gitignore",
            (pattern.casefold() for pattern in self.deny_patterns),
        )

    def _load_user_config(self, config_path: str | Path | None) -> Mapping[str, Any]:
        explicit = config_path is not None
        path = (
            Path(config_path).absolute()
            if config_path is not None
            else self.root_path / DEFAULT_CONFIG_FILENAME
        )
        if not path.exists():
            if explicit:
                raise ValueError(f"Yapılandırma dosyası bulunamadı: {path}")
            return {}
        try:
            with path.open("rb") as config_handle:
                document = tomllib.load(config_handle)
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise ValueError(f"Yapılandırma dosyası okunamadı: {path}: {exc}") from exc
        scanner_config = document.get("scanner", {})
        if not isinstance(scanner_config, dict):
            raise ValueError("[scanner] yapılandırması bir TOML tablosu olmalıdır")
        return scanner_config

    @staticmethod
    def _string_list(config: Mapping[str, Any], key: str) -> list[str] | None:
        value = config.get(key)
        if value is None:
            return None
        if not isinstance(value, list) or not all(
            isinstance(item, str) for item in value
        ):
            raise ValueError(f"scanner.{key} metin listesi olmalıdır")
        return value

    @staticmethod
    def _normalize_extension(extension: str) -> str:
        normalized = extension.lower()
        return normalized if normalized.startswith(".") else f".{normalized}"

    def _load_ignore_patterns(
        self, configured_patterns: Iterable[str]
    ) -> pathspec.PathSpec:
        """`.gitignore`, varsayılanlar ve kullanıcı kalıplarını birleştirir."""
        patterns = [*DEFAULT_IGNORE_PATTERNS, *configured_patterns]
        gitignore_path = self.root_path / ".gitignore"
        if gitignore_path.exists():
            try:
                with gitignore_path.open("r", encoding="utf-8") as gitignore_handle:
                    patterns.extend(gitignore_handle.readlines())
            except OSError:
                pass
        return pathspec.PathSpec.from_lines("gitignore", patterns)

    def is_ignored(self, file_path: Path) -> bool:
        """Verilen yolun ignore kurallarına uyup uymadığını kontrol eder."""
        try:
            lexical = file_path.absolute()
            resolved = safe_resolve_path(
                file_path,
                strict=False,
                allow_fallback=False,
            )
            lexical_relative = lexical.relative_to(self.root_path)
            resolved_relative = resolved.relative_to(self.canonical_root)
            rel_str = lexical_relative.as_posix()
            resolved_rel_str = resolved_relative.as_posix()
            if not rel_str:
                return False
            if not self.include_hidden and any(
                part.startswith(".") for part in lexical_relative.parts
            ):
                return True
            if self.deny_spec.match_file(
                rel_str.casefold()
            ) or self.deny_spec.match_file(resolved_rel_str.casefold()):
                return True
            if file_path.is_dir() and not rel_str.endswith("/"):
                rel_str += "/"
            return self.ignore_spec.match_file(rel_str)
        except (OSError, ValueError):
            return True

    @staticmethod
    def is_binary_file(file_path: Path) -> bool:
        """NUL ve kontrol karakteri yoğunluğuyla binary içeriği saptar."""
        try:
            with file_path.open("rb") as binary_handle:
                sample = binary_handle.read(DEFAULT_BINARY_SAMPLE_BYTES)
        except OSError:
            return True
        if not sample:
            return False
        if b"\x00" in sample:
            return True
        text_controls = {7, 8, 9, 10, 12, 13, 27}
        suspicious = sum(byte < 32 and byte not in text_controls for byte in sample)
        return suspicious / len(sample) > 0.05

    def is_allowed_file(self, file_path: Path) -> bool:
        """Dosya adı/uzantısı, boyut ve binary politikasını uygular."""
        try:
            if not file_path.is_file() or file_path.stat().st_size > self.max_file_size:
                return False
        except OSError:
            return False
        allowed_type = (
            file_path.name in self.allowed_filenames
            or file_path.suffix.lower() in self.allowed_extensions
        )
        return allowed_type and (
            not self.detect_binary or not self.is_binary_file(file_path)
        )

    def scan_files(
        self,
        checkpoint: Callable[[], None] | None = None,
    ) -> Generator[Path, None, None]:
        """Politikaya uyan dosyaları deterministik sırada üretir."""
        for root, dirs, files in os.walk(self.root_path):
            if checkpoint:
                checkpoint()
            current_dir = Path(root).absolute()
            dirs[:] = sorted(
                directory
                for directory in dirs
                if not self.is_ignored(current_dir / directory)
            )
            for filename in sorted(files):
                if checkpoint:
                    checkpoint()
                file_path = current_dir / filename
                if not self.is_ignored(file_path) and self.is_allowed_file(file_path):
                    yield file_path
