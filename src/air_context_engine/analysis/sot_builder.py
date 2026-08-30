from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .scanner import RepositoryScanner

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib


class SOTBuilder:
    """Depo envanterini ve doğrulanabilir çalışma ipuçlarını tek belgede toplar."""

    def __init__(self, root_path: Path, scanner: RepositoryScanner) -> None:
        self.root_path = root_path.absolute()
        self.scanner = scanner

    def _relative(self, path: Path) -> str:
        return path.relative_to(self.root_path).as_posix()

    @staticmethod
    def _append_unique(target: list[str], value: str) -> None:
        if value not in target:
            target.append(value)

    def _extract_tech_stack(self, files: list[Path]) -> dict[str, list[str]]:
        """Yalnız manifest ve yapılandırma dosyalarında doğrulanan teknolojileri döndürür."""
        stack: dict[str, list[str]] = {
            "backend": [],
            "frontend": [],
            "infrastructure": [],
        }
        for path in files:
            if path.name == "requirements.txt":
                try:
                    content = path.read_text(
                        encoding="utf-8", errors="ignore"
                    ).casefold()
                except OSError:
                    continue
                detections: tuple[tuple[str, str], ...] = (
                    ("fastapi", "FastAPI"),
                    ("pydantic", "Pydantic"),
                    ("langchain", "LangChain"),
                )
                for token, label in detections:
                    if token in content:
                        self._append_unique(stack["backend"], label)
                if "sqlalchemy" in content or "alembic" in content:
                    self._append_unique(stack["backend"], "SQLAlchemy / Alembic")
                if any(
                    token in content
                    for token in ("google-generativeai", "google-genai", "gemini")
                ):
                    self._append_unique(stack["backend"], "Gemini LLM SDK")
            elif path.name == "package.json":
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                dependencies = {
                    **data.get("dependencies", {}),
                    **data.get("devDependencies", {}),
                }
                detections = (
                    ("next", "Next.js"),
                    ("react", "React"),
                    ("tailwindcss", "Tailwind CSS"),
                    ("typescript", "TypeScript"),
                    ("vue", "Vue"),
                    ("svelte", "Svelte"),
                )
                for package, label in detections:
                    if package in dependencies:
                        self._append_unique(stack["frontend"], label)
            if path.name in {"docker-compose.yml", "docker-compose.yaml", "Dockerfile"}:
                self._append_unique(stack["infrastructure"], "Docker / Docker Compose")
        return stack

    @staticmethod
    def _shell_command(directory: str, command: str) -> str:
        return command if directory == "." else f"cd {directory} && {command}"

    def _package_scripts(self, path: Path) -> dict[str, Any]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        scripts = data.get("scripts", {})
        return scripts if isinstance(scripts, dict) else {}

    @staticmethod
    def _pyproject_scripts(path: Path) -> dict[str, str]:
        try:
            with path.open("rb") as handle:
                data = tomllib.load(handle)
        except (OSError, tomllib.TOMLDecodeError):
            return {}
        scripts = data.get("project", {}).get("scripts", {})
        if not isinstance(scripts, dict):
            return {}
        return {
            str(command): str(target)
            for command, target in scripts.items()
            if isinstance(command, str) and isinstance(target, str)
        }

    def _generate_commands_summary(self, files: list[Path]) -> list[str]:
        """Manifestlerden path-duyarlı kurulum ve çalışma komutları çıkarır."""
        commands: list[str] = []
        for path in sorted(files):
            relative = self._relative(path)
            directory = Path(relative).parent.as_posix()
            if path.name in {"docker-compose.yml", "docker-compose.yaml"}:
                commands.append(
                    f"- **Docker Compose (`{relative}`):** "
                    f"`docker compose -f {relative} up --build`"
                )
            elif path.name == "requirements.txt":
                commands.append(
                    f"- **Python bağımlılıkları (`{relative}`):** "
                    f"`python -m pip install -r {relative}`"
                )
            elif path.name == "package.json":
                commands.append(
                    f"- **Node bağımlılıkları (`{relative}`):** "
                    f"`{self._shell_command(directory, 'npm install')}`"
                )
                for script in ("dev", "start", "build", "test", "lint"):
                    if script in self._package_scripts(path):
                        commands.append(
                            f"- **npm `{script}` (`{relative}`):** "
                            f"`{self._shell_command(directory, f'npm run {script}')}`"
                        )
            elif path.name == "pyproject.toml":
                install = self._shell_command(directory, "python -m pip install -e .")
                commands.append(f"- **Python paketi (`{relative}`):** `{install}`")
        return list(dict.fromkeys(commands))

    def _entry_points(self, files: list[Path]) -> list[str]:
        entry_names = {
            "main.py",
            "app.py",
            "cli.py",
            "manage.py",
            "index.ts",
            "index.js",
        }
        entries = [self._relative(path) for path in files if path.name in entry_names]
        entries.extend(
            self._relative(path)
            for path in files
            if path.name
            in {"docker-compose.yml", "docker-compose.yaml", "package.json"}
        )
        for path in files:
            if path.name != "pyproject.toml":
                continue
            relative = self._relative(path)
            entries.extend(
                f"{relative} → {command} = {target}"
                for command, target in self._pyproject_scripts(path).items()
            )
        return sorted(set(entries))

    def _top_level_summary(self, files: list[Path]) -> list[str]:
        counts: Counter[str] = Counter()
        for path in files:
            parts = path.relative_to(self.root_path).parts
            counts[parts[0] if len(parts) > 1 else "(kök)"] += 1
        return [f"- `{name}/`: {count} dosya" for name, count in sorted(counts.items())]

    def _render_tree(self, files: list[Path]) -> list[str]:
        """Düz yol listesi yerine tekrar etmeyen, girintili klasör ağacı üretir."""
        lines: list[str] = []
        seen_directories: set[tuple[str, ...]] = set()
        for path in sorted(files, key=lambda item: self._relative(item)):
            parts = path.relative_to(self.root_path).parts
            for depth in range(1, len(parts)):
                directory = parts[:depth]
                if directory not in seen_directories:
                    lines.append(f"{'  ' * (depth - 1)}{directory[-1]}/")
                    seen_directories.add(directory)
            lines.append(f"{'  ' * (len(parts) - 1)}{parts[-1]}")
        return lines

    def generate_sot(self) -> str:
        """LLM için proje büyük resmini ve dosya hiyerarşisini üretir."""
        files = list(self.scanner.scan_files())
        tech_stack = self._extract_tech_stack(files)
        commands = self._generate_commands_summary(files)
        extension_counts = Counter(
            path.suffix.lower() if path.suffix else "uzantısız" for path in files
        )

        content = [
            "# PROJECT SINGLE SOURCE OF TRUTH (SOT)\n",
            "## 1. Proje Özeti",
            f"- **Proje Adı / Kök Dizin:** `{self.root_path.name}`",
            f"- **Toplam Taranabilir Dosya:** {len(files)}",
            "- **Belge rolü:** Projenin teknoloji, giriş noktası, komut ve dosya hiyerarşisi görünümü.",
            "- **Tamamlayıcı belgeler:** `repo_map.md` sembol sorumluluklarını; "
            "`dependency_graph.md` modül ilişkilerini gösterir.",
            "- **Kanıt sınırı:** Yalnız depo içeriğinden doğrulanan bilgiler yazılır.",
            "\n## 2. Teknoloji Yığını",
        ]
        labels = {
            "backend": "Backend",
            "frontend": "Frontend",
            "infrastructure": "Altyapı",
        }
        detected_stack = False
        for category, values in tech_stack.items():
            if values:
                detected_stack = True
                content.append(f"- **{labels[category]}:** {', '.join(values)}")
        if not detected_stack:
            content.append("- Manifestlerden teknoloji tespit edilemedi.")

        content.extend(["\n## 3. Doğrulanmış Komutlar"])
        content.extend(commands or ["- Komutlar tespit edilemedi."])
        content.extend(["\n## 4. Olası Giriş Noktaları"])
        content.extend(
            [f"- `{entry}`" for entry in self._entry_points(files)]
            or ["- Standart isimli giriş noktası tespit edilemedi."]
        )
        content.extend(["\n## 5. Üst Düzey Dizinler"])
        content.extend(self._top_level_summary(files))
        content.extend(["\n## 6. Dosya Türü Dağılımı"])
        for extension, count in extension_counts.most_common(12):
            content.append(f"- `{extension}`: {count} dosya")
        content.extend(["\n## 7. Dizin Ağacı", "```text"])
        content.extend(self._render_tree(files))
        content.append("```")

        return "\n".join(content)

    def save(self, output_file: str = "CLAUDE.md") -> Path:
        output_path = (self.root_path / output_file).absolute()
        output_path.write_text(self.generate_sot(), encoding="utf-8")
        return output_path
