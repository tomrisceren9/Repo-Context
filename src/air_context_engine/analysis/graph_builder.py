import ast
import hashlib
import html
import re
from collections import defaultdict
from pathlib import Path

from tree_sitter import Node

from ..core.fs import safe_resolve_path
from .scanner import RepositoryScanner
from .tree_sitter_support import get_parser


class GraphBuilder:
    """Paket-duyarlı modül bağımlılığı ve sınıf kalıtım grafı üretir."""

    SCRIPT_EXTENSIONS = frozenset(
        {".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".mts", ".cts"}
    )

    def __init__(self, root_path: Path, scanner: RepositoryScanner):
        self.root_path = root_path.absolute()
        self.scanner = scanner

    def _python_module_name(self, file_path: Path) -> str:
        relative = file_path.relative_to(self.root_path).with_suffix("")
        parts = list(relative.parts)
        if parts and parts[-1] == "__init__":
            parts.pop()
        return ".".join(parts) if parts else self.root_path.name.replace("-", "_")

    def _script_module_name(self, file_path: Path) -> str:
        relative = file_path.relative_to(self.root_path).with_suffix("")
        return ".".join(relative.parts)

    def _script_alias_root(self, file_path: Path) -> Path:
        """`@/` aliası için en yakın tsconfig/jsconfig dizinini kullanır."""
        for parent in (file_path.parent, *file_path.parents):
            if parent == self.root_path.parent:
                break
            if (parent / "tsconfig.json").is_file() or (
                parent / "jsconfig.json"
            ).is_file():
                return parent
            if parent == self.root_path:
                break
        return self.root_path

    def _resolve_script_import(
        self,
        file_path: Path,
        specifier: str,
        modules_by_path: dict[Path, str],
    ) -> str | None:
        if specifier.startswith("@/"):
            candidate = self._script_alias_root(file_path) / specifier[2:]
        elif specifier.startswith("."):
            candidate = file_path.parent / specifier
        else:
            return None

        candidates = [candidate]
        candidates.extend(
            candidate.with_suffix(extension) for extension in self.SCRIPT_EXTENSIONS
        )
        candidates.extend(
            candidate / f"index{extension}" for extension in self.SCRIPT_EXTENSIONS
        )
        for possible_path in candidates:
            module = modules_by_path.get(safe_resolve_path(possible_path, strict=False))
            if module:
                return module
        return None

    def _extract_script_dependencies(
        self,
        file_path: Path,
        modules_by_path: dict[Path, str],
    ) -> set[str]:
        try:
            source = file_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return set()
        patterns = (
            r"(?:import|export)\s+(?:[^;]*?\s+from\s*)?[\"']([^\"']+)[\"']",
            r"(?:require|import)\(\s*[\"']([^\"']+)[\"']\s*\)",
        )
        specifiers = {
            match
            for pattern in patterns
            for match in re.findall(pattern, source, flags=re.MULTILINE)
        }
        return {
            resolved
            for specifier in specifiers
            if (
                resolved := self._resolve_script_import(
                    file_path,
                    specifier,
                    modules_by_path,
                )
            )
        }

    @staticmethod
    def _resolve_known_module(candidate: str, known_modules: set[str]) -> str | None:
        parts = candidate.split(".") if candidate else []
        while parts:
            current = ".".join(parts)
            if current in known_modules:
                return current
            parts.pop()
        return None

    def _resolve_import_from(
        self,
        node: ast.ImportFrom,
        current_module: str,
        is_package: bool,
        known_modules: set[str],
    ) -> set[str]:
        if node.level:
            package_parts = (
                current_module.split(".")
                if is_package
                else current_module.split(".")[:-1]
            )
            parents_to_remove = node.level - 1
            if parents_to_remove > len(package_parts):
                return set()
            base_parts = package_parts[: len(package_parts) - parents_to_remove]
            if node.module:
                base_parts.extend(node.module.split("."))
            base_module = ".".join(base_parts)
        else:
            base_module = node.module or ""

        resolved: set[str] = set()
        for alias in node.names:
            if alias.name == "*":
                candidate = self._resolve_known_module(base_module, known_modules)
            else:
                alias_candidate = (
                    f"{base_module}.{alias.name}" if base_module else alias.name
                )
                candidate = self._resolve_known_module(alias_candidate, known_modules)
                if candidate is None:
                    candidate = self._resolve_known_module(base_module, known_modules)
            if candidate:
                resolved.add(candidate)
        return resolved

    def _extract_python_dependencies_and_inheritance(
        self,
        file_path: Path,
        current_module: str,
        known_modules: set[str],
    ) -> tuple[set[str], list[tuple[str, str]]]:
        imports: set[str] = set()
        inheritance: list[tuple[str, str]] = []
        try:
            with open(
                file_path, "r", encoding="utf-8", errors="ignore"
            ) as source_handle:
                tree = ast.parse(source_handle.read(), filename=str(file_path))
        except (OSError, SyntaxError):
            return imports, inheritance

        is_package = file_path.name == "__init__.py"
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    resolved = self._resolve_known_module(alias.name, known_modules)
                    if resolved:
                        imports.add(resolved)
            elif isinstance(node, ast.ImportFrom):
                imports.update(
                    self._resolve_import_from(
                        node, current_module, is_package, known_modules
                    )
                )
            elif isinstance(node, ast.ClassDef):
                for base in node.bases:
                    try:
                        base_name = ast.unparse(base)
                    except (TypeError, ValueError):
                        continue
                    inheritance.append((f"{current_module}.{node.name}", base_name))

        imports.discard(current_module)
        return imports, inheritance

    def _extract_treesitter_inheritance(
        self,
        file_path: Path,
        language: str,
    ) -> list[tuple[str, str]]:
        inheritance: list[tuple[str, str]] = []
        try:
            code = file_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return inheritance

        parser = get_parser(language)
        if parser is not None:
            try:
                tree = parser.parse(bytes(code, "utf-8"))

                def traverse(node: Node) -> None:
                    if node.type in {
                        "class_specifier",
                        "struct_specifier",
                        "class_declaration",
                    }:
                        first_line = code[node.start_byte : node.end_byte].split("\n")[
                            0
                        ]
                        match = re.search(
                            r"class\s+([A-Za-z_]\w*)\s*:\s*"
                            r"(?:public|protected|private)?\s*([A-Za-z_]\w*)",
                            first_line,
                        )
                        if match:
                            inheritance.append((match.group(1), match.group(2)))
                    for child in node.children:
                        traverse(child)

                traverse(tree.root_node)
            except (AttributeError, TypeError, ValueError):
                pass

        if not inheritance:
            inheritance.extend(
                re.findall(
                    r"class\s+([A-Za-z_]\w*)\s*:\s*"
                    r"(?:public|protected|private)?\s*([A-Za-z_]\w*)",
                    code,
                )
            )
        return inheritance

    def collect_relationships(
        self,
    ) -> tuple[dict[str, set[str]], list[tuple[str, str]]]:
        """Paket niteleyicilerini koruyan bağımlılık ve kalıtım ilişkilerini döndürür."""
        files = list(self.scanner.scan_files())
        python_modules = {
            file_path: self._python_module_name(file_path)
            for file_path in files
            if file_path.suffix.lower() == ".py"
        }
        script_modules = {
            file_path: self._script_module_name(file_path)
            for file_path in files
            if file_path.suffix.lower() in self.SCRIPT_EXTENSIONS
        }
        script_modules_by_path = {
            safe_resolve_path(path, strict=False): module
            for path, module in script_modules.items()
        }
        known_modules = set(python_modules.values())
        module_deps: dict[str, set[str]] = defaultdict(set)
        all_inheritance: list[tuple[str, str]] = []

        for file_path in files:
            extension = file_path.suffix.lower()
            if extension == ".py":
                module_name = python_modules[file_path]
                imports, inheritance = (
                    self._extract_python_dependencies_and_inheritance(
                        file_path,
                        module_name,
                        known_modules,
                    )
                )
                module_deps[module_name].update(imports)
                all_inheritance.extend(inheritance)
            elif extension in self.SCRIPT_EXTENSIONS:
                module_name = script_modules[file_path]
                module_deps[module_name].update(
                    self._extract_script_dependencies(file_path, script_modules_by_path)
                )
                module_deps[module_name].discard(module_name)
            elif extension in {".cpp", ".hpp", ".h", ".cc", ".cxx"}:
                all_inheritance.extend(
                    self._extract_treesitter_inheritance(file_path, "cpp")
                )

        return dict(module_deps), all_inheritance

    @staticmethod
    def _mermaid_node(name: str, prefix: str) -> str:
        digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:32]
        label = (
            html.escape(name, quote=True).replace("\r", "&#13;").replace("\n", "&#10;")
        )
        return f'{prefix}_{digest}["{label}"]'

    def generate_mermaid_graph(self) -> str:
        """Mermaid formatında modül ve kalıtım haritasını oluşturur."""
        module_deps, all_inheritance = self.collect_relationships()
        output = [
            "# PROJECT DEPENDENCY & INHERITANCE GRAPH\n",
            "## 1. Module Dependencies",
            "```mermaid",
            "graph TD",
        ]

        edges = [
            (source, target)
            for source, targets in module_deps.items()
            for target in targets
        ]
        for source, target in sorted(edges):
            output.append(
                f"    {self._mermaid_node(source, 'm')} --> "
                f"{self._mermaid_node(target, 'm')}"
            )
        if not edges:
            output.append(
                "    %% Doğrudan modül bağımlılığı bulunamadı veya modüller bağımsız."
            )

        output.extend(
            ["```\n", "## 2. Class Inheritance Map (OOP)", "```mermaid", "graph TD"]
        )
        unique_inheritance = sorted(set(all_inheritance))
        for child, base in unique_inheritance:
            if child and base:
                output.append(
                    f"    {self._mermaid_node(base, 'c')} <|-- "
                    f"{self._mermaid_node(child, 'c')}"
                )
        if not unique_inheritance:
            output.append(
                "    %% Sınıf kalıtım ilişkisi (Inheritance) tespit edilmedi."
            )
        output.append("```\n")
        return "\n".join(output)

    def save(self, output_file: str = "dependency_graph.md") -> Path:
        out_path = self.root_path / output_file
        out_path.write_text(self.generate_mermaid_graph(), encoding="utf-8")
        return out_path
