import ast
from collections import Counter, defaultdict
from pathlib import Path

from .cache_manager import CacheManager
from .scanner import RepositoryScanner
from .tree_sitter_support import extract_symbol_signatures, language_for_extension


class RepoMapBuilder:
    def __init__(
        self,
        root_path: Path,
        scanner: RepositoryScanner,
        cache_mgr: CacheManager | None = None,
    ) -> None:
        self.root_path = root_path.absolute()
        self.scanner = scanner
        self.cache_mgr = cache_mgr or CacheManager(self.root_path)

    def _parse_python_file(self, file_path: Path) -> str:
        """Python dosyalarındaki sınıf ve fonksiyon imzalarını AST ile detaylı çıkarır."""
        try:
            tree = ast.parse(
                file_path.read_text(encoding="utf-8", errors="ignore"),
                filename=str(file_path),
            )
        except (OSError, SyntaxError, UnicodeError):
            return ""

        lines: list[str] = []
        function_nodes = (ast.FunctionDef, ast.AsyncFunctionDef)

        def get_return_type(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
            return_annotation = node.returns
            if return_annotation is None:
                return ""
            try:
                return f" -> {ast.unparse(return_annotation)}"
            except Exception:
                return " -> Any"

        def decorators(
            node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
        ) -> list[str]:
            rendered: list[str] = []
            for item in node.decorator_list:
                try:
                    rendered.append(f"@{ast.unparse(item)}")
                except (TypeError, ValueError):
                    rendered.append("@<unparseable>")
            return rendered

        def function_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
            keyword = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
            return f"{keyword} {node.name}({ast.unparse(node.args)}){get_return_type(node)}:"

        def overload_prefix(
            node: ast.FunctionDef | ast.AsyncFunctionDef,
            counts: Counter[str],
            tracker: defaultdict[str, int],
        ) -> str:
            tracker[node.name] += 1
            total = counts[node.name]
            return f"[Overload {tracker[node.name]}/{total}] " if total > 1 else ""

        named_nodes = (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)

        def scoped_children(
            nodes: list[ast.stmt],
        ) -> list[ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef]:
            """Başka bir adlandırılmış scope'a girmeden bu scope'un sembollerini bulur."""
            children: list[ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef] = []
            for node in nodes:
                if isinstance(node, named_nodes):
                    children.append(node)
                else:
                    nested_statements = [
                        child
                        for child in ast.iter_child_nodes(node)
                        if isinstance(child, ast.stmt)
                    ]
                    children.extend(scoped_children(nested_statements))
            return children

        def render_scope(nodes: list[ast.stmt], indent: int) -> None:
            children = scoped_children(nodes)
            counts = Counter(
                node.name for node in children if isinstance(node, function_nodes)
            )
            tracker: defaultdict[str, int] = defaultdict(int)
            prefix = " " * indent
            for node in children:
                lines.extend(f"{prefix}{item}" for item in decorators(node))
                if isinstance(node, ast.ClassDef):
                    bases: list[str] = []
                    for base in node.bases:
                        try:
                            bases.append(ast.unparse(base))
                        except (TypeError, ValueError):
                            bases.append("Any")
                    for keyword in node.keywords:
                        if keyword.arg is not None:
                            try:
                                bases.append(
                                    f"{keyword.arg}={ast.unparse(keyword.value)}"
                                )
                            except (TypeError, ValueError):
                                bases.append(f"{keyword.arg}=Any")
                    suffix = f"({', '.join(bases)})" if bases else ""
                    lines.append(f"{prefix}class {node.name}{suffix}:")
                    render_scope(node.body, indent + 2)
                    continue

                index_prefix = overload_prefix(node, counts, tracker)
                signature = function_signature(node)
                if index_prefix:
                    signature = signature.replace(
                        f"{node.name}(", f"{index_prefix}{node.name}(", 1
                    )
                lines.append(f"{prefix}{signature}")
                render_scope(node.body, indent + 2)

        render_scope(tree.body, 2)

        return "\n".join(lines)

    def _parse_with_treesitter(self, file_path: Path, lang: str) -> str:
        """Registry'deki diller için Tree-sitter ile gövdesiz imzalar çıkarır."""
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                code = f.read()
            extracted_symbols = extract_symbol_signatures(lang, code)
            if not extracted_symbols:
                return "  // Bu dil için sembol bulunamadı veya parser hazır değil."

            return "\n".join(f"  {symbol}" for symbol in extracted_symbols)
        except (OSError, UnicodeError, ValueError) as exc:
            return f"  // Tree-sitter parse hatası: {exc}"

    def _get_language_from_extension(self, ext: str) -> str | None:
        """Dosya uzantısını Tree-sitter dil adına eşler."""
        return language_for_extension(ext)

    def collect_symbol_sections(self) -> dict[str, str]:
        """Dosya yollarını oluşturulan imza bölümlerine eşler."""
        files = list(self.scanner.scan_files())
        with self.cache_mgr.transaction():
            changed_files, _ = self.cache_mgr.get_changed_and_deleted_files(files)
            cached_symbols = self.cache_mgr.cache_data.setdefault("symbols", {})

            for file_path in files:
                rel_path = str(file_path.relative_to(self.root_path)).replace("\\", "/")

                if file_path not in changed_files and rel_path in cached_symbols:
                    continue

                ext = file_path.suffix.lower()
                symbols = ""
                if ext == ".py":
                    symbols = self._parse_python_file(file_path)
                else:
                    lang = self._get_language_from_extension(ext)
                    if lang:
                        symbols = self._parse_with_treesitter(file_path, lang)

                cached_symbols[rel_path] = symbols

            return dict(cached_symbols)

    def generate_map(self) -> str:
        symbols_snapshot = self.collect_symbol_sections()
        output = ["# REPOSITORY TOPOLOGY MAP (SIGNATURE INDEX)\n"]
        for rel_path, symbols in sorted(symbols_snapshot.items()):
            if symbols:
                ext = Path(rel_path).suffix.replace(".", "") or "text"
                output.append(f"### 📄 `{rel_path}`")
                output.append(f"```{ext}\n{symbols}\n```\n")

        return "\n".join(output)

    def save(self, output_file: str = "repo_map.md") -> Path:
        out_path = self.root_path / output_file
        out_path.write_text(self.generate_map(), encoding="utf-8")
        return out_path
