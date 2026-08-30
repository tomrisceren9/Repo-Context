from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any

from ..analysis.graph_builder import GraphBuilder
from ..analysis.repo_map import RepoMapBuilder
from ..analysis.scanner import RepositoryScanner
from ..analysis.sot_builder import SOTBuilder
from ..analysis.tree_sitter_support import (
    extract_symbol_ranges,
    language_for_extension,
)


QUALITY_THRESHOLDS = {
    "file_inventory_coverage": 100.0,
    "symbol_recall": 95.0,
    "graph_completeness": 90.0,
    "compression_ratio": 50.0,
}


class StaticBenchmarkEngine:
    """Üç bağlam dosyasının kapsamını dış servis kullanmadan ölçer."""

    def __init__(self, root_path: Path) -> None:
        self.root_path = root_path.absolute()
        self.scanner = RepositoryScanner(self.root_path)

    @staticmethod
    def _estimate_tokens(character_count: int) -> int:
        return character_count // 4

    @staticmethod
    def _python_symbol_count(content: str) -> int:
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return 0
        return sum(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            for node in ast.walk(tree)
        )

    def _count_raw_symbols(self, files: list[Path]) -> int:
        """AST/Tree-sitter ile kaynak depodaki yapısal sembolleri sayar."""
        symbol_count = 0
        for file_path in files:
            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if file_path.suffix.lower() == ".py":
                symbol_count += self._python_symbol_count(content)
                continue
            language = language_for_extension(file_path.suffix.lower())
            if language:
                symbol_count += len(extract_symbol_ranges(language, content))
        return symbol_count

    @staticmethod
    def _count_rendered_symbols(symbol_sections: dict[str, str]) -> int:
        """Repo map üreticisinin gerçekten belgeye koyduğu imzaları sayar."""
        count = 0
        python_pattern = re.compile(r"^\s*(?:async\s+def|def|class)\s+")
        for file_path, section in symbol_sections.items():
            if not section or section.lstrip().startswith("//"):
                continue
            lines = section.splitlines()
            if Path(file_path).suffix.lower() == ".py":
                count += sum(bool(python_pattern.match(line)) for line in lines)
            else:
                count += sum(
                    bool(line.strip()) and "parse hatası" not in line for line in lines
                )
        return count

    def _raw_graph_edges(
        self,
        graph_builder: GraphBuilder,
    ) -> set[tuple[str, str, str]]:
        module_dependencies, class_inheritance = graph_builder.collect_relationships()
        import_edges = {
            ("import", source, target)
            for source, targets in module_dependencies.items()
            for target in targets
        }
        inheritance_edges = {
            ("inheritance", child, parent) for child, parent in class_inheritance
        }
        return import_edges | inheritance_edges

    @staticmethod
    def _rendered_graph_edge_count(graph_content: str) -> int:
        return sum(
            "-->" in line or "<|--" in line for line in graph_content.splitlines()
        )

    @staticmethod
    def _tree_files(sot_content: str) -> set[str]:
        """CLAUDE.md içindeki girintili ağacı yeniden yol listesine çevirir."""
        marker = "## 7. Dizin Ağacı"
        if marker not in sot_content:
            return set()
        tree_block = sot_content.split(marker, 1)[1]
        if "```text" not in tree_block:
            return set()
        tree_block = tree_block.split("```text", 1)[1].split("```", 1)[0]
        directories: list[str] = []
        files: set[str] = set()
        for raw_line in tree_block.splitlines():
            if not raw_line.strip():
                continue
            depth = (len(raw_line) - len(raw_line.lstrip(" "))) // 2
            name = raw_line.strip()
            if name.endswith("/"):
                directories = directories[:depth]
                directories.append(name[:-1])
            else:
                files.add("/".join((*directories[:depth], name)))
        return files

    @staticmethod
    def _gate(value: float, threshold: float) -> dict[str, Any]:
        return {
            "value": round(value, 2),
            "minimum": threshold,
            "passed": value >= threshold,
        }

    def run_benchmark(self) -> dict[str, Any]:
        files = list(self.scanner.scan_files())
        if not files:
            return {"error": "Taranacak dosya bulunamadı."}

        raw_character_count = 0
        for file_path in files:
            try:
                raw_character_count += len(
                    file_path.read_text(encoding="utf-8", errors="ignore")
                )
            except OSError:
                continue
        raw_tokens = self._estimate_tokens(raw_character_count)

        sot_builder = SOTBuilder(self.root_path, self.scanner)
        repo_map_builder = RepoMapBuilder(self.root_path, self.scanner)
        graph_builder = GraphBuilder(self.root_path, self.scanner)
        sot_content = sot_builder.generate_sot()
        symbol_sections = repo_map_builder.collect_symbol_sections()
        map_content = repo_map_builder.generate_map()
        graph_content = graph_builder.generate_mermaid_graph()

        context = "\n".join((sot_content, map_content, graph_content))
        context_tokens = self._estimate_tokens(len(context))
        compression = (
            (raw_tokens - context_tokens) / raw_tokens * 100 if raw_tokens else 0.0
        )
        lines = context.splitlines()
        density = sum(bool(line.strip()) for line in lines) / len(lines) * 100

        raw_symbols = self._count_raw_symbols(files)
        rendered_symbols = self._count_rendered_symbols(symbol_sections)
        symbol_recall = rendered_symbols / raw_symbols * 100 if raw_symbols else 100.0

        raw_edges = self._raw_graph_edges(graph_builder)
        rendered_edges = self._rendered_graph_edge_count(graph_content)
        graph_completeness = (
            rendered_edges / len(raw_edges) * 100 if raw_edges else 100.0
        )

        expected_files = {
            file_path.relative_to(self.root_path).as_posix() for file_path in files
        }
        rendered_files = self._tree_files(sot_content)
        inventory_coverage = (
            len(expected_files & rendered_files) / len(expected_files) * 100
        )

        gates = {
            "file_inventory_coverage": self._gate(
                inventory_coverage,
                QUALITY_THRESHOLDS["file_inventory_coverage"],
            ),
            "symbol_recall": self._gate(
                symbol_recall,
                QUALITY_THRESHOLDS["symbol_recall"],
            ),
            "graph_completeness": self._gate(
                graph_completeness,
                QUALITY_THRESHOLDS["graph_completeness"],
            ),
            "compression_ratio": self._gate(
                compression,
                QUALITY_THRESHOLDS["compression_ratio"],
            ),
        }
        return {
            "total_files": len(files),
            "raw_tokens": raw_tokens,
            "context_tokens": context_tokens,
            "ccr_compression_ratio": round(max(compression, 0.0), 2),
            "cds_density_score": round(density, 2),
            "fic_file_inventory_coverage": round(min(inventory_coverage, 100.0), 2),
            "srr_symbol_recall": round(min(symbol_recall, 100.0), 2),
            "cgc_graph_completeness": round(min(graph_completeness, 100.0), 2),
            "raw_symbols_count": raw_symbols,
            "extracted_symbols_count": rendered_symbols,
            "raw_graph_edges_count": len(raw_edges),
            "rendered_graph_edges_count": rendered_edges,
            "quality_gates": gates,
            "quality_gate_passed": all(gate["passed"] for gate in gates.values()),
        }
