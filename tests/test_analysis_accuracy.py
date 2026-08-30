import re
import tempfile
import unittest
from pathlib import Path

from air_context_engine.analysis.graph_builder import GraphBuilder
from air_context_engine.analysis.repo_map import RepoMapBuilder
from air_context_engine.analysis.scanner import RepositoryScanner
from air_context_engine.analysis.tree_sitter_support import GRAMMAR_REGISTRY, get_parser
from air_context_engine.benchmarks.benchmark_static import StaticBenchmarkEngine


FIXTURES = Path(__file__).parent / "fixtures"

EXPECTED_SYMBOLS = {
    "sample.py": {
        "component",
        "route",
        "ServiceMeta",
        "BaseService",
        "Service",
        "run",
        "stop",
        "helper",
        "parse",
        "validate",
        "build",
    },
    "sample.c": {
        "Service",
        "Payload",
        "State",
        "add",
        "subtract",
        "multiply",
        "divide",
        "reset",
        "create_service",
        "destroy_service",
    },
    "sample.cpp": {
        "application",
        "Service",
        "run",
        "stop",
        "Worker",
        "execute",
        "Mode",
        "add",
        "subtract",
        "create_service",
        "destroy_service",
    },
    "sample.js": {
        "Service",
        "run",
        "stop",
        "BaseService",
        "start",
        "helper",
        "parse",
        "validate",
        "stream",
        "normalize",
    },
    "sample.ts": {
        "Runner",
        "run",
        "stop",
        "Config",
        "Mode",
        "Service",
        "helper",
        "parse",
        "validate",
        "normalize",
    },
    "sample.tsx": {
        "Props",
        "Loader",
        "load",
        "Widget",
        "render",
        "mount",
        "helper",
        "Panel",
        "formatter",
        "normalize",
    },
    "Sample.java": {
        "Sample",
        "run",
        "stop",
        "Runner",
        "execute",
        "Mode",
        "Payload",
        "Helper",
        "add",
        "subtract",
        "parse",
    },
    "Sample.cs": {
        "IRunner",
        "Run",
        "Stop",
        "Mode",
        "Payload",
        "Worker",
        "Work",
        "Service",
        "RunAsync",
        "Utility",
        "Parse",
    },
    "sample.go": {
        "Service",
        "Config",
        "Mode",
        "Runner",
        "Run",
        "Stop",
        "helper",
        "parse",
        "validate",
    },
    "sample.rs": {
        "Service",
        "Config",
        "Worker",
        "State",
        "Runner",
        "run",
        "stop",
        "helper",
        "parse",
        "validate",
        "execute",
    },
}

EXPECTED_GRAPH_EDGES = {
    ("app", "app.services"),
    ("app.models", "app.models.user"),
    ("app.models.user", "app.utils"),
    ("app.repositories", "app.repositories.user_repo"),
    ("app.repositories.user_repo", "app.models.user"),
    ("app.services", "app.models.user"),
    ("app.services", "app.repositories.user_repo"),
    ("app.services", "app.utils"),
}


class AnalysisAccuracyTests(unittest.TestCase):
    def test_every_registry_grammar_loads_and_parses_without_errors(self) -> None:
        samples = {
            "c": b"int main(void) { return 0; }",
            "cpp": b"class Service {};",
            "c_sharp": b"class Service { void Run() {} }",
            "go": b"package main\nfunc main() {}",
            "java": b"class Service { void run() {} }",
            "javascript": b"function run() {}",
            "python": b"def run(): pass",
            "rust": b"fn main() {}",
            "typescript": b"function run(value: number): number { return value; }",
            "tsx": b"const App = () => <div />;",
        }
        self.assertEqual(set(samples), set(GRAMMAR_REGISTRY))
        for language, source in samples.items():
            with self.subTest(language=language):
                parser = get_parser(language)
                self.assertIsNotNone(parser)
                assert parser is not None
                self.assertFalse(parser.parse(source).root_node.has_error)

    def test_modern_javascript_and_typescript_extensions_use_registry(self) -> None:
        builder = RepoMapBuilder(
            FIXTURES / "multilang",
            RepositoryScanner(FIXTURES / "multilang"),
        )
        for extension in (".js", ".jsx", ".mjs", ".cjs"):
            self.assertEqual(
                builder._get_language_from_extension(extension), "javascript"
            )
        for extension in (".ts", ".mts", ".cts"):
            self.assertEqual(
                builder._get_language_from_extension(extension), "typescript"
            )
        self.assertEqual(builder._get_language_from_extension(".tsx"), "tsx")

    def test_symbol_recall_is_at_least_95_percent_for_every_language(self) -> None:
        builder = RepoMapBuilder(
            FIXTURES / "multilang",
            RepositoryScanner(FIXTURES / "multilang"),
        )
        recalls: dict[str, float] = {}
        for filename, expected in EXPECTED_SYMBOLS.items():
            path = FIXTURES / "multilang" / filename
            if path.suffix == ".py":
                output = builder._parse_python_file(path)
            else:
                language = builder._get_language_from_extension(path.suffix)
                self.assertIsNotNone(language)
                output = builder._parse_with_treesitter(path, str(language))
            found = {
                symbol
                for symbol in expected
                if re.search(rf"\b{re.escape(symbol)}\b", output)
            }
            recalls[filename] = len(found) / len(expected)

        self.assertTrue(recalls)
        for filename, recall in recalls.items():
            with self.subTest(filename=filename, recall=recall):
                self.assertGreaterEqual(recall, 0.95)

    def test_internal_dependency_edges_reach_90_percent_precision_and_recall(
        self,
    ) -> None:
        root = FIXTURES / "python_graph"
        dependencies, _ = GraphBuilder(
            root, RepositoryScanner(root)
        ).collect_relationships()
        actual = {
            (source, target)
            for source, targets in dependencies.items()
            for target in targets
        }
        true_positive = len(actual & EXPECTED_GRAPH_EDGES)
        precision = true_positive / len(actual) if actual else 0.0
        recall = true_positive / len(EXPECTED_GRAPH_EDGES)

        self.assertGreaterEqual(precision, 0.90, actual)
        self.assertGreaterEqual(recall, 0.90, actual)

    def test_static_graph_completeness_counts_import_and_inheritance_edges(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "base.py").write_text("class Base: pass\n", encoding="utf-8")
            (root / "child.py").write_text(
                "from base import Base\n\nclass Child(Base): pass\n",
                encoding="utf-8",
            )

            metrics = StaticBenchmarkEngine(root).run_benchmark()

            self.assertEqual(metrics["raw_graph_edges_count"], 2)
            self.assertEqual(metrics["rendered_graph_edges_count"], 2)
            self.assertEqual(metrics["cgc_graph_completeness"], 100.0)

    def test_python_signatures_preserve_full_call_and_type_semantics(self) -> None:
        root = FIXTURES / "multilang"
        output = RepoMapBuilder(root, RepositoryScanner(root))._parse_python_file(
            root / "sample.py"
        )
        self.assertIn('@component("core")', output.replace("'", '"'))
        self.assertIn("class Service(BaseService, metaclass=ServiceMeta):", output)
        self.assertIn('@route("/run")', output.replace("'", '"'))
        self.assertIn(
            "async def run(self, value: int=1, /, *, verbose: bool=False) -> str:",
            output,
        )
        self.assertIn("def helper(value: str='default') -> str:", output)

    def test_mermaid_ids_are_unique_stable_and_labels_are_escaped(self) -> None:
        first = GraphBuilder._mermaid_node('pkg.one.service\n"quoted"', "m")
        repeated = GraphBuilder._mermaid_node('pkg.one.service\n"quoted"', "m")
        same_stem = GraphBuilder._mermaid_node("pkg.two.service", "m")

        self.assertEqual(first, repeated)
        self.assertNotEqual(first.split("[", 1)[0], same_stem.split("[", 1)[0])
        self.assertIn("&quot;quoted&quot;", first)
        self.assertIn("&#10;", first)

    def test_same_stem_modules_remain_distinct_in_rendered_graph(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "one").mkdir()
            (root / "two").mkdir()
            for package in ("one", "two"):
                (root / package / "__init__.py").write_text("", encoding="utf-8")
                (root / package / "service.py").write_text(
                    "VALUE = 1\n", encoding="utf-8"
                )
            (root / "consumer.py").write_text(
                "import one.service\nimport two.service\n",
                encoding="utf-8",
            )
            graph = GraphBuilder(root, RepositoryScanner(root)).generate_mermaid_graph()

            first_id = GraphBuilder._mermaid_node("one.service", "m").split("[", 1)[0]
            second_id = GraphBuilder._mermaid_node("two.service", "m").split("[", 1)[0]
            self.assertNotEqual(first_id, second_id)
            self.assertIn(first_id, graph)
            self.assertIn(second_id, graph)

    def test_typescript_relative_and_project_alias_imports_are_connected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            frontend = root / "frontend"
            (frontend / "app").mkdir(parents=True)
            (frontend / "lib").mkdir()
            (frontend / "components").mkdir()
            (frontend / "tsconfig.json").write_text("{}\n", encoding="utf-8")
            (frontend / "lib" / "api.ts").write_text(
                "export const request = () => 1;\n",
                encoding="utf-8",
            )
            (frontend / "components" / "button.tsx").write_text(
                "export const Button = () => null;\n",
                encoding="utf-8",
            )
            (frontend / "app" / "page.tsx").write_text(
                'import { request } from "@/lib/api";\n'
                'export { Button } from "../components/button";\n',
                encoding="utf-8",
            )

            dependencies, _ = GraphBuilder(
                root,
                RepositoryScanner(root),
            ).collect_relationships()

            self.assertEqual(
                dependencies["frontend.app.page"],
                {"frontend.lib.api", "frontend.components.button"},
            )

    def test_nested_python_functions_remain_visible_in_signature_map(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "service.py"
            source.write_text(
                "def outer():\n"
                "    def inner():\n"
                "        return True\n"
                "    return inner()\n",
                encoding="utf-8",
            )

            output = RepoMapBuilder(root, RepositoryScanner(root))._parse_python_file(
                source
            )

            self.assertIn("def outer():", output)
            self.assertIn("def inner():", output)


class ScannerPolicyTests(unittest.TestCase):
    def test_binary_detection_and_user_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / ".air-context.toml").write_text(
                "[scanner]\n"
                'allowed_extensions = [".custom", ".py"]\n'
                "max_file_size = 20\n"
                "detect_binary = true\n"
                'ignore_patterns = ["ignored.custom"]\n',
                encoding="utf-8",
            )
            (root / "source.custom").write_text("text source\n", encoding="utf-8")
            (root / "ignored.custom").write_text("ignored\n", encoding="utf-8")
            (root / "binary.py").write_bytes(b"header\x00payload")
            (root / "large.py").write_text("x" * 21, encoding="utf-8")
            (root / "not-allowed.txt").write_text("text\n", encoding="utf-8")

            scanned = {path.name for path in RepositoryScanner(root).scan_files()}
            binary_override = {
                path.name
                for path in RepositoryScanner(
                    root,
                    allowed_extensions={".py"},
                    max_file_size=100,
                    detect_binary=False,
                ).scan_files()
            }

            self.assertEqual(scanned, {"source.custom"})
            self.assertEqual(binary_override, {"binary.py", "large.py"})

    def test_invalid_explicit_configuration_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            invalid = root / "invalid.toml"
            invalid.write_text(
                '[scanner]\nallowed_extensions = "py"\n', encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "metin listesi"):
                RepositoryScanner(root, config_path=invalid)


if __name__ == "__main__":
    unittest.main()
