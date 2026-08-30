import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from air_context_engine.analysis.cache_manager import CacheManager
from air_context_engine.analysis.graph_builder import GraphBuilder
from air_context_engine.analysis.repo_map import RepoMapBuilder
from air_context_engine.analysis.scanner import RepositoryScanner
from air_context_engine.analysis.sot_builder import SOTBuilder
from air_context_engine.analysis.tree_sitter_support import get_parser
from air_context_engine.search.rag_engine import ASTAwareCodeChunker, RAGEngine


class CacheCoverageTests(unittest.TestCase):
    def test_cache_normalization_corruption_change_and_deletion(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_path = root / ".air_cache.json"
            cache_path.write_text(
                json.dumps({"files": [], "symbols": {"gone.py": "old"}}),
                encoding="utf-8",
            )
            manager = CacheManager(root)
            self.assertEqual(manager.cache_data["files"], {})
            self.assertEqual(manager.cache_data["symbols"], {})
            self.assertEqual(manager.cache_data["version"], 4)

            cache_path.write_text("not-json", encoding="utf-8")
            self.assertEqual(manager._load_cache()["version"], 4)

            source = root / "live.py"
            source.write_text("VALUE = 1\n", encoding="utf-8")
            with manager.transaction():
                changed, deleted = manager.get_changed_and_deleted_files([source])
                manager.cache_data["symbols"]["deleted.py"] = "symbol"
                manager.cache_data["dependencies"]["deleted.py"] = ["x"]
                manager.cache_data["files"]["deleted.py"] = {"hash": "x"}
            self.assertEqual(changed, {source})
            self.assertEqual(deleted, set())

            unchanged, deleted = manager.get_changed_and_deleted_files([source])
            self.assertEqual(unchanged, set())
            self.assertEqual(deleted, {"deleted.py"})
            self.assertNotIn("deleted.py", manager.cache_data["symbols"])
            self.assertNotIn("deleted.py", manager.cache_data["dependencies"])

            manager.save_cache()
            self.assertTrue(cache_path.exists())
            self.assertEqual(manager._calculate_hash(root / "missing.py"), "")

    def test_atomic_temp_file_is_cleaned_when_replace_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manager = CacheManager(root)
            with patch(
                "air_context_engine.analysis.cache_manager.os.replace",
                side_effect=OSError("replace failed"),
            ):
                with self.assertRaisesRegex(OSError, "replace failed"):
                    manager._save_cache_atomic()
            self.assertEqual(list(root.glob(".*.tmp")), [])


class RepoMapCoverageTests(unittest.TestCase):
    def test_python_signatures_cache_refresh_and_deleted_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "sample.py"
            source.write_text(
                "def top(value: int) -> str:\n"
                "    return str(value)\n\n"
                "class Service:\n"
                "    def run(self, item) -> bool:\n"
                "        return True\n",
                encoding="utf-8",
            )
            builder = RepoMapBuilder(root, RepositoryScanner(root))

            first = builder.generate_map()
            second = builder.generate_map()

            self.assertIn("def top(value: int) -> str:", first)
            self.assertIn("class Service:", first)
            self.assertIn("def run(self, item) -> bool:", first)
            self.assertEqual(first, second)

            source.unlink()
            empty = builder.generate_map()
            self.assertNotIn("sample.py", empty)

    def test_python_overload_labels_and_invalid_python(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            valid = root / "valid.py"
            valid.write_text(
                "class A:\n"
                "    def same(self, value: int): pass\n"
                "    def same(self, value: str): pass\n\n"
                "class B:\n"
                "    def same(self): pass\n",
                encoding="utf-8",
            )
            invalid = root / "invalid.py"
            invalid.write_text("def broken(:\n", encoding="utf-8")
            builder = RepoMapBuilder(root, RepositoryScanner(root))

            signatures = builder._parse_python_file(valid)

            self.assertIn("[Overload 1/2] same", signatures)
            self.assertIn("[Overload 2/2] same", signatures)
            self.assertEqual(signatures.count("[Overload"), 2)
            self.assertEqual(builder._parse_python_file(invalid), "")

    def test_cpp_symbols_language_mapping_and_save(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "service.cpp"
            source.write_text(
                "class Service { public: void run(); };\n"
                "void calculate(int value) { }\n",
                encoding="utf-8",
            )
            builder = RepoMapBuilder(root, RepositoryScanner(root))

            symbols = builder._parse_with_treesitter(source, "cpp")
            output_path = builder.save("symbols.md")

            self.assertIn("Service", symbols)
            self.assertIn("service.cpp", output_path.read_text(encoding="utf-8"))
            self.assertEqual(builder._get_language_from_extension(".hpp"), "cpp")
            self.assertIsNone(builder._get_language_from_extension(".unknown"))
            self.assertIn("Service", builder._parse_with_treesitter(source, "java"))


class GraphCoverageTests(unittest.TestCase):
    def test_absolute_imports_inheritance_invalid_source_and_cpp_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "pkg").mkdir()
            (root / "pkg" / "__init__.py").write_text("", encoding="utf-8")
            (root / "pkg" / "utils.py").write_text("VALUE = 1\n", encoding="utf-8")
            (root / "pkg" / "service.py").write_text(
                "import pkg.utils\nclass Child(pkg.utils.Base):\n    pass\n",
                encoding="utf-8",
            )
            invalid = root / "broken.py"
            invalid.write_text("class Broken(:\n", encoding="utf-8")
            cpp = root / "child.cpp"
            cpp.write_text("class ChildCpp : public BaseCpp {};\n", encoding="utf-8")
            builder = GraphBuilder(root, RepositoryScanner(root))

            dependencies, inheritance = builder.collect_relationships()
            mermaid_path = builder.save()

            self.assertEqual(dependencies["pkg.service"], {"pkg.utils"})
            self.assertIn(("pkg.service.Child", "pkg.utils.Base"), inheritance)
            self.assertIn(("ChildCpp", "BaseCpp"), inheritance)
            self.assertEqual(
                builder._extract_python_dependencies_and_inheritance(
                    invalid,
                    "broken",
                    {"broken"},
                ),
                (set(), []),
            )
            self.assertIn("pkg.service", mermaid_path.read_text(encoding="utf-8"))

    def test_relative_import_beyond_root_and_empty_graph(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "single.py"
            source.write_text("from ...missing import value\n", encoding="utf-8")
            builder = GraphBuilder(root, RepositoryScanner(root))

            dependencies, inheritance = builder.collect_relationships()
            graph = builder.generate_mermaid_graph()

            self.assertEqual(dependencies["single"], set())
            self.assertEqual(inheritance, [])
            self.assertIn("bağımlılığı bulunamadı", graph)
            self.assertIn("Inheritance", graph)


class RAGCoverageTests(unittest.TestCase):
    def test_cpp_tree_sitter_and_text_sliding_window(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cpp = root / "sample.cpp"
            cpp.write_text("void calculate(int value) { value++; }\n", encoding="utf-8")
            text = root / "notes.txt"
            text.write_text(
                "\n".join(f"line word{index}" for index in range(12)), encoding="utf-8"
            )
            chunker = ASTAwareCodeChunker(fallback_lines=6)

            cpp_chunks = chunker.chunk_file(cpp, root)
            text_chunks = chunker.chunk_file(text, root)

            self.assertTrue(cpp_chunks)
            self.assertIn("calculate", cpp_chunks[0]["content"])
            self.assertGreater(len(text_chunks), 1)
            self.assertEqual(text_chunks[0]["start_line"], 1)
            self.assertEqual(text_chunks[-1]["end_line"], 12)

    def test_empty_file_missing_file_and_zero_similarity(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            empty = root / "empty.py"
            empty.write_text("", encoding="utf-8")
            missing = root / "missing.py"
            chunker = ASTAwareCodeChunker()

            self.assertEqual(chunker.chunk_file(empty, root), [])
            self.assertEqual(chunker.chunk_file(missing, root), [])

            source = root / "source.py"
            source.write_text("def alpha():\n    return beta\n", encoding="utf-8")
            engine = RAGEngine(root, RepositoryScanner(root))
            self.assertEqual(engine.search("completely_unrelated"), [])


class SOTCoverageTests(unittest.TestCase):
    def test_frontend_docker_and_command_detection(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "package.json").write_text(
                json.dumps(
                    {
                        "scripts": {"dev": "next dev"},
                        "dependencies": {"next": "1", "react": "1"},
                        "devDependencies": {"tailwindcss": "1", "typescript": "1"},
                    }
                ),
                encoding="utf-8",
            )
            (root / "Dockerfile").write_text("FROM python:3.13\n", encoding="utf-8")
            (root / "docker-compose.yml").write_text("services: {}\n", encoding="utf-8")
            builder = SOTBuilder(root, RepositoryScanner(root))

            content = builder.generate_sot()

            self.assertIn("Next.js", content)
            self.assertIn("React", content)
            self.assertIn("Tailwind CSS", content)
            self.assertIn("TypeScript", content)
            self.assertIn("Docker / Docker Compose", content)
            self.assertIn("docker compose -f docker-compose.yml up --build", content)
            self.assertIn("npm install", content)
            self.assertIn("npm run dev", content)

    def test_empty_project_reports_no_commands(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            content = SOTBuilder(root, RepositoryScanner(root)).generate_sot()
            self.assertIn("Komutlar tespit edilemedi", content)

    def test_pyproject_console_script_is_reported_without_inventing_cli_command(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "pyproject.toml").write_text(
                '[project]\nname = "sample"\nversion = "1.0"\n'
                '[project.scripts]\nsample = "sample.cli:main"\n',
                encoding="utf-8",
            )
            (root / "cli.py").write_text("def main(): pass\n", encoding="utf-8")

            content = SOTBuilder(root, RepositoryScanner(root)).generate_sot()

            self.assertIn("python -m pip install -e .", content)
            self.assertIn("sample = sample.cli:main", content)
            self.assertNotIn("python cli.py --help", content)


class TreeSitterCoverageTests(unittest.TestCase):
    def test_supported_and_unsupported_languages(self) -> None:
        self.assertIsNotNone(get_parser("c"))
        self.assertIsNotNone(get_parser("python"))
        self.assertIsNotNone(get_parser("java"))
        self.assertIsNone(get_parser("unknown"))


if __name__ == "__main__":
    unittest.main()
