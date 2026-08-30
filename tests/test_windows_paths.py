import json
import tempfile
import unittest
from pathlib import Path

from air_context_engine.analysis.cache_manager import CacheManager
from air_context_engine.analysis.graph_builder import GraphBuilder
from air_context_engine.analysis.scanner import RepositoryScanner
from air_context_engine.search.rag_engine import RAGEngine


class WindowsPathBehaviorTests(unittest.TestCase):
    """Gerçek Windows runner'da da çalışan path sözleşmeleri."""

    def test_nested_scanner_paths_remain_under_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            nested = root / "package" / "submodule"
            nested.mkdir(parents=True)
            source = nested / "service.py"
            source.write_text("VALUE = 1\n", encoding="utf-8")

            scanned = list(RepositoryScanner(root).scan_files())

            self.assertEqual(scanned, [source])
            self.assertEqual(
                scanned[0].relative_to(root).parts,
                ("package", "submodule", "service.py"),
            )

    def test_cache_keys_are_portable_posix_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            nested = root / "package" / "submodule"
            nested.mkdir(parents=True)
            source = nested / "service.py"
            source.write_text("VALUE = 1\n", encoding="utf-8")
            manager = CacheManager(root)

            with manager.transaction():
                manager.get_changed_and_deleted_files([source])

            cache = json.loads((root / ".air_cache.json").read_text(encoding="utf-8"))
            self.assertIn("package/submodule/service.py", cache["files"])
            self.assertNotIn("package\\submodule\\service.py", cache["files"])

    def test_python_module_names_use_dots_on_all_platforms(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            package = root / "package" / "submodule"
            package.mkdir(parents=True)
            (root / "package" / "__init__.py").write_text("", encoding="utf-8")
            (package / "__init__.py").write_text("", encoding="utf-8")
            source = package / "service.py"
            source.write_text("VALUE = 1\n", encoding="utf-8")
            builder = GraphBuilder(root, RepositoryScanner(root))

            self.assertEqual(
                builder._python_module_name(source), "package.submodule.service"
            )

    def test_sqlite_index_uses_portable_paths_and_reopens(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            nested = root / "package" / "submodule"
            nested.mkdir(parents=True)
            source = nested / "service.py"
            source.write_text("def locate_symbol(): pass\n", encoding="utf-8")

            cold = RAGEngine(root, RepositoryScanner(root))
            warm = RAGEngine(root, RepositoryScanner(root))
            result = warm.search("locateSymbol", top_k=1)[0]

            self.assertTrue((root / ".air_index.sqlite3").exists())
            self.assertEqual(cold.index_stats.parsed_files, 1)
            self.assertEqual(warm.index_stats.reused_files, 1)
            self.assertEqual(result["file_path"], "package/submodule/service.py")
            self.assertNotIn("\\", result["file_path"])


if __name__ == "__main__":
    unittest.main()
