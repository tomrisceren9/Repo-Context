import json
import multiprocessing
import tempfile
import time
import unittest
from pathlib import Path

from air_context_engine.analysis.cache_manager import CacheManager
from air_context_engine.analysis.graph_builder import GraphBuilder
from air_context_engine.analysis.scanner import RepositoryScanner
from air_context_engine.analysis.sot_builder import SOTBuilder
from air_context_engine.analysis.tree_sitter_support import get_parser
from air_context_engine.benchmarks.benchmark_llm import LLMBenchmarkEngine
from air_context_engine.search.rag_engine import RAGEngine


def _write_cache_entry(root_path: str, key: str) -> None:
    manager = CacheManager(Path(root_path))
    with manager.transaction():
        manager.cache_data.setdefault("symbols", {})[key] = key
        time.sleep(0.03)


class ScannerTests(unittest.TestCase):
    def test_generated_files_and_cache_are_ignored(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "main.py").write_text("print('ok')\n", encoding="utf-8")
            (root / "BENCHMARK_REPORT.md").write_text("generated\n", encoding="utf-8")
            (root / ".air-benchmark").mkdir()
            (root / ".air-benchmark" / "run.json").write_text("{}\n", encoding="utf-8")
            (root / ".air_cache.json").write_text("{}\n", encoding="utf-8")
            (root / ".ruff_cache").mkdir()
            (root / ".ruff_cache" / "index").write_text("generated\n", encoding="utf-8")

            scanned = {path.name for path in RepositoryScanner(root).scan_files()}

            self.assertEqual(scanned, {"main.py"})

    def test_extension_and_size_policy_is_enforced_and_configurable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "small.py").write_text("x = 1\n", encoding="utf-8")
            (root / "unknown.blob").write_text("secret\n", encoding="utf-8")
            (root / "large.py").write_text("x" * 20, encoding="utf-8")

            default_files = {
                path.name
                for path in RepositoryScanner(root, max_file_size=10).scan_files()
            }
            custom_files = {
                path.name
                for path in RepositoryScanner(
                    root,
                    allowed_extensions={".blob"},
                    max_file_size=100,
                ).scan_files()
            }

            self.assertEqual(default_files, {"small.py"})
            self.assertEqual(custom_files, {"unknown.blob"})


class CacheManagerTests(unittest.TestCase):
    def test_concurrent_transactions_preserve_all_updates_and_valid_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            process_context = multiprocessing.get_context("spawn")
            processes = [
                process_context.Process(
                    target=_write_cache_entry,
                    args=(temp_dir, f"worker-{index}"),
                )
                for index in range(4)
            ]
            for process in processes:
                process.start()
            for process in processes:
                process.join(timeout=30)
                self.assertEqual(process.exitcode, 0)

            cache_path = Path(temp_dir) / ".air_cache.json"
            cache_data = json.loads(cache_path.read_text(encoding="utf-8"))

            self.assertEqual(
                set(cache_data["symbols"]),
                {"worker-0", "worker-1", "worker-2", "worker-3"},
            )
            self.assertEqual(list(Path(temp_dir).glob(".*.tmp")), [])


class GraphBuilderTests(unittest.TestCase):
    def test_qualified_and_relative_python_imports_are_preserved(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "pkg" / "sub").mkdir(parents=True)
            (root / "pkg" / "__init__.py").write_text("", encoding="utf-8")
            (root / "pkg" / "utils.py").write_text(
                "def util(): pass\n", encoding="utf-8"
            )
            (root / "pkg" / "sub" / "__init__.py").write_text("", encoding="utf-8")
            (root / "pkg" / "sub" / "helper.py").write_text(
                "from .. import utils\n",
                encoding="utf-8",
            )
            (root / "pkg" / "service.py").write_text(
                "from . import utils\nfrom .sub.helper import helper\n",
                encoding="utf-8",
            )
            builder = GraphBuilder(root, RepositoryScanner(root))

            dependencies, _ = builder.collect_relationships()
            mermaid = builder.generate_mermaid_graph()

            self.assertEqual(
                dependencies["pkg.service"],
                {"pkg.utils", "pkg.sub.helper"},
            )
            self.assertEqual(dependencies["pkg.sub.helper"], {"pkg.utils"})
            self.assertIn('["pkg.service"]', mermaid)
            self.assertIn('["pkg.sub.helper"]', mermaid)


class SOTBuilderTests(unittest.TestCase):
    def test_save_writes_file_and_does_not_invent_commands_or_domain(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "requirements.txt").write_text(
                "google-genai>=1\n", encoding="utf-8"
            )
            (root / "cli.py").write_text("def main():\n    pass\n", encoding="utf-8")
            builder = SOTBuilder(root, RepositoryScanner(root))

            output_path = builder.save()
            content = output_path.read_text(encoding="utf-8")

            self.assertTrue(output_path.exists())
            self.assertIn("Gemini LLM SDK", content)
            self.assertIn("`cli.py`", content)
            self.assertNotIn("python cli.py --help", content)
            self.assertNotIn("Hotelsense", content)
            self.assertNotIn("uvicorn app.main:app", content)
            self.assertNotIn("alembic upgrade head", content)


class RAGEngineTests(unittest.TestCase):
    def test_empty_vocabulary_repository_returns_no_results(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "numbers.txt").write_text("123 456 !!!\n", encoding="utf-8")

            engine = RAGEngine(root, RepositoryScanner(root))

            self.assertEqual(engine.search("anything"), [])
            self.assertEqual(engine.search("anything", top_k=0), [])

    def test_search_result_content_does_not_shift_source_lines(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "sample.py").write_text(
                "def calculate_total(value):\n    return value * 2\n",
                encoding="utf-8",
            )

            result = RAGEngine(root, RepositoryScanner(root)).search(
                "calculate_total", top_k=1
            )[0]

            self.assertEqual(result["start_line"], 1)
            self.assertTrue(result["content"].startswith("def calculate_total"))


class TreeSitterSupportTests(unittest.TestCase):
    def test_cpp_parser_uses_installed_grammar_package(self):
        parser = get_parser("cpp")

        self.assertIsNotNone(parser)
        tree = parser.parse(b"class Child : public Base {};")
        self.assertFalse(tree.root_node.has_error)


class FailingBenchmarkEngine(LLMBenchmarkEngine):
    def __init__(self, target_path: Path):
        self.target_path = target_path
        self.judge_model_name = "fake-judge"
        self.request_delay = 0

    def _get_raw_code_context(self) -> str:
        return ""

    def _get_macro_context(self) -> str:
        return ""

    def _get_rag_micro_context(self, question: str, top_k: int = 3) -> str:
        return ""

    def query_llm(self, prompt: str, context: str):
        return {"success": False, "error": "offline", "tokens": 0, "time_sec": 0.1}


class LLMBenchmarkTests(unittest.TestCase):
    def test_failed_requests_count_as_failed_evaluations(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            questions_path = root / "questions.json"
            questions_path.write_text(
                json.dumps(
                    [
                        {
                            "id": 1,
                            "category": "NRA",
                            "question": "Where is it?",
                            "reference_answer": "The answer is in sample.py.",
                            "expected_keywords": ["answer"],
                        }
                    ]
                ),
                encoding="utf-8",
            )

            result = FailingBenchmarkEngine(root).run_eval(questions_path)

            self.assertEqual(result["benchmark_status"], "invalid")
            self.assertIsNone(result["raw_metrics"]["nra_score"])
            self.assertIsNone(result["engine_metrics"]["nra_score"])

    def test_semantic_judge_is_authoritative_and_anchors_are_diagnostic(self):
        engine = object.__new__(LLMBenchmarkEngine)
        engine.judge_model_name = "fake-judge"
        engine.judge_answer = lambda *args: {
            "success": True,
            "passed": True,
            "score": 0.95,
            "reason": "reference ile uyumlu",
            "tokens": 10,
            "time_sec": 0.1,
        }
        question = {
            "question": "Where?",
            "reference_answer": "sample.py",
            "expected_keywords": ["sample.py"],
            "forbidden_keywords": ["other.py"],
            "match_mode": "all",
            "semantic_threshold": 0.8,
        }

        passing = engine._evaluate_answer("It is in sample.py", question)
        anchor_failure = engine._evaluate_answer("It is in other.py", question)

        self.assertTrue(passing["passed"])
        self.assertTrue(anchor_failure["passed"])
        self.assertFalse(anchor_failure["structural_passed"])
        self.assertTrue(anchor_failure["semantic_passed"])
        self.assertEqual(
            anchor_failure["method"],
            "semantic_judge_with_deterministic_diagnostics",
        )
        self.assertEqual(passing["semantic_score"], 0.95)

    def test_question_schema_requires_reference_answer(self):
        engine = object.__new__(LLMBenchmarkEngine)

        with self.assertRaisesRegex(ValueError, "reference_answer"):
            engine._validate_questions(
                [{"id": 1, "category": "NRA", "question": "Where?"}]
            )


if __name__ == "__main__":
    unittest.main()
