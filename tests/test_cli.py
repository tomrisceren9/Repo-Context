import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from air_context_engine.cli import main


class CLIErrorContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = CliRunner()

    def assert_structured_error(
        self,
        args: list[str],
        expected_code: str,
        expected_exit_code: int,
    ) -> dict[str, object]:
        result = self.runner.invoke(main, args)
        self.assertEqual(result.exit_code, expected_exit_code, result.output)
        payload = json.loads(result.stderr.strip())
        self.assertEqual(payload["error"]["code"], expected_code)
        self.assertIsInstance(payload["error"]["message"], str)
        self.assertIsInstance(payload["error"]["details"], dict)
        return payload

    def test_missing_target_path_is_structured_and_nonzero(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            missing = Path(temp_dir) / "missing"
            payload = self.assert_structured_error(
                ["scan", "--path", str(missing)],
                "PATH_NOT_FOUND",
                2,
            )
            self.assertEqual(
                payload["error"]["details"]["path"], str(missing.absolute())
            )

    def test_target_file_is_rejected_as_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            target_file = Path(temp_dir) / "target.py"
            target_file.write_text("VALUE = 1\n", encoding="utf-8")
            self.assert_structured_error(
                ["scan", "--path", str(target_file)],
                "PATH_NOT_DIRECTORY",
                2,
            )

    def test_invalid_top_k_is_structured_and_nonzero(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            self.assert_structured_error(
                ["query", "anything", "--path", temp_dir, "--top-k", "0"],
                "INVALID_TOP_K",
                2,
            )

    def test_click_usage_errors_follow_structured_contract(self) -> None:
        self.assert_structured_error(
            ["query", "prompt", "--top-k", "not-an-integer"],
            "CLI_USAGE_ERROR",
            2,
        )

    def test_missing_question_file_is_structured_and_nonzero(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            missing = str(Path(temp_dir) / "questions.json")
            self.assert_structured_error(
                ["benchmark-llm", "--path", temp_dir, "--questions", missing],
                "QUESTION_FILE_NOT_FOUND",
                2,
            )

    def test_invalid_questions_are_rejected_before_optional_api_import(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            questions = Path(temp_dir) / "invalid.json"
            questions.write_text('[{"id": 1, "category": "NRA"}]', encoding="utf-8")
            with patch(
                "air_context_engine.cli._load_llm_benchmark",
                side_effect=AssertionError("API yüklenmemeli"),
            ):
                payload = self.assert_structured_error(
                    [
                        "benchmark-llm",
                        "--path",
                        temp_dir,
                        "--questions",
                        str(questions),
                    ],
                    "QUESTION_SCHEMA_INVALID",
                    2,
                )
            self.assertIn("question", str(payload["error"]["details"]))

    def test_validate_questions_command_needs_no_api_key(self) -> None:
        fixture = (
            Path(__file__).parent / "fixtures" / "benchmark" / "valid_questions.json"
        )
        result = self.runner.invoke(main, ["validate-questions", str(fixture)])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn('"valid": true', result.output)
        self.assertIn('"question_count": 1', result.output)

    def test_project_question_validation_accepts_matching_positive_and_zhr_anchors(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "service.py").write_text(
                "class BillingService:\n"
                "    def calculate_total(self):\n"
                "        return 1\n",
                encoding="utf-8",
            )
            questions = root / "questions.json"
            questions.write_text(
                json.dumps(
                    [
                        {
                            "id": "positive",
                            "category": "NRA",
                            "question": "Hangi sembol?",
                            "reference_answer": "BillingService.calculate_total",
                            "expected_symbols": ["BillingService.calculate_total"],
                        },
                        {
                            "id": "negative",
                            "category": "ZHR",
                            "question": "MissingGateway var mı?",
                            "reference_answer": "Hayır.",
                            "expected_symbols": ["MissingGateway"],
                        },
                    ]
                ),
                encoding="utf-8",
            )

            result = self.runner.invoke(
                main,
                ["validate-questions", str(questions), "--path", temp_dir],
            )

            self.assertEqual(result.exit_code, 0, result.output)
            self.assertIn('"valid": true', result.output)
            self.assertIn('"checked_symbol_count": 2', result.output)

    def test_project_mismatch_is_rejected_before_api_or_optional_sdk(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "service.py").write_text(
                "class ExistingService: pass\n", encoding="utf-8"
            )
            questions = root / "questions.json"
            questions.write_text(
                json.dumps(
                    [
                        {
                            "id": "wrong-project",
                            "category": "NRA",
                            "question": "Hangi servis?",
                            "reference_answer": "OrderService",
                            "expected_symbols": ["OrderService"],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            with patch(
                "air_context_engine.cli._load_llm_benchmark",
                side_effect=AssertionError("SDK yüklenmemeli"),
            ):
                payload = self.assert_structured_error(
                    [
                        "benchmark-llm",
                        "--path",
                        temp_dir,
                        "--questions",
                        str(questions),
                    ],
                    "QUESTION_PROJECT_MISMATCH",
                    2,
                )

            issues = payload["error"]["details"]["issues"]
            self.assertEqual(issues[0]["symbol"], "OrderService")
            self.assertEqual(issues[0]["problem"], "expected_symbol_not_found")

    def test_zhr_question_is_rejected_when_symbol_exists_in_project(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "gateway.py").write_text(
                "class QuantumPaymentGateway: pass\n", encoding="utf-8"
            )
            questions = root / "questions.json"
            questions.write_text(
                json.dumps(
                    [
                        {
                            "id": "invalid-zhr",
                            "category": "ZHR",
                            "question": "Sınıf var mı?",
                            "reference_answer": "Hayır.",
                            "expected_symbols": ["QuantumPaymentGateway"],
                        }
                    ]
                ),
                encoding="utf-8",
            )

            payload = self.assert_structured_error(
                ["validate-questions", str(questions), "--path", temp_dir],
                "QUESTION_PROJECT_MISMATCH",
                2,
            )

            issue = payload["error"]["details"]["issues"][0]
            self.assertEqual(issue["problem"], "zhr_symbol_present_in_project")

    def test_partial_benchmark_pricing_is_rejected_before_api_import(self) -> None:
        fixture = (
            Path(__file__).parent / "fixtures" / "benchmark" / "valid_questions.json"
        )
        with patch(
            "air_context_engine.cli._load_llm_benchmark",
            side_effect=AssertionError("API yüklenmemeli"),
        ):
            self.assert_structured_error(
                [
                    "benchmark-llm",
                    "--path",
                    str(fixture.parent),
                    "--questions",
                    str(fixture),
                    "--input-cost-per-million",
                    "1.0",
                ],
                "INVALID_PRICING",
                2,
            )

    def test_remote_endpoint_permission_is_required_before_engine_import(self) -> None:
        fixture = (
            Path(__file__).parent / "fixtures" / "benchmark" / "valid_questions.json"
        )
        with (
            patch.dict(os.environ, {"AIR_CONTEXT_ALLOW_REMOTE_LLM": ""}),
            patch(
                "air_context_engine.cli._load_llm_benchmark",
                side_effect=AssertionError("SDK yüklenmemeli"),
            ),
        ):
            self.assert_structured_error(
                [
                    "benchmark-llm",
                    "--path",
                    str(fixture.parent),
                    "--questions",
                    str(fixture),
                    "--provider-base-url",
                    "https://llm.example/v1",
                ],
                "REMOTE_LLM_PERMISSION_REQUIRED",
                3,
            )

    def test_local_options_are_forwarded_and_provider_is_closed(self) -> None:
        fixture = (
            Path(__file__).parent / "fixtures" / "benchmark" / "valid_questions.json"
        )
        captured: dict[str, object] = {}

        class FakeEngine:
            def __init__(self, target_path: Path, **kwargs: object) -> None:
                captured["target_path"] = target_path
                captured["kwargs"] = kwargs

            def run_eval(self, questions_path: Path) -> dict[str, object]:
                captured["questions_path"] = questions_path
                metrics = {
                    "total_tokens": 10,
                    "avg_time_sec": 0.1,
                    "nra_score": 100.0,
                    "aas_score": 0.0,
                    "zhr_score": 0.0,
                    "api_error_count": 0,
                    "truncated_call_count": 0,
                    "observed_cost_usd": None,
                    "cost_complete": False,
                }
                return {
                    "raw_metrics": metrics,
                    "engine_metrics": dict(metrics),
                    "token_savings_percent": 0.0,
                }

            def close(self) -> None:
                captured["closed"] = True

        with (
            patch(
                "air_context_engine.cli._load_llm_benchmark",
                return_value=(FakeEngine, object),
            ),
            patch(
                "air_context_engine.cli.write_benchmark_artifacts",
                return_value={"json": Path("run.json"), "markdown": Path("run.md")},
            ),
        ):
            result = self.runner.invoke(
                main,
                [
                    "benchmark-llm",
                    "--path",
                    str(fixture.parent),
                    "--questions",
                    str(fixture),
                    "--provider-base-url",
                    "http://127.0.0.1:11434/v1",
                    "--model",
                    "qwen3:4b",
                    "--max-output-tokens",
                    "384",
                ],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        kwargs = captured["kwargs"]
        self.assertIsInstance(kwargs, dict)
        self.assertEqual(kwargs["provider_name"], "local")
        self.assertEqual(kwargs["provider_base_url"], "http://127.0.0.1:11434/v1")
        self.assertEqual(kwargs["model_name"], "qwen3:4b")
        self.assertEqual(kwargs["repetitions"], 3)
        self.assertEqual(kwargs["request_delay"], 0.0)
        self.assertEqual(kwargs["max_output_tokens"], 384)
        self.assertEqual(kwargs["reasoning_effort"], "none")
        self.assertTrue(captured["closed"])
        self.assertIn("Giriş Tokenı", result.output)
        self.assertIn("Cache'lenen Giriş", result.output)
        self.assertIn("Çıkış Tokenı", result.output)

    def test_runtime_exception_is_structured_and_nonzero(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch(
                "air_context_engine.cli.RepositoryScanner",
                side_effect=OSError("permission denied"),
            ):
                payload = self.assert_structured_error(
                    ["scan", "--path", temp_dir],
                    "SCAN_FAILED",
                    1,
                )
            self.assertEqual(payload["error"]["details"]["exception"], "OSError")

    def test_empty_query_result_is_a_successful_business_result(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = self.runner.invoke(
                main,
                ["query", "unmatched", "--path", temp_dir],
            )
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertIn("Alakalı bir kod parçası bulunamadı", result.output)

    def test_index_command_reports_warm_incremental_state(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "source.py").write_text(
                "def searchable(): pass\n", encoding="utf-8"
            )
            cold = self.runner.invoke(main, ["index", "--path", temp_dir])
            warm = self.runner.invoke(main, ["index", "--path", temp_dir])
            self.assertEqual(cold.exit_code, 0, cold.output)
            self.assertEqual(warm.exit_code, 0, warm.output)
            self.assertIn('"parsed_files": 1', cold.output)
            self.assertIn('"parsed_files": 0', warm.output)
            self.assertIn('"warm": true', warm.output)

    def test_query_filters_and_hybrid_scores_are_visible(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "service.py").write_text(
                "class BillingService:\n"
                "    def calculate_total(self):\n"
                "        return 1\n",
                encoding="utf-8",
            )
            result = self.runner.invoke(
                main,
                [
                    "query",
                    "calculateTotal",
                    "--path",
                    temp_dir,
                    "--language",
                    "python",
                    "--symbol-type",
                    "method",
                    "--file",
                    "service.py",
                    "--top-k",
                    "1",
                ],
            )
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertIn("BillingService.calculate_total", result.output)
            self.assertIn("exact=1.0", result.output)
            self.assertIn("bm25=", result.output)
            self.assertIn("tfidf=", result.output)
            self.assertIn("metadata=", result.output)
            self.assertIn("callable=", result.output)
            self.assertIn("scope=", result.output)

    def test_build_produces_the_three_complementary_context_documents(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            package = root / "app"
            package.mkdir()
            (package / "__init__.py").write_text("", encoding="utf-8")
            (package / "service.py").write_text(
                "class BookingService:\n"
                "    async def reserve(self, room_id: int) -> bool:\n"
                "        return room_id > 0\n",
                encoding="utf-8",
            )
            (package / "main.py").write_text(
                "from .service import BookingService\n\n"
                "def create_app():\n"
                "    return BookingService()\n",
                encoding="utf-8",
            )

            result = self.runner.invoke(main, ["build", "--path", temp_dir])

            self.assertEqual(result.exit_code, 0, result.output)
            expected = {
                "CLAUDE.md",
                "repo_map.md",
                "dependency_graph.md",
            }
            for name in expected:
                self.assertTrue((root / name).is_file(), name)
                self.assertIn(name, result.output)
            self.assertIn("  main.py", (root / "CLAUDE.md").read_text(encoding="utf-8"))
            self.assertIn(
                "async def reserve",
                (root / "repo_map.md").read_text(encoding="utf-8"),
            )
            self.assertIn(
                "app.main",
                (root / "dependency_graph.md").read_text(encoding="utf-8"),
            )


if __name__ == "__main__":
    unittest.main()
