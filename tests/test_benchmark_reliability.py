import copy
import tempfile
import unittest
from pathlib import Path

from air_context_engine.benchmarks.benchmark_llm import (
    LLMBenchmarkEngine,
    _contains_symbol,
    _sample_summary,
)
from air_context_engine.benchmarks.benchmark_schema import (
    load_questions,
    validate_questions,
)
from air_context_engine.benchmarks.question_preflight import (
    validate_questions_for_project,
)
from air_context_engine.benchmarks.report_generator import (
    compare_benchmark_runs,
    load_benchmark_artifact,
    write_benchmark_artifacts,
)


class QuestionValidationTests(unittest.TestCase):
    def test_fixture_passes_schema_and_project_preflight(self) -> None:
        fixture = Path(__file__).parent / "fixtures" / "benchmark"
        questions = load_questions(fixture / "valid_questions.json")

        self.assertEqual(len(questions), 1)
        report = validate_questions_for_project(fixture, questions)
        self.assertTrue(report["valid"])
        self.assertEqual(report["checked_symbol_count"], 1)

    def test_duplicate_ids_and_invalid_regex_are_rejected(self) -> None:
        question = {
            "id": "duplicate",
            "category": "NRA",
            "question": "Which symbol?",
            "reference_answer": "Service.run",
            "expected_symbols": ["Service.run"],
            "expected_patterns": ["["],
        }
        with self.assertRaises(ValueError):
            validate_questions([question, copy.deepcopy(question)])

    def test_mismatched_positive_symbol_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "service.py").write_text(
                "class ExistingService:\n    pass\n",
                encoding="utf-8",
            )
            report = validate_questions_for_project(
                root,
                [
                    {
                        "id": "missing",
                        "category": "NRA",
                        "question": "Which service?",
                        "reference_answer": "MissingService",
                        "expected_symbols": ["MissingService"],
                    }
                ],
            )

        self.assertFalse(report["valid"])
        self.assertEqual(report["issues"][0]["problem"], "expected_symbol_not_found")


class EvaluationTests(unittest.TestCase):
    def test_qualified_symbols_match_without_substring_false_positives(self) -> None:
        answer = "The call is handled by BillingService.calculate_total."
        self.assertTrue(_contains_symbol(answer, "BillingService.calculate_total"))
        self.assertFalse(_contains_symbol(answer, "BillingService.calculate_tot"))

    def test_confidence_interval_requires_three_repetitions(self) -> None:
        two = _sample_summary([50.0, 70.0])
        three = _sample_summary([50.0, 70.0, 90.0])

        self.assertFalse(two["ci_available"])
        self.assertIsNone(two["ci95_low"])
        self.assertTrue(three["ci_available"])
        self.assertEqual(three["ci_method"], "student_t")

    def test_candidate_prompt_is_bounded_and_evidence_based(self) -> None:
        prompt = LLMBenchmarkEngine._candidate_prompt("Question?", "context")
        self.assertIn("Bağlamda kanıt yoksa", prompt)
        self.assertIn("220 kelime", prompt)
        self.assertIn("Question?", prompt)

    def test_judge_json_parser_accepts_fence_and_rejects_arrays(self) -> None:
        parsed = LLMBenchmarkEngine._parse_json_response(
            '```json\n{"passed": true}\n```'
        )
        self.assertTrue(parsed["passed"])
        with self.assertRaisesRegex(ValueError, "JSON nesnesi"):
            LLMBenchmarkEngine._parse_json_response("[]")

    def test_only_local_provider_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "Desteklenmeyen"):
                LLMBenchmarkEngine(Path(temp_dir), provider_name="cloud")


class ArtifactTests(unittest.TestCase):
    @staticmethod
    def _artifact(run_id: str, model: str = "qwen3:4b") -> dict:
        metrics = {
            "overall_score": 100.0,
            "nra_score": 100.0,
            "aas_score": None,
            "zhr_score": None,
            "retrieval_score": 100.0,
            "structural_score": 100.0,
            "semantic_score": None,
            "total_tokens": 10,
            "input_tokens": 8,
            "cached_tokens": 0,
            "output_tokens": 2,
            "thought_tokens": 0,
            "avg_time_sec": 0.1,
            "api_error_count": 0,
            "truncated_call_count": 0,
            "judge_error_count": 0,
            "observed_cost_usd": 0.0,
            "total_cost_usd": 0.0,
            "cost_complete": True,
            "resolved_model_versions": [model],
            "score_statistics": {},
        }
        return {
            "schema_version": "1.7.0",
            "run_id": run_id,
            "experiment_key": "experiment",
            "started_at": "2026-01-01T00:00:00.000Z",
            "completed_at": "2026-01-01T00:00:01.000Z",
            "benchmark_status": "valid",
            "scores_valid": True,
            "invalid_reasons": [],
            "config": {
                "provider": "local",
                "provider_base_url": "http://127.0.0.1:11434/v1",
                "model": model,
                "temperature": 0.0,
                "repetitions": 3,
            },
            "repository": {"commit": "abc", "dirty": False},
            "question_set": {"path": "questions.json", "sha256": "q"},
            "context_sha256": "c",
            "raw_metrics": copy.deepcopy(metrics),
            "engine_metrics": copy.deepcopy(metrics),
            "api_error_count": 0,
            "judge_error_count": 0,
            "truncated_call_count": 0,
            "token_savings_percent": 0.0,
            "details": [],
        }

    def test_json_and_markdown_are_written_from_the_same_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            artifact = self._artifact("run-a")
            paths = write_benchmark_artifacts(artifact, output)
            loaded = load_benchmark_artifact(paths["json"])

            self.assertEqual(loaded["run_id"], "run-a")
            self.assertTrue(paths["markdown"].is_file())
            self.assertIn("run-a", paths["markdown"].read_text(encoding="utf-8"))

    def test_comparison_reports_changed_model(self) -> None:
        baseline = self._artifact("baseline")
        current = self._artifact("current", model="qwen3:8b")
        comparison = compare_benchmark_runs(current, baseline)

        self.assertIn("model", comparison["changed_dimensions"])


if __name__ == "__main__":
    unittest.main()
