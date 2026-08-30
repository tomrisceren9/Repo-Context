import json
import multiprocessing
import os
import sqlite3
import tempfile
import time
import unittest
from contextlib import closing
from pathlib import Path
from typing import Sequence

from air_context_engine.analysis.scanner import RepositoryScanner
from air_context_engine.benchmarks.retrieval_benchmark import RetrievalBenchmark
from air_context_engine.search.rag_engine import RAGEngine
from air_context_engine.search.semantic_search import (
    SentenceTransformerEmbeddingProvider,
    _normalize,
)
from air_context_engine.search.search_index import (
    SQLiteSearchIndex,
    interpret_query,
    split_identifier,
    tokenize_code,
)


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "retrieval"
EVALUATION_FILE = FIXTURE_ROOT / "evaluation.json"


def _open_index(root_path: str) -> None:
    root = Path(root_path)
    RAGEngine(root, RepositoryScanner(root))


class TokenizationTests(unittest.TestCase):
    def test_snake_camel_acronym_and_qualified_names_are_split(self) -> None:
        self.assertEqual(
            split_identifier("HTTPPaymentService.validate_accessToken"),
            ["http", "payment", "service", "validate", "access", "token"],
        )
        terms = tokenize_code("payments/gateway.ts::PaymentGateway.processPayment")
        for expected in (
            "payments",
            "gateway",
            "ts",
            "payment",
            "process",
            "payments/gateway.ts",
        ):
            self.assertIn(expected, terms)

    def test_turkish_software_flow_terms_expand_to_code_vocabulary(self) -> None:
        interpretation = interpret_query(
            "Görsel kuyruğa alındıktan sonra hangi semboller paralel işler?"
        )

        self.assertTrue(interpretation.callable_intent)
        for expected in ("image", "enqueue", "parallel", "process"):
            self.assertIn(expected, interpretation.term_weights)

    def test_turkish_natural_language_is_normalized_and_explained(self) -> None:
        terms = tokenize_code("Öneri akışı arayüzde nerede?")
        self.assertIn("öneri", terms)
        self.assertIn("oneri", terms)
        self.assertIn("akisi", terms)

        interpretation = interpret_query(
            "Arayüzde kullanıcı öneri oluşturduğunda HTTP çağrısını yapan kodu bul"
        )
        self.assertTrue(interpretation.natural_language)
        self.assertTrue(interpretation.callable_intent)
        self.assertEqual(interpretation.scope_hints, ("frontend",))
        self.assertIn("recommendation", interpretation.expanded_terms)
        self.assertIn("client", interpretation.expanded_terms)
        self.assertEqual(interpretation.term_weights["recommendation"], 0.35)

    def test_direct_identifier_query_is_not_semantically_expanded(self) -> None:
        interpretation = interpret_query("PaymentGateway.processPayment")
        self.assertFalse(interpretation.natural_language)
        self.assertFalse(interpretation.callable_intent)
        self.assertEqual(interpretation.expanded_terms, ())


class PersistentIndexTests(unittest.TestCase):
    def test_only_changed_files_are_reparsed_and_deleted_files_are_removed(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = root / "first.py"
            second = root / "second.py"
            first.write_text("def alpha():\n    return 'one'\n", encoding="utf-8")
            second.write_text("def beta():\n    return 'two'\n", encoding="utf-8")

            cold = RAGEngine(root, RepositoryScanner(root))
            warm = RAGEngine(root, RepositoryScanner(root))
            self.assertEqual(cold.index_stats.parsed_files, 2)
            self.assertEqual(warm.index_stats.parsed_files, 0)
            self.assertEqual(warm.index_stats.reused_files, 2)

            time.sleep(0.002)
            first.write_text("def gamma():\n    return 'three'\n", encoding="utf-8")
            incremental = RAGEngine(root, RepositoryScanner(root))
            self.assertEqual(incremental.index_stats.parsed_files, 1)
            self.assertEqual(incremental.index_stats.reused_files, 1)
            self.assertEqual(incremental.search("alpha"), [])
            self.assertEqual(
                incremental.search("gamma", top_k=1)[0]["symbol_name"], "gamma"
            )

            second.unlink()
            deleted = RAGEngine(root, RepositoryScanner(root))
            self.assertEqual(deleted.index_stats.deleted_files, 1)
            self.assertEqual(deleted.search("beta"), [])

            cache = json.loads((root / ".air_cache.json").read_text(encoding="utf-8"))
            self.assertEqual(cache["version"], 4)
            self.assertEqual(set(cache["rag_index"]["files"]), {"first.py"})

    def test_vectors_and_metadata_are_persisted_in_sqlite(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "service.py").write_text(
                "class PaymentService:\n"
                "    def capture_charge(self, amount: float):\n"
                "        return amount\n",
                encoding="utf-8",
            )
            engine = RAGEngine(root, RepositoryScanner(root))
            first = engine.search("captureCharge", top_k=1)
            restarted = RAGEngine(root, RepositoryScanner(root)).search(
                "captureCharge", top_k=1
            )

            self.assertEqual(first[0]["id"], restarted[0]["id"])
            self.assertEqual(first[0]["symbol_name"], "PaymentService.capture_charge")
            with closing(sqlite3.connect(root / ".air_index.sqlite3")) as connection:
                chunk = connection.execute(
                    "SELECT language, symbol_type, vector_norm FROM chunks "
                    "WHERE symbol_name = ?",
                    ("PaymentService.capture_charge",),
                ).fetchone()
                term = connection.execute(
                    "SELECT tfidf_weight FROM terms WHERE term = 'capture' LIMIT 1"
                ).fetchone()
            self.assertEqual(chunk[:2], ("python", "method"))
            self.assertGreater(chunk[2], 0)
            self.assertGreater(term[0], 0)

    def test_sql_candidate_selection_excludes_nonmatching_chunks(self) -> None:
        engine = RAGEngine(FIXTURE_ROOT, RepositoryScanner(FIXTURE_ROOT))
        rows, _, _, _ = engine.index.search_rows("warehouse stock")
        self.assertTrue(rows)
        self.assertLess(len(rows), engine.index_stats.chunk_count)
        self.assertTrue(any(row["file_path"] == "inventory/stock.cs" for row in rows))

    def test_database_schema_mismatch_rebuilds_safely(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "source.py").write_text("def active(): pass\n", encoding="utf-8")
            index = SQLiteSearchIndex(root)
            with index._connect() as connection:
                index._create_schema(connection)
                connection.execute(
                    "INSERT OR REPLACE INTO meta(key, value) VALUES('schema_version', '1')"
                )
            engine = RAGEngine(root, RepositoryScanner(root))
            self.assertEqual(
                engine.search("active", top_k=1)[0]["symbol_name"], "active"
            )

    def test_concurrent_processes_preserve_index_integrity(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for index in range(12):
                (root / f"module_{index}.py").write_text(
                    f"def symbol_{index}():\n    return {index}\n",
                    encoding="utf-8",
                )
            context = multiprocessing.get_context("spawn")
            processes = [
                context.Process(target=_open_index, args=(temp_dir,)) for _ in range(3)
            ]
            for process in processes:
                process.start()
            for process in processes:
                process.join(timeout=30)
                self.assertEqual(process.exitcode, 0)

            engine = RAGEngine(root, RepositoryScanner(root))
            self.assertEqual(engine.index_stats.chunk_count, 12)
            self.assertEqual(
                engine.search("symbol_11", top_k=1)[0]["symbol_name"], "symbol_11"
            )
            with closing(sqlite3.connect(root / ".air_index.sqlite3")) as connection:
                self.assertEqual(
                    connection.execute("PRAGMA integrity_check").fetchone()[0], "ok"
                )


class HybridSearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.engine = RAGEngine(FIXTURE_ROOT, RepositoryScanner(FIXTURE_ROOT))

    def test_exact_symbol_bm25_and_tfidf_signals_are_explainable(self) -> None:
        exact = self.engine.search("PaymentGateway.processPayment", top_k=1)[0]
        semantic = self.engine.search("invoice payable amount with tax", top_k=1)[0]

        self.assertEqual(exact["symbol_name"], "PaymentGateway.processPayment")
        self.assertEqual(exact["exact_score"], 1.0)
        self.assertGreater(exact["bm25_score"], 0)
        self.assertGreater(exact["tfidf_score"], 0)
        self.assertEqual(semantic["symbol_name"], "calculate_invoice_total")
        self.assertEqual(semantic["exact_score"], 0.0)
        self.assertGreater(semantic["bm25_score"], 0)
        self.assertGreater(semantic["tfidf_score"], 0)

    def test_callable_query_prefers_production_ast_symbol_over_docs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "docs").mkdir()
            (root / "src").mkdir()
            (root / "docs" / "queue.md").write_text(
                "Kuyruk görsel işleme sembolleri paralel paralel paralel.",
                encoding="utf-8",
            )
            (root / "src" / "service.py").write_text(
                "def enqueue_image():\n    return process_image()\n\n"
                "def process_image():\n    return 'done'\n",
                encoding="utf-8",
            )
            engine = RAGEngine(root, RepositoryScanner(root))

            result = engine.search(
                "Görsel kuyruğa alındıktan sonra hangi sembol işler?", top_k=1
            )[0]

        self.assertEqual(result["file_path"], "src/service.py")
        self.assertEqual(result["source_role"], "production")

    def test_path_language_symbol_type_and_line_filters_compose(self) -> None:
        result = self.engine.search(
            "payment",
            top_k=10,
            file_paths=["payments/*"],
            languages=["typescript"],
            symbol_types=["method"],
            start_line=1,
            end_line=6,
        )
        self.assertTrue(result)
        self.assertTrue(
            all(item["file_path"].startswith("payments/") for item in result)
        )
        self.assertTrue(all(item["language"] == "typescript" for item in result))
        self.assertTrue(all(item["symbol_type"] == "method" for item in result))
        self.assertTrue(
            all(item["start_line"] <= 6 and item["end_line"] >= 1 for item in result)
        )
        self.assertEqual(
            self.engine.search(
                "payment", languages=["rust"], file_paths=["payments/*"]
            ),
            [],
        )

    def test_diversification_reduces_overlapping_parent_child_chunks(self) -> None:
        without_diversity = self.engine.search(
            "payment gateway payment",
            top_k=3,
            diversity_strength=0.0,
        )
        diversified = self.engine.search(
            "payment gateway payment",
            top_k=3,
            diversity_strength=0.8,
        )
        plain_files = {item["file_path"] for item in without_diversity}
        diverse_files = {item["file_path"] for item in diversified}
        self.assertGreaterEqual(len(diverse_files), len(plain_files))
        self.assertEqual(
            diversified[0]["symbol_name"], without_diversity[0]["symbol_name"]
        )
        self.assertEqual(without_diversity[1]["file_path"], "payments/gateway.ts")
        self.assertEqual(diversified[2]["file_path"], "shared/text.py")
        self.assertNotIn(
            "PaymentGateway",
            {item["symbol_name"] for item in diversified},
        )

    def test_invalid_line_filters_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "pozitif"):
            self.engine.search("payment", start_line=0)
        with self.assertRaisesRegex(ValueError, "büyük"):
            self.engine.search("payment", start_line=10, end_line=2)


class _FakeLocalEmbeddings:
    model_id = "test-local-embeddings-v1"

    @staticmethod
    def _vector(text: str) -> list[float]:
        return [1.0, 0.0] if "calculate_invoice_total" in text else [0.0, 1.0]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]


class LocalSemanticSearchTests(unittest.TestCase):
    def test_local_model_path_and_vector_validation_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "yerel dosya sisteminde"):
            SentenceTransformerEmbeddingProvider(Path("/definitely/missing/model"))
        with self.assertRaisesRegex(ValueError, "sıfır normlu"):
            _normalize([0.0, 0.0])

    def test_semantic_search_can_retrieve_without_lexical_overlap(self) -> None:
        engine = RAGEngine(
            FIXTURE_ROOT,
            RepositoryScanner(FIXTURE_ROOT),
            embedding_provider=_FakeLocalEmbeddings(),
            search_mode="semantic",
        )
        result = engine.search("completely unrelated wording", top_k=1)[0]
        self.assertEqual(result["symbol_name"], "calculate_invoice_total")
        self.assertGreater(result["semantic_score"], 0)
        self.assertEqual(engine.last_query_interpretation["search_mode"], "semantic")

        sync = engine.semantic_index.sync()
        self.assertEqual(sync["embedded"], 0)
        self.assertGreater(sync["reused"], 0)

    def test_hybrid_search_exposes_both_rankings(self) -> None:
        engine = RAGEngine(
            FIXTURE_ROOT,
            RepositoryScanner(FIXTURE_ROOT),
            embedding_provider=_FakeLocalEmbeddings(),
            search_mode="hybrid",
        )
        result = engine.search("invoice payable amount with tax", top_k=1)[0]
        self.assertEqual(result["symbol_name"], "calculate_invoice_total")
        self.assertIn("lexical_score", result)
        self.assertIn("semantic_score", result)
        self.assertIn("fusion_score", result)

    def test_semantic_mode_never_falls_back_without_local_model(self) -> None:
        with self.assertRaisesRegex(ValueError, "yerel bir embedding modeli"):
            RAGEngine(
                FIXTURE_ROOT,
                RepositoryScanner(FIXTURE_ROOT),
                search_mode="semantic",
            )


class RetrievalAcceptanceTests(unittest.TestCase):
    def test_fixed_evaluation_recall_at_5_and_mrr(self) -> None:
        metrics = RetrievalBenchmark(FIXTURE_ROOT, EVALUATION_FILE).evaluate(top_k=5)
        self.assertGreaterEqual(metrics["recall_at_5"], 0.90, metrics["details"])
        self.assertGreaterEqual(metrics["mrr"], 0.80, metrics["details"])
        self.assertEqual(metrics["exact_symbol"]["recall_at_5"], metrics["recall_at_5"])
        self.assertGreaterEqual(
            metrics["snippet"]["recall_at_5"], metrics["recall_at_5"]
        )
        self.assertIn("unspecified", metrics["by_query_style"])
        self.assertIn("interpretation", metrics["details"][0])

    def test_parent_chunk_covering_target_counts_as_relevant_snippet(self) -> None:
        results = [{"file_path": "service.py", "start_line": 1, "end_line": 50}]
        target = {"start_line": 12, "end_line": 20}
        expected = {"file": "service.py"}
        self.assertEqual(
            RetrievalBenchmark._snippet_rank(results, expected, target),
            1,
        )

    def test_natural_language_identifier_leakage_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            evaluation = Path(temp_dir) / "evaluation.json"
            evaluation.write_text(
                json.dumps(
                    [
                        {
                            "id": "leaked",
                            "query_style": "natural_language",
                            "query": "PaymentGateway metodunu bul",
                            "file": "payments/gateway.ts",
                            "symbol": "PaymentGateway.processPayment",
                            "forbidden_identifiers": ["PaymentGateway"],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "sızdırıyor"):
                RetrievalBenchmark(FIXTURE_ROOT, evaluation).evaluate(top_k=5)

    def test_warm_start_indexing_reduction_is_at_least_80_percent(self) -> None:
        metrics = RetrievalBenchmark(FIXTURE_ROOT, EVALUATION_FILE).measure_indexing(
            repetitions=3,
            scale_files=300,
        )
        self.assertGreaterEqual(metrics["warm_reduction"], 0.80, metrics)

    @unittest.skipIf(os.name == "nt", "Symlink davranışı Unix üzerinde doğrulanır.")
    def test_index_measurement_skips_broken_virtualenv_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "source.py").write_text(
                "def active():\n    return True\n", encoding="utf-8"
            )
            bin_dir = root / "venv" / "bin"
            bin_dir.mkdir(parents=True)
            (bin_dir / "python").symlink_to(root / "missing-python")

            metrics = RetrievalBenchmark(root, EVALUATION_FILE).measure_indexing(
                repetitions=1,
                scale_files=0,
            )
            self.assertEqual(metrics["repetitions"], 1)


if __name__ == "__main__":
    unittest.main()
