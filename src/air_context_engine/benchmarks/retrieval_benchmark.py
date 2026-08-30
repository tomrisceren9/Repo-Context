from __future__ import annotations

import json
import shutil
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any

from ..analysis.scanner import RepositoryScanner
from ..search.rag_engine import RAGEngine


class RetrievalBenchmark:
    """Sabit relevance seti üzerinde Recall@k, MRR ve warm-index hızını ölçer."""

    def __init__(self, fixture_root: Path, evaluation_file: Path) -> None:
        self.fixture_root = fixture_root.absolute()
        self.evaluation_file = evaluation_file.absolute()

    def _load_evaluation(self) -> list[dict[str, Any]]:
        data = json.loads(self.evaluation_file.read_text(encoding="utf-8"))
        if not isinstance(data, list) or not data:
            raise ValueError("Retrieval değerlendirme seti boş veya geçersiz.")
        required = {"query", "file", "symbol"}
        seen_ids: set[str] = set()
        for index, item in enumerate(data, 1):
            if not isinstance(item, dict) or not required <= set(item):
                raise ValueError(
                    "Her retrieval kaydı query, file ve symbol içermelidir."
                )
            if any(
                not isinstance(item[field], str) or not item[field].strip()
                for field in required
            ):
                raise ValueError(
                    f"Retrieval kaydı {index} için query/file/symbol dolu metin olmalıdır."
                )
            item_id = item.get("id", f"query-{index}")
            if (
                not isinstance(item_id, str)
                or not item_id.strip()
                or item_id in seen_ids
            ):
                raise ValueError(f"Retrieval kaydı {index} için id boş veya tekrarlı.")
            seen_ids.add(item_id)
            item["id"] = item_id
            query_style = item.get("query_style", "unspecified")
            if not isinstance(query_style, str) or not query_style.strip():
                raise ValueError(
                    f"Retrieval kaydı {item_id} için query_style geçersiz."
                )
            item["query_style"] = query_style
            forbidden = item.get("forbidden_identifiers", [])
            if not isinstance(forbidden, list) or not all(
                isinstance(value, str) and value.strip() for value in forbidden
            ):
                raise ValueError(
                    f"Retrieval kaydı {item_id} için forbidden_identifiers geçersiz."
                )
            normalized_query = item["query"].casefold()
            leaked = [
                value for value in forbidden if value.casefold() in normalized_query
            ]
            if leaked:
                raise ValueError(
                    f"Retrieval kaydı {item_id} hedef sembolü sorguya sızdırıyor: {leaked}"
                )
        return data

    @staticmethod
    def _rank(results: list[dict[str, Any]], expected: dict[str, Any]) -> int | None:
        for rank, result in enumerate(results, 1):
            if (
                result["file_path"] == expected["file"]
                and result["symbol_name"] == expected["symbol"]
            ):
                return rank
        return None

    @staticmethod
    def _snippet_rank(
        results: list[dict[str, Any]],
        expected: dict[str, Any],
        target: dict[str, Any],
    ) -> int | None:
        """Hedef sembolün tüm satırlarını içeren sonuç parçasının sırasını bulur."""
        for rank, result in enumerate(results, 1):
            if (
                result["file_path"] == expected["file"]
                and int(result["start_line"]) <= int(target["start_line"])
                and int(result["end_line"]) >= int(target["end_line"])
            ):
                return rank
        return None

    @staticmethod
    def _summary(ranks: list[int | None], top_k: int) -> dict[str, float]:
        cutoffs = sorted({1, min(3, top_k), top_k})
        summary = {
            f"recall_at_{cutoff}": round(
                sum(rank is not None and rank <= cutoff for rank in ranks) / len(ranks),
                6,
            )
            for cutoff in cutoffs
        }
        summary["mrr"] = round(
            sum(1.0 / rank for rank in ranks if rank is not None) / len(ranks),
            6,
        )
        return summary

    def evaluate(self, top_k: int = 5) -> dict[str, Any]:
        evaluation = self._load_evaluation()
        engine = RAGEngine(self.fixture_root, RepositoryScanner(self.fixture_root))
        ranks: list[int | None] = []
        snippet_ranks: list[int | None] = []
        details: list[dict[str, Any]] = []
        for item in evaluation:
            targets = engine.index.symbol_locations(item["file"], item["symbol"])
            if not targets:
                raise ValueError(
                    f"Retrieval hedefi indekste bulunamadı: {item['file']}::{item['symbol']}"
                )
            target = targets[0]
            results = engine.search(item["query"], top_k=top_k)
            rank = self._rank(results, item)
            snippet_rank = self._snippet_rank(results, item, target)
            ranks.append(rank)
            snippet_ranks.append(snippet_rank)
            details.append(
                {
                    **item,
                    "rank": rank,
                    "exact_symbol_rank": rank,
                    "snippet_rank": snippet_rank,
                    "target_span": {
                        "start_line": target["start_line"],
                        "end_line": target["end_line"],
                    },
                    "interpretation": engine.last_query_interpretation,
                    "returned": [
                        {
                            "file": result["file_path"],
                            "symbol": result["symbol_name"],
                            "start_line": result["start_line"],
                            "end_line": result["end_line"],
                            "score": result["score"],
                        }
                        for result in results
                    ],
                }
            )
        exact_summary = self._summary(ranks, top_k)
        snippet_summary = self._summary(snippet_ranks, top_k)
        by_query_style: dict[str, Any] = {}
        for style in sorted({str(item["query_style"]) for item in evaluation}):
            selected = [
                index
                for index, item in enumerate(evaluation)
                if item["query_style"] == style
            ]
            by_query_style[style] = {
                "queries": len(selected),
                "exact_symbol": self._summary(
                    [ranks[index] for index in selected], top_k
                ),
                "snippet": self._summary(
                    [snippet_ranks[index] for index in selected], top_k
                ),
            }
        result = {
            "queries": len(ranks),
            **exact_summary,
            "exact_symbol": exact_summary,
            "snippet": snippet_summary,
            "by_query_style": by_query_style,
            "details": details,
        }
        return result

    @staticmethod
    def _add_scale_fixture(root: Path, file_count: int) -> None:
        """Performans ölçümünü zamanlayıcı gürültüsünden çıkaran deterministik büyük fixture."""
        scale_root = root / "scale"
        scale_root.mkdir()
        for index in range(file_count):
            (scale_root / f"module_{index:04d}.py").write_text(
                f"class GeneratedService{index}:\n"
                f"    def transform_payload_{index}(self, value: str) -> str:\n"
                f"        return value + '-{index}'\n",
                encoding="utf-8",
            )

    def measure_indexing(
        self,
        repetitions: int = 3,
        scale_files: int = 300,
    ) -> dict[str, Any]:
        if repetitions <= 0:
            raise ValueError("repetitions pozitif olmalıdır")
        if scale_files < 0:
            raise ValueError("scale_files negatif olamaz")
        cold_samples: list[float] = []
        warm_samples: list[float] = []
        with tempfile.TemporaryDirectory() as temp_dir:
            benchmark_root = Path(temp_dir) / "repository"
            benchmark_root.mkdir()
            for source in RepositoryScanner(self.fixture_root).scan_files():
                destination = benchmark_root / source.relative_to(self.fixture_root)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
            self._add_scale_fixture(benchmark_root, scale_files)
            for _ in range(repetitions):
                for artifact in (
                    ".air_index.sqlite3",
                    ".air_index.sqlite3-shm",
                    ".air_index.sqlite3-wal",
                    ".air_index.sqlite3.lock",
                    ".air_cache.json",
                    ".air_cache.json.lock",
                ):
                    (benchmark_root / artifact).unlink(missing_ok=True)
                started = time.perf_counter()
                cold = RAGEngine(benchmark_root, RepositoryScanner(benchmark_root))
                cold_samples.append(time.perf_counter() - started)
                started = time.perf_counter()
                warm = RAGEngine(benchmark_root, RepositoryScanner(benchmark_root))
                warm_samples.append(time.perf_counter() - started)
                if (
                    cold.index_stats.parsed_files == 0
                    or warm.index_stats.parsed_files != 0
                ):
                    raise AssertionError(
                        "Cold/warm indeks yaşam döngüsü beklenen durumda değil."
                    )
        cold_median = statistics.median(cold_samples)
        warm_median = statistics.median(warm_samples)
        reduction = (cold_median - warm_median) / cold_median if cold_median else 0.0
        return {
            "repetitions": repetitions,
            "scale_files": scale_files,
            "cold_median_seconds": cold_median,
            "warm_median_seconds": warm_median,
            "warm_reduction": round(reduction, 6),
            "warm_reduction_percent": round(reduction * 100, 2),
        }

    def run(
        self,
        top_k: int = 5,
        repetitions: int = 3,
        scale_files: int = 300,
    ) -> dict[str, Any]:
        return {
            "retrieval": self.evaluate(top_k),
            "indexing": self.measure_indexing(repetitions, scale_files),
        }
