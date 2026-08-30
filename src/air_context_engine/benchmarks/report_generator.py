"""JSON and Markdown artifacts for reproducible benchmark runs."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .benchmark_static import StaticBenchmarkEngine
from .benchmark_schema import load_questions
from .question_preflight import validate_questions_for_project


def load_benchmark_artifact(path: Path) -> dict[str, Any]:
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Benchmark artefaktı okunamadı: {path}: {exc}") from exc
    required = {
        "schema_version",
        "run_id",
        "experiment_key",
        "raw_metrics",
        "engine_metrics",
    }
    if not isinstance(artifact, dict) or not required.issubset(artifact):
        raise ValueError(f"Geçersiz benchmark artefaktı: {path}")
    return artifact


def compare_benchmark_runs(
    current: dict[str, Any], previous: dict[str, Any]
) -> dict[str, Any]:
    """Compare decision metrics and surface experimental configuration drift."""
    config_fields = (
        "provider",
        "provider_base_url",
        "model",
        "judge_model",
        "candidate_prompt_version",
        "judge_prompt_version",
        "temperature",
        "repetitions",
        "max_output_tokens",
        "reasoning_effort",
        "search_mode",
        "retrieval_top_k",
        "request_delay_sec",
        "max_retries",
        "retry_base_delay_sec",
        "embedding_model_path",
        "input_cost_per_million_usd",
        "output_cost_per_million_usd",
    )
    current_config = current.get("config", {})
    previous_config = previous.get("config", {})
    changed = {
        field: {
            "previous": previous_config.get(field),
            "current": current_config.get(field),
        }
        for field in config_fields
        if previous_config.get(field) != current_config.get(field)
    }
    for label, path in (
        ("git_commit", ("repository", "commit")),
        ("git_dirty", ("repository", "dirty")),
        ("question_set_sha256", ("question_set", "sha256")),
        ("context_sha256", ("context_sha256",)),
    ):

        def nested(source: dict[str, Any]) -> Any:
            value: Any = source
            for key in path:
                value = value.get(key) if isinstance(value, dict) else None
            return value

        old, new = nested(previous), nested(current)
        if old != new:
            changed[label] = {"previous": old, "current": new}

    for variant in ("raw", "engine"):
        for field in ("resolved_model_versions", "resolved_judge_model_versions"):
            old = previous.get(f"{variant}_metrics", {}).get(field, [])
            new = current.get(f"{variant}_metrics", {}).get(field, [])
            if old != new:
                changed[f"{variant}_{field}"] = {"previous": old, "current": new}

    metrics: dict[str, Any] = {}
    for variant in ("raw", "engine"):
        old_metrics = previous.get(f"{variant}_metrics", {})
        new_metrics = current.get(f"{variant}_metrics", {})
        old_overall = (
            old_metrics.get("score_statistics", {}).get("overall", {}).get("mean", 0.0)
        )
        new_overall = (
            new_metrics.get("score_statistics", {}).get("overall", {}).get("mean", 0.0)
        )
        metrics[variant] = {
            "overall_score_delta_points": round(
                float(new_overall) - float(old_overall), 4
            ),
            "total_tokens_delta": int(new_metrics.get("total_tokens", 0))
            - int(old_metrics.get("total_tokens", 0)),
            "avg_time_sec_delta": round(
                float(new_metrics.get("avg_time_sec", 0))
                - float(old_metrics.get("avg_time_sec", 0)),
                4,
            ),
            "api_error_count_delta": int(new_metrics.get("api_error_count", 0))
            - int(old_metrics.get("api_error_count", 0)),
            "judge_error_count_delta": int(new_metrics.get("judge_error_count", 0))
            - int(old_metrics.get("judge_error_count", 0)),
            "total_cost_usd_delta": (
                round(
                    float(new_metrics["total_cost_usd"])
                    - float(old_metrics["total_cost_usd"]),
                    8,
                )
                if new_metrics.get("total_cost_usd") is not None
                and old_metrics.get("total_cost_usd") is not None
                else None
            ),
        }
    return {
        "baseline_run_id": previous["run_id"],
        "current_run_id": current["run_id"],
        "same_experiment": current["experiment_key"] == previous["experiment_key"],
        "changed_dimensions": changed,
        "metrics": metrics,
    }


def _money(value: Any) -> str:
    return "hesaplanmadı" if value is None else f"${float(value):.8f}"


def render_markdown(artifact: dict[str, Any]) -> str:
    raw = artifact["raw_metrics"]
    engine = artifact["engine_metrics"]
    repository = artifact.get("repository", {})
    config = artifact.get("config", {})
    question_set = artifact.get("question_set", {})
    question_validation = artifact.get("question_project_validation", {})
    static = artifact.get("static_metrics")
    benchmark_invalid = artifact.get("benchmark_status") == "invalid"

    def score(value: Any, *, judge_metric: bool = False) -> str:
        if value is not None:
            return f"%{value}"
        if benchmark_invalid:
            return "hesaplanmadı — altyapı hatası"
        return "judge kapalı" if judge_metric else "hesaplanmadı"

    lines = [
        "# Air Context Engine Benchmark Report",
        "",
        "## İzlenebilirlik",
        "",
        "| Alan | Değer |",
        "| :--- | :--- |",
        f"| Run ID | `{artifact.get('run_id')}` |",
        f"| Koşu durumu | `{artifact.get('benchmark_status', 'unknown')}` |",
        f"| Experiment key | `{artifact.get('experiment_key')}` |",
        f"| Başlangıç / bitiş | `{artifact.get('started_at')}` / `{artifact.get('completed_at')}` |",
        f"| Commit | `{repository.get('commit') or 'git-dışı'}` (dirty: `{repository.get('dirty')}`) |",
        f"| Soru seti | `{question_set.get('path')}` / `{question_set.get('sha256')}` |",
        f"| Bağlam hash'i | `{artifact.get('context_sha256')}` |",
        f"| Sağlayıcı / API kökü | `{config.get('provider', 'local')}` / `{config.get('provider_base_url') or '-'}` |",
        f"| Model / judge | `{config.get('model')}` / `{config.get('judge_model') or 'kapalı'}` |",
        f"| Çözülen model sürümleri (raw / engine) | `{raw.get('resolved_model_versions', [])}` / `{engine.get('resolved_model_versions', [])}` |",
        f"| Prompt sürümü | `{config.get('candidate_prompt_version')}` / `{config.get('judge_prompt_version') or '-'}` |",
        f"| Sıcaklık / tekrar | `{config.get('temperature')}` / `{config.get('repetitions')}` |",
        f"| Azami çıktı tokenı | `{config.get('max_output_tokens') or '-'}` |",
        f"| Reasoning eforu | `{config.get('reasoning_effort') or '-'}` |",
        f"| Retrieval modu / embedding | `{config.get('search_mode', 'lexical')}` / `{config.get('embedding_model_path') or '-'}` |",
        f"| Retrieval top-k | `{config.get('retrieval_top_k', 3)}` |",
        f"| İstek aralığı / retry | `{config.get('request_delay_sec', 0)}` sn / `{config.get('max_retries', 0)}` kez, taban `{config.get('retry_base_delay_sec', 0)}` sn |",
        f"| Milyon-token fiyatı (giriş / çıkış) | `{config.get('input_cost_per_million_usd')}` / `{config.get('output_cost_per_million_usd')}` USD |",
        "",
    ]
    if benchmark_invalid:
        lines.extend(
            [
                "## Geçersiz koşu",
                "",
                "Bu koşudaki semantic ve overall kalite skorları altyapı hataları nedeniyle hesaplanmamıştır.",
                "",
            ]
        )
        for reason in artifact.get("invalid_reasons", []):
            lines.append(
                f"- `{reason.get('variant')}` / `{reason.get('type')}`: "
                f"{reason.get('count')} çağrı"
            )
        lines.append("")
    if question_validation:
        lines.extend(
            [
                "## Soru seti preflight",
                "",
                f"- Sonuç: `{'geçti' if question_validation.get('valid') else 'başarısız'}`",
                f"- Doğrulanan sembol: `{question_validation.get('checked_symbol_count', 0)}`",
                f"- Hata / uyarı: `{len(question_validation.get('issues', []))}` / `{question_validation.get('warning_count', 0)}`",
            ]
        )
        for warning in question_validation.get("warnings", []):
            lines.append(
                f"- Uyarı `{warning.get('question_id')}` / `{warning.get('problem')}`: "
                f"{warning.get('message', '')}"
            )
        lines.append("")
    if static:
        lines.extend(
            [
                "## Statik bağlam metrikleri",
                "",
                "| Dosya | Ham token | Bağlam token | CCR | CDS | FIC | SRR | CGC | Kapı |",
                "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |",
                f"| {static.get('total_files', 0)} | {static.get('raw_tokens', 0)} | {static.get('context_tokens', 0)} | %{static.get('ccr_compression_ratio', 0)} | %{static.get('cds_density_score', 0)} | %{static.get('fic_file_inventory_coverage', 0)} | %{static.get('srr_symbol_recall', 0)} | %{static.get('cgc_graph_completeness', 0)} | {'GEÇTİ' if static.get('quality_gate_passed') else 'BAŞARISIZ'} |",
                "",
            ]
        )
    lines.extend(
        [
            "## LLM A/B sonuçları",
            "",
            "| Metrik | Raw | Air-Context + RAG |",
            "| :--- | ---: | ---: |",
            f"| Retrieval strict pass@{engine.get('retrieval_top_k', config.get('retrieval_top_k', 3))} | - | {score(engine.get('retrieval_score'))} |",
            f"| Retrieval symbol recall@{engine.get('retrieval_top_k', config.get('retrieval_top_k', 3))} | - | {score(engine.get('retrieval_symbol_recall'))} |",
            f"| Structural/static score | %{raw.get('structural_score', 0)} | %{engine.get('structural_score', 0)} |",
            f"| Semantic/judge score | {score(raw.get('semantic_score'), judge_metric=True)} | {score(engine.get('semantic_score'), judge_metric=True)} |",
            f"| Overall score | {score(raw.get('overall_score'))} | {score(engine.get('overall_score'))} |",
            f"| NRA | {score(raw.get('nra_score'))} | {score(engine.get('nra_score'))} |",
            f"| AAS | {score(raw.get('aas_score'))} | {score(engine.get('aas_score'))} |",
            f"| ZHR | {score(raw.get('zhr_score'))} | {score(engine.get('zhr_score'))} |",
            f"| Toplam token | {raw.get('total_tokens', 0)} | {engine.get('total_tokens', 0)} |",
            f"| Giriş tokenı | {raw.get('input_tokens', 0)} | {engine.get('input_tokens', 0)} |",
            f"| Cache'lenen giriş | {raw.get('cached_tokens', 0)} | {engine.get('cached_tokens', 0)} |",
            f"| Çıkış tokenı | {raw.get('output_tokens', 0)} | {engine.get('output_tokens', 0)} |",
            f"| Reasoning tokenı | {raw.get('thought_tokens', 0)} | {engine.get('thought_tokens', 0)} |",
            f"| Token kaynağı | {raw.get('usage_source') or '-'} | {engine.get('usage_source') or '-'} |",
            f"| Ortalama süre | {raw.get('avg_time_sec', 0)} sn | {engine.get('avg_time_sec', 0)} sn |",
            f"| API hatası | {raw.get('api_error_count', 0)} | {engine.get('api_error_count', 0)} |",
            f"| Kesilmiş yanıt | {raw.get('truncated_call_count', 0)} | {engine.get('truncated_call_count', 0)} |",
            f"| Judge hatası | {raw.get('judge_error_count', 0)} | {engine.get('judge_error_count', 0)} |",
            f"| Candidate retry | {raw.get('candidate_retry_count', 0)} | {engine.get('candidate_retry_count', 0)} |",
            f"| Judge retry | {raw.get('judge_retry_count', 0)} | {engine.get('judge_retry_count', 0)} |",
            f"| Gözlenen maliyet | {_money(raw.get('observed_cost_usd'))} | {_money(engine.get('observed_cost_usd'))} |",
            f"| Maliyet eksiksiz mi? | {raw.get('cost_complete', False)} | {engine.get('cost_complete', False)} |",
            "",
            "### Tekrar istatistikleri",
            "",
            "%95 güven aralığı yalnız `n >= 3` olduğunda hesaplanır.",
            "",
            "| Varyant | Metrik | n | Ortalama | Std. sapma | Min | Max | %95 GA |",
            "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | :--- |",
        ]
    )
    for variant, metrics in (("Raw", raw), ("Air-Context + RAG", engine)):
        for category, statistics in metrics.get("score_statistics", {}).items():
            interval = (
                f"[{statistics['ci95_low']}, {statistics['ci95_high']}]"
                if statistics.get("ci_available")
                else "hesaplanmadı"
            )
            lines.append(
                f"| {variant} | {category.upper()} | {statistics.get('n', 0)} | "
                f"{statistics.get('mean', 0)} | "
                f"{statistics.get('stddev') if statistics.get('stddev') is not None else '-'} | "
                f"{statistics.get('min') if statistics.get('min') is not None else '-'} | "
                f"{statistics.get('max') if statistics.get('max') is not None else '-'} | "
                f"{interval} |"
            )
    lines.extend(["", "## Soru ve çağrı detayları", ""])
    for detail in artifact.get("details", []):
        lines.append(
            f"- Tekrar {detail.get('repetition')}, soru `{detail.get('id')}` "
            f"({detail.get('category')}): Raw "
            f"{'✅' if detail.get('raw_result', {}).get('passed') else '❌'}, "
            f"Engine {'✅' if detail.get('engine_result', {}).get('passed') else '❌'}"
        )
        for variant in ("raw", "engine"):
            result = detail.get(f"{variant}_result", {})
            if result.get("error"):
                lines.append(f"  - {variant} hata: `{result['error']}`")

    comparison = artifact.get("comparison")
    if comparison:
        lines.extend(
            [
                "",
                "## Geçmiş koşu karşılaştırması",
                "",
                f"Baseline: `{comparison['baseline_run_id']}` — aynı deney: `{comparison['same_experiment']}`",
                "",
            ]
        )
        if comparison["changed_dimensions"]:
            lines.append("Değişen deney boyutları:")
            lines.append("")
            for field, values in comparison["changed_dimensions"].items():
                lines.append(
                    f"- `{field}`: `{values.get('previous')}` → `{values.get('current')}`"
                )
            lines.append("")
        lines.extend(
            [
                "| Varyant | Skor farkı (puan) | Token farkı | Süre farkı | API hata farkı | Maliyet farkı |",
                "| :--- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for variant in ("raw", "engine"):
            metric = comparison["metrics"][variant]
            lines.append(
                f"| {variant} | {metric['overall_score_delta_points']} | "
                f"{metric['total_tokens_delta']} | {metric['avg_time_sec_delta']} | "
                f"{metric['api_error_count_delta']} | {_money(metric['total_cost_usd_delta'])} |"
            )
    lines.extend(
        ["", "---", "Bu rapor JSON artefaktıyla aynı koşudan üretilmiştir.", ""]
    )
    return "\n".join(lines)


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            Path(temp_name).unlink()
        except OSError:
            pass
        raise


def write_benchmark_artifacts(
    artifact: dict[str, Any],
    output_dir: Path,
    compare_to: Path | None = None,
) -> dict[str, Path]:
    """Write matching JSON and Markdown files for one immutable run."""
    if compare_to:
        artifact["comparison"] = compare_benchmark_runs(
            artifact,
            load_benchmark_artifact(compare_to),
        )
    output_dir = output_dir.absolute()
    stem = str(artifact["run_id"])
    json_path = output_dir / f"{stem}.json"
    markdown_path = output_dir / f"{stem}.md"
    _atomic_write(
        json_path,
        json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    _atomic_write(markdown_path, render_markdown(artifact))
    return {"json": json_path, "markdown": markdown_path}


class BenchmarkReportGenerator:
    """Run static and LLM benchmarks and emit paired artifacts."""

    def __init__(
        self,
        target_path: Path,
        questions_file: Path,
        model_name: str | None = None,
        judge_model_name: str | None = None,
        *,
        provider_name: str = "local",
        provider_base_url: str | None = None,
        temperature: float = 0.0,
        repetitions: int = 3,
        request_delay: float = 1.0,
        input_cost_per_million: float | None = None,
        output_cost_per_million: float | None = None,
        max_output_tokens: int = 1024,
        reasoning_effort: str | None = None,
        search_mode: str = "lexical",
        embedding_model_path: Path | None = None,
        max_retries: int = 3,
        retry_base_delay: float = 1.0,
        retrieval_top_k: int = 8,
        output_dir: Path | None = None,
        compare_to: Path | None = None,
    ):
        self.target_path = target_path.absolute()
        self.questions_file = questions_file.absolute()
        self.model_name = model_name
        self.judge_model_name = judge_model_name
        self.provider_name = provider_name
        self.provider_base_url = provider_base_url
        self.temperature = temperature
        self.repetitions = repetitions
        self.request_delay = request_delay
        self.input_cost_per_million = input_cost_per_million
        self.output_cost_per_million = output_cost_per_million
        self.max_output_tokens = max_output_tokens
        self.reasoning_effort = reasoning_effort
        self.search_mode = search_mode
        self.embedding_model_path = embedding_model_path
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.retrieval_top_k = retrieval_top_k
        self.output_dir = output_dir or self.target_path / ".air-benchmark"
        self.compare_to = compare_to

    def generate_full_report(self) -> dict[str, Path]:
        from .benchmark_llm import LLMBenchmarkEngine

        question_validation = validate_questions_for_project(
            self.target_path,
            load_questions(self.questions_file),
        )
        if not question_validation["valid"]:
            raise ValueError(
                f"Soru seti hedef projeyle uyumsuz: {question_validation['issues']}"
            )
        static_metrics = StaticBenchmarkEngine(self.target_path).run_benchmark()
        engine = LLMBenchmarkEngine(
            self.target_path,
            model_name=self.model_name,
            judge_model_name=self.judge_model_name,
            provider_name=self.provider_name,
            provider_base_url=self.provider_base_url,
            temperature=self.temperature,
            repetitions=self.repetitions,
            request_delay=self.request_delay,
            input_cost_per_million=self.input_cost_per_million,
            output_cost_per_million=self.output_cost_per_million,
            max_output_tokens=self.max_output_tokens,
            reasoning_effort=self.reasoning_effort,
            search_mode=self.search_mode,
            embedding_model_path=self.embedding_model_path,
            max_retries=self.max_retries,
            retry_base_delay=self.retry_base_delay,
            retrieval_top_k=self.retrieval_top_k,
        )
        try:
            artifact = engine.run_eval(self.questions_file)
        finally:
            engine.close()
        artifact["static_metrics"] = static_metrics
        artifact["question_project_validation"] = question_validation
        return write_benchmark_artifacts(artifact, self.output_dir, self.compare_to)
