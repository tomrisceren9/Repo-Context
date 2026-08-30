import sys
import json
import os
import ipaddress
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import click
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.tree import Tree

from .analysis.graph_builder import GraphBuilder
from .analysis.repo_map import RepoMapBuilder
from .analysis.scanner import RepositoryScanner
from .analysis.sot_builder import SOTBuilder
from .benchmarks.benchmark_schema import load_questions
from .benchmarks.benchmark_static import StaticBenchmarkEngine
from .benchmarks.question_preflight import validate_questions_for_project
from .benchmarks.report_generator import (
    compare_benchmark_runs,
    load_benchmark_artifact,
    write_benchmark_artifacts,
)
from .benchmarks.retrieval_benchmark import RetrievalBenchmark
from .cli_errors import StructuredGroup, fail
from .search.rag_engine import RAGEngine


stdout_reconfigure = getattr(sys.stdout, "reconfigure", None)
if (
    sys.stdout.encoding
    and sys.stdout.encoding.lower() != "utf-8"
    and callable(stdout_reconfigure)
):
    stdout_reconfigure(encoding="utf-8")

console = Console()
status_console = Console(stderr=True)


def _target_directory(path: str) -> Path:
    target_path = Path(path).absolute()
    if not target_path.exists():
        fail(
            "PATH_NOT_FOUND",
            "Hedef yol bulunamadı.",
            details={"path": str(target_path)},
            exit_code=2,
        )
    if not target_path.is_dir():
        fail(
            "PATH_NOT_DIRECTORY",
            "Hedef yol bir dizin değil.",
            details={"path": str(target_path)},
            exit_code=2,
        )
    return target_path


def _question_file(path: str) -> Path:
    question_path = Path(path).absolute()
    if not question_path.exists():
        fail(
            "QUESTION_FILE_NOT_FOUND",
            "Soru dosyası bulunamadı.",
            details={"path": str(question_path)},
            exit_code=2,
        )
    if not question_path.is_file():
        fail(
            "QUESTION_PATH_NOT_FILE",
            "Soru yolu bir dosya değil.",
            details={"path": str(question_path)},
            exit_code=2,
        )
    return question_path


def _validated_question_file(path: str) -> Path:
    question_path = _question_file(path)
    try:
        load_questions(question_path)
    except ValueError as exc:
        fail(
            "QUESTION_SCHEMA_INVALID",
            "Soru dosyası JSON Schema doğrulamasını geçemedi.",
            details={"path": str(question_path), "reason": str(exc)},
            exit_code=2,
        )
    return question_path


def _validate_questions_for_target(
    target_path: Path,
    question_path: Path,
) -> dict[str, Any]:
    questions = load_questions(question_path)
    report = validate_questions_for_project(target_path, questions)
    if not report["valid"]:
        fail(
            "QUESTION_PROJECT_MISMATCH",
            "Soru setindeki sembol beklentileri hedef projeyle uyuşmuyor.",
            details={
                "questions": str(question_path),
                **report,
                "suggestion": (
                    "Hedef projeye özel, kaynak koddan doğrulanmış soru setini kullanın."
                ),
            },
            exit_code=2,
        )
    return report


def _benchmark_artifact_file(path: str) -> Path:
    artifact_path = Path(path).absolute()
    if not artifact_path.is_file():
        fail(
            "BENCHMARK_ARTIFACT_NOT_FOUND",
            "Benchmark JSON artefaktı bulunamadı.",
            details={"path": str(artifact_path)},
            exit_code=2,
        )
    try:
        load_benchmark_artifact(artifact_path)
    except ValueError as exc:
        fail(
            "BENCHMARK_ARTIFACT_INVALID",
            "Benchmark JSON artefaktı geçersiz.",
            details={"path": str(artifact_path), "reason": str(exc)},
            exit_code=2,
        )
    return artifact_path


def _runtime_failure(code: str, message: str, exc: Exception) -> None:
    fail(
        code,
        message,
        details={"exception": type(exc).__name__, "reason": str(exc)},
    )


def _validate_pricing(
    input_cost_per_million: float | None,
    output_cost_per_million: float | None,
) -> None:
    if (input_cost_per_million is None) != (output_cost_per_million is None):
        fail(
            "INVALID_PRICING",
            "Giriş ve çıkış milyon-token fiyatları birlikte verilmelidir.",
            details={
                "input_cost_per_million": input_cost_per_million,
                "output_cost_per_million": output_cost_per_million,
            },
            exit_code=2,
        )


def _is_loopback_endpoint(base_url: str) -> bool:
    hostname = urlparse(base_url).hostname
    if hostname is None:
        return False
    if hostname.casefold() == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def _require_remote_endpoint_permission(
    base_url: str | None,
    allowed: bool,
) -> None:
    configured_url = base_url or os.getenv("LOCAL_LLM_BASE_URL")
    if configured_url is None:
        try:
            from dotenv import dotenv_values

            value = dotenv_values().get("LOCAL_LLM_BASE_URL")
            configured_url = str(value) if value else None
        except ImportError:
            configured_url = None
    resolved_url = configured_url or "http://127.0.0.1:11434/v1"
    if _is_loopback_endpoint(resolved_url):
        return
    env_allowed = os.getenv("AIR_CONTEXT_ALLOW_REMOTE_LLM", "").casefold() in {
        "1",
        "true",
        "yes",
    }
    if not allowed and not env_allowed:
        fail(
            "REMOTE_LLM_PERMISSION_REQUIRED",
            "Bu komut depo içeriğini loopback dışındaki bir LLM endpoint'ine gönderebilir.",
            details={
                "endpoint": resolved_url,
                "permission": "--allow-remote-endpoint",
                "environment": "AIR_CONTEXT_ALLOW_REMOTE_LLM=true",
            },
            exit_code=3,
        )


def _load_llm_benchmark() -> tuple[type[Any], type[Any]]:
    try:
        from .benchmarks.benchmark_llm import LLMBenchmarkEngine
        from .benchmarks.report_generator import BenchmarkReportGenerator
    except ImportError as exc:
        fail(
            "OPTIONAL_DEPENDENCY_MISSING",
            "LLM benchmark bağımlılıkları kurulu değil.",
            details={
                "install": "uv sync --locked --extra benchmark",
                "missing_module": exc.name,
            },
            exit_code=3,
        )
    return LLMBenchmarkEngine, BenchmarkReportGenerator


def _local_embedding_provider(model_path: str | None) -> Any:
    if not model_path:
        return None
    from .search.semantic_search import SentenceTransformerEmbeddingProvider

    return SentenceTransformerEmbeddingProvider(Path(model_path))


def _validate_search_configuration(
    search_mode: str, embedding_model: str | Path | None
) -> None:
    if search_mode in {"semantic", "hybrid"} and embedding_model is None:
        fail(
            "LOCAL_EMBEDDING_MODEL_REQUIRED",
            "Semantic/hybrid arama için yerel embedding model dizini gereklidir.",
            details={
                "search_mode": search_mode,
                "cloud_fallback": False,
                "option": "--embedding-model",
            },
            exit_code=2,
        )


@click.group(cls=StructuredGroup)
def main() -> None:
    """Local-first repository context, search, and benchmark CLI."""


@main.command()
@click.option("--path", "-p", default=".", help="Taranacak proje dizini")
def scan(path: str) -> None:
    """Projedeki geçerli dosyaları tarar ve ağaç yapısını görüntüler."""
    target_path = _target_directory(path)
    console.print(f"[bold green]Proje Taranıyor:[/bold green] {target_path}\n")
    try:
        scanner = RepositoryScanner(target_path)
        tree = Tree(f"📁 [bold blue]{target_path.name}[/bold blue]")
        count = 0
        for file_path in scanner.scan_files():
            tree.add(f"📄 {file_path.relative_to(target_path)}")
            count += 1
    except Exception as exc:
        _runtime_failure("SCAN_FAILED", "Proje taranamadı.", exc)
        return

    console.print(tree)
    console.print(
        f"\n[bold yellow]Toplam Bulunan Geçerli Dosya Sayısı:[/bold yellow] {count}"
    )


@main.command()
@click.option("--path", "-p", default=".", help="İşlenecek proje dizini")
def build(path: str) -> None:
    """SOT, repo map ve bağımlılık grafı dosyalarını üretir."""
    target_path = _target_directory(path)
    console.print(
        f"[bold blue]Bağlam Dosyaları Üretiliyor:[/bold blue] {target_path}\n"
    )
    try:
        scanner = RepositoryScanner(target_path)
        outputs = [
            SOTBuilder(target_path, scanner).save(),
            RepoMapBuilder(target_path, scanner).save(),
            GraphBuilder(target_path, scanner).save(),
        ]
    except Exception as exc:
        _runtime_failure("BUILD_FAILED", "Bağlam dosyaları üretilemedi.", exc)
        return

    for output in outputs:
        console.print(f"✅ Oluşturuldu: [bold green]{output}[/bold green]")
    console.print("\n[bold gold1]🎉 İşlem Tamamlandı![/bold gold1]")


@main.command()
@click.argument("prompt")
@click.option("--path", "-p", default=".", help="Aranacak proje dizini")
@click.option(
    "--top-k", "-k", default=3, type=int, help="Getirilecek maksimum kod bloğu sayısı"
)
@click.option(
    "--file",
    "file_paths",
    multiple=True,
    help="Dosya yolu GLOB filtresi (tekrarlanabilir)",
)
@click.option(
    "--language", "languages", multiple=True, help="Dil filtresi (tekrarlanabilir)"
)
@click.option(
    "--symbol-type", "symbol_types", multiple=True, help="Sembol türü filtresi"
)
@click.option(
    "--start-line", type=int, default=None, help="Örtüşmesi gereken ilk satır"
)
@click.option("--end-line", type=int, default=None, help="Örtüşmesi gereken son satır")
@click.option(
    "--diversity", type=click.FloatRange(0.0, 1.0), default=0.25, show_default=True
)
@click.option(
    "--search-mode",
    type=click.Choice(["lexical", "semantic", "hybrid"]),
    default="lexical",
    show_default=True,
)
@click.option(
    "--embedding-model",
    default=None,
    help="Yalnız yerel sentence-transformers model dizini",
)
def query(
    prompt: str,
    path: str,
    top_k: int,
    file_paths: tuple[str, ...],
    languages: tuple[str, ...],
    symbol_types: tuple[str, ...],
    start_line: int | None,
    end_line: int | None,
    diversity: float,
    search_mode: str,
    embedding_model: str | None,
) -> None:
    """Sorguya en uygun kod parçalarını arar ve sıralar."""
    target_path = _target_directory(path)
    _validate_search_configuration(search_mode, embedding_model)
    if top_k <= 0:
        fail(
            "INVALID_TOP_K",
            "top-k pozitif bir tam sayı olmalıdır.",
            details={"top_k": top_k},
            exit_code=2,
        )
    console.print(f"[bold cyan]🔍 Arama Yapılıyor:[/bold cyan] '{prompt}'\n")
    try:
        engine = RAGEngine(
            target_path,
            RepositoryScanner(target_path),
            embedding_provider=_local_embedding_provider(embedding_model),
            search_mode=search_mode,
        )
        results = engine.search(
            prompt,
            top_k=top_k,
            file_paths=file_paths,
            languages=languages,
            symbol_types=symbol_types,
            start_line=start_line,
            end_line=end_line,
            diversity_strength=diversity,
        )
    except Exception as exc:
        _runtime_failure("QUERY_FAILED", "Kod araması çalıştırılamadı.", exc)
        return

    if not results:
        console.print("[yellow]Alakalı bir kod parçası bulunamadı.[/yellow]")
        return

    for index, result in enumerate(results, 1):
        file_info = (
            f"📄 {result['file_path']} "
            f"(Satır: {result['start_line']}-{result['end_line']}) | "
            f"{result['language']} / {result['symbol_type']} / "
            f"{result['symbol_name'] or '-'} | Skor: {result['score']} "
            f"(lexical={result.get('lexical_score', result.get('score', 0))}, "
            f"semantic={result.get('semantic_score', 0)}, "
            f"exact={result.get('exact_score', 0)}, "
            f"bm25={result.get('bm25_score', 0)}, "
            f"tfidf={result.get('tfidf_score', 0)}, "
            f"metadata={result.get('metadata_score', 0)}, "
            f"callable={result.get('callable_score', 0)}, "
            f"scope={result.get('scope_score', 0)})"
        )
        extension = Path(result["file_path"]).suffix.replace(".", "") or "text"
        syntax = Syntax(
            result["content"],
            extension,
            theme="monokai",
            line_numbers=True,
            start_line=result["start_line"],
        )
        console.print(f"[bold yellow]Sonuç #{index}[/bold yellow] — {file_info}")
        console.print(
            Panel(
                syntax,
                title=f"[bold yellow]Kod #{index}[/bold yellow]",
                expand=False,
            )
        )


@main.command(name="index")
@click.option("--path", "-p", default=".", help="İndekslenecek proje dizini")
@click.option(
    "--embedding-model",
    default=None,
    help="Opsiyonel yerel sentence-transformers model dizini",
)
def index_repository(path: str, embedding_model: str | None) -> None:
    """SQLite arama indeksini artımlı olarak günceller."""
    target_path = _target_directory(path)
    try:
        provider = _local_embedding_provider(embedding_model)
        engine = RAGEngine(
            target_path,
            RepositoryScanner(target_path),
            embedding_provider=provider,
        )
        semantic_stats = engine.semantic_index.sync() if engine.semantic_index else None
    except Exception as exc:
        _runtime_failure("INDEX_FAILED", "Arama indeksi güncellenemedi.", exc)
        return
    console.print_json(
        json.dumps(
            {**engine.index_stats.to_dict(), "semantic": semantic_stats},
            ensure_ascii=False,
        )
    )


@main.command(name="benchmark-retrieval")
@click.option("--path", "-p", required=True, help="Sabit fixture deposu")
@click.option("--evaluation", "-e", required=True, help="Relevance JSON dosyası")
@click.option("--top-k", default=5, type=click.IntRange(min=1), show_default=True)
@click.option("--repetitions", default=3, type=click.IntRange(min=1), show_default=True)
@click.option(
    "--scale-files", default=300, type=click.IntRange(min=0), show_default=True
)
def benchmark_retrieval(
    path: str,
    evaluation: str,
    top_k: int,
    repetitions: int,
    scale_files: int,
) -> None:
    """Recall@k, MRR ve cold/warm indeks hızını çevrimdışı ölçer."""
    target_path = _target_directory(path)
    evaluation_path = _question_file(evaluation)
    try:
        result = RetrievalBenchmark(target_path, evaluation_path).run(
            top_k=top_k,
            repetitions=repetitions,
            scale_files=scale_files,
        )
    except Exception as exc:
        _runtime_failure(
            "RETRIEVAL_BENCHMARK_FAILED", "Retrieval benchmark çalıştırılamadı.", exc
        )
        return
    console.print_json(json.dumps(result, ensure_ascii=False))


@main.command(name="benchmark-static")
@click.option("--path", "-p", default=".", help="Test edilecek proje dizini")
def benchmark_static(path: str) -> None:
    """Statik bağlam kalitesi ve sıkıştırma metriklerini hesaplar."""
    target_path = _target_directory(path)
    console.print(
        f"[bold cyan]📊 Statik Bağlam Metrikleri Hesaplanıyor:[/bold cyan] {target_path}\n"
    )
    try:
        metrics = StaticBenchmarkEngine(target_path).run_benchmark()
    except Exception as exc:
        _runtime_failure(
            "STATIC_BENCHMARK_FAILED", "Statik benchmark çalıştırılamadı.", exc
        )
        return
    if "error" in metrics:
        fail(
            "STATIC_BENCHMARK_EMPTY",
            str(metrics["error"]),
            details={"path": str(target_path)},
        )

    table = Table(title="=== BENCHMARK STATİK METRİK RAPORU ===")
    table.add_column("Metrik Adı", style="cyan", no_wrap=True)
    table.add_column("Değer", style="bold green")
    rows = [
        ("Toplam Taranan Dosya", str(metrics["total_files"])),
        ("Ham Kod Token Sayısı", f"{metrics['raw_tokens']:,}"),
        ("Üretilen Bağlam Token Sayısı", f"{metrics['context_tokens']:,}"),
        ("CCR (Sıkıştırma Oranı)", f"%{metrics['ccr_compression_ratio']}"),
        ("CDS (Yoğunluk Skoru)", f"%{metrics['cds_density_score']}"),
        ("FIC (Dosya Envanteri Kapsamı)", f"%{metrics['fic_file_inventory_coverage']}"),
        ("SRR (Sembol Kapsama Oranı)", f"%{metrics['srr_symbol_recall']}"),
        ("CGC (Graf Eksiksizliği)", f"%{metrics['cgc_graph_completeness']}"),
        (
            "Kalite Kapısı",
            "GEÇTİ" if metrics["quality_gate_passed"] else "BAŞARISIZ",
        ),
    ]
    for label, value in rows:
        table.add_row(label, value)
    console.print(table)


@main.command(name="validate-questions")
@click.argument("questions")
@click.option(
    "--path",
    "project_path",
    default=None,
    help="Sembol beklentilerinin doğrulanacağı hedef proje dizini",
)
def validate_benchmark_questions(questions: str, project_path: str | None) -> None:
    """Bir benchmark soru dosyasını API çağrısından önce doğrular."""
    questions_path = _validated_question_file(questions)
    validated = load_questions(questions_path)
    project_validation = None
    if project_path is not None:
        target_path = _target_directory(project_path)
        project_validation = _validate_questions_for_target(target_path, questions_path)
    console.print_json(
        json.dumps(
            {
                "valid": True,
                "path": str(questions_path),
                "question_count": len(validated),
                "project_validation": project_validation,
            },
            ensure_ascii=False,
        )
    )


@main.command(name="benchmark-llm")
@click.option("--path", "-p", default=".", help="Test edilecek proje dizini")
@click.option("--questions", "-q", required=True, help="Soru seti JSON dosyası")
@click.option("--provider-base-url", default=None, help="OpenAI-uyumlu yerel API kökü")
@click.option("--model", default=None, help="Model adı (varsayılan: qwen3:4b)")
@click.option("--judge-model", default=None, help="İsteğe bağlı semantik judge modeli")
@click.option(
    "--search-mode",
    type=click.Choice(["lexical", "semantic", "hybrid"]),
    default="lexical",
    show_default=True,
    help="Engine varyantının retrieval modu",
)
@click.option(
    "--embedding-model",
    type=click.Path(path_type=Path),
    default=None,
    help="Semantic/hybrid retrieval için yerel model dizini",
)
@click.option(
    "--temperature", type=click.FloatRange(0, 2), default=0.0, show_default=True
)
@click.option(
    "--max-output-tokens",
    type=click.IntRange(min=1),
    default=1024,
    show_default=True,
    help="Her model yanıtı için azami çıktı tokenı",
)
@click.option(
    "--reasoning-effort",
    type=click.Choice(["none", "minimal", "low", "medium", "high", "xhigh", "max"]),
    default=None,
    help="Uyumlu yerel endpoint için reasoning eforu (varsayılan: none)",
)
@click.option(
    "--runs",
    "--repetitions",
    "repetitions",
    type=click.IntRange(min=1),
    default=None,
    help="LLM tekrar sayısı (varsayılan: 3)",
)
@click.option(
    "--request-delay",
    type=click.FloatRange(min=0),
    default=None,
    help="İstekler arası saniye (varsayılan: 0)",
)
@click.option(
    "--max-retries",
    type=click.IntRange(min=0),
    default=3,
    show_default=True,
    help="Geçici LLM hatalarında azami yeniden deneme sayısı",
)
@click.option(
    "--retry-base-delay",
    type=click.FloatRange(min=0),
    default=1.0,
    show_default=True,
    help="Retry-After yoksa exponential backoff başlangıcı (saniye)",
)
@click.option(
    "--retrieval-top-k",
    type=click.IntRange(min=1),
    default=8,
    show_default=True,
    help="Her soru için getirilecek mikro bağlam parçası sayısı",
)
@click.option("--input-cost-per-million", type=click.FloatRange(min=0), default=None)
@click.option("--output-cost-per-million", type=click.FloatRange(min=0), default=None)
@click.option("--output-dir", type=click.Path(path_type=Path), default=None)
@click.option("--compare-to", type=click.Path(path_type=Path), default=None)
@click.option(
    "--allow-remote-endpoint",
    is_flag=True,
    help="Depo bağlamının loopback dışındaki LLM endpoint'ine gönderilmesine izin ver",
)
def benchmark_llm(
    path: str,
    questions: str,
    provider_base_url: str | None,
    model: str | None,
    judge_model: str | None,
    search_mode: str,
    embedding_model: Path | None,
    temperature: float,
    max_output_tokens: int,
    reasoning_effort: str | None,
    repetitions: int | None,
    request_delay: float | None,
    max_retries: int,
    retry_base_delay: float,
    retrieval_top_k: int,
    input_cost_per_million: float | None,
    output_cost_per_million: float | None,
    output_dir: Path | None,
    compare_to: Path | None,
    allow_remote_endpoint: bool,
) -> None:
    """Yerel LLM ile Raw Code ve Air-Context A/B testini çalıştırır."""
    target_path = _target_directory(path)
    questions_path = _validated_question_file(questions)
    _validate_search_configuration(search_mode, embedding_model)
    _validate_pricing(input_cost_per_million, output_cost_per_million)
    compare_path = _benchmark_artifact_file(str(compare_to)) if compare_to else None
    project_validation = _validate_questions_for_target(target_path, questions_path)
    _require_remote_endpoint_permission(provider_base_url, allow_remote_endpoint)
    resolved_repetitions = repetitions if repetitions is not None else 3
    resolved_request_delay = request_delay if request_delay is not None else 0.0
    resolved_reasoning_effort = reasoning_effort or "none"
    engine_type, _ = _load_llm_benchmark()
    engine = None
    try:
        engine = engine_type(
            target_path,
            model_name=model,
            judge_model_name=judge_model,
            provider_name="local",
            provider_base_url=provider_base_url,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            reasoning_effort=resolved_reasoning_effort,
            search_mode=search_mode,
            embedding_model_path=embedding_model,
            repetitions=resolved_repetitions,
            request_delay=resolved_request_delay,
            max_retries=max_retries,
            retry_base_delay=retry_base_delay,
            retrieval_top_k=retrieval_top_k,
            input_cost_per_million=input_cost_per_million,
            output_cost_per_million=output_cost_per_million,
        )
        result = engine.run_eval(questions_path)
        result["question_project_validation"] = project_validation
        artifacts = write_benchmark_artifacts(
            result,
            (output_dir or target_path / ".air-benchmark"),
            compare_path,
        )
    except Exception as exc:
        _runtime_failure("LLM_BENCHMARK_FAILED", "LLM benchmark çalıştırılamadı.", exc)
        return
    finally:
        close = getattr(engine, "close", None)
        if callable(close):
            close()

    display_result = (
        load_benchmark_artifact(artifacts["json"])
        if artifacts["json"].is_file()
        else result
    )
    raw = display_result["raw_metrics"]
    engine = display_result["engine_metrics"]

    def score_text(value: object) -> str:
        return "hesaplanmadı" if value is None else f"%{value}"

    table = Table(title="A/B TEST SONUÇLARI")
    table.add_column("Metrik", style="bold white")
    table.add_column("Ham Kod", style="bold red")
    table.add_column("Air-Context + RAG", style="bold green")
    table.add_row(
        "Token Kullanımı",
        f"{raw['total_tokens']:,}",
        f"{engine['total_tokens']:,} "
        f"(%{display_result['token_savings_percent']} tasarruf)",
    )
    table.add_row(
        "Giriş Tokenı",
        f"{raw.get('input_tokens', 0):,}",
        f"{engine.get('input_tokens', 0):,}",
    )
    table.add_row(
        "Cache'lenen Giriş",
        f"{raw.get('cached_tokens', 0):,}",
        f"{engine.get('cached_tokens', 0):,}",
    )
    table.add_row(
        "Çıkış Tokenı",
        f"{raw.get('output_tokens', 0):,}",
        f"{engine.get('output_tokens', 0):,}",
    )
    table.add_row(
        "Ort. Yanıt Süresi", f"{raw['avg_time_sec']} sn", f"{engine['avg_time_sec']} sn"
    )
    table.add_row(
        "Retrieval strict",
        "-",
        score_text(engine.get("retrieval_score")),
    )
    table.add_row(
        f"Symbol recall@{engine.get('retrieval_top_k', retrieval_top_k)}",
        "-",
        score_text(engine.get("retrieval_symbol_recall")),
    )
    table.add_row(
        "Structural/Static",
        str(raw.get("structural_score")),
        str(engine.get("structural_score")),
    )
    table.add_row(
        "Semantic/Judge",
        score_text(raw.get("semantic_score")),
        score_text(engine.get("semantic_score")),
    )
    table.add_row(
        "Overall",
        score_text(raw.get("overall_score")),
        score_text(engine.get("overall_score")),
    )
    table.add_row("NRA", score_text(raw["nra_score"]), score_text(engine["nra_score"]))
    table.add_row("AAS", score_text(raw["aas_score"]), score_text(engine["aas_score"]))
    table.add_row("ZHR", score_text(raw["zhr_score"]), score_text(engine["zhr_score"]))
    table.add_row(
        "API Hatası", str(raw["api_error_count"]), str(engine["api_error_count"])
    )
    table.add_row(
        "Kesilmiş Yanıt",
        str(raw["truncated_call_count"]),
        str(engine["truncated_call_count"]),
    )
    table.add_row(
        "Gözlenen Maliyet",
        str(raw["observed_cost_usd"]),
        str(engine["observed_cost_usd"]),
    )
    table.add_row(
        "Maliyet Eksiksiz", str(raw["cost_complete"]), str(engine["cost_complete"])
    )
    console.print(table)
    console.print(
        "Koşu durumu: "
        f"[{'bold green' if display_result.get('benchmark_status') == 'valid' else 'bold red'}]"
        f"{display_result.get('benchmark_status', 'unknown')}[/]"
    )
    console.print(f"JSON: [bold green]{artifacts['json']}[/bold green]")
    console.print(f"Markdown: [bold green]{artifacts['markdown']}[/bold green]")
    if display_result.get("benchmark_status") == "invalid":
        fail(
            "BENCHMARK_INVALID",
            "Benchmark artefaktı üretildi ancak altyapı hataları nedeniyle skorlar geçersiz.",
            details={
                "json": str(artifacts["json"]),
                "markdown": str(artifacts["markdown"]),
                "invalid_reasons": display_result.get("invalid_reasons", []),
            },
        )


@main.command()
@click.option("--path", "-p", default=".", help="Test edilecek proje dizini")
@click.option("--questions", "-q", required=True, help="Soru seti JSON dosyası")
@click.option("--provider-base-url", default=None, help="OpenAI-uyumlu yerel API kökü")
@click.option("--model", default=None, help="Model adı (varsayılan: qwen3:4b)")
@click.option("--judge-model", default=None, help="İsteğe bağlı semantik judge modeli")
@click.option(
    "--search-mode",
    type=click.Choice(["lexical", "semantic", "hybrid"]),
    default="lexical",
    show_default=True,
    help="Engine varyantının retrieval modu",
)
@click.option(
    "--embedding-model",
    type=click.Path(path_type=Path),
    default=None,
    help="Semantic/hybrid retrieval için yerel model dizini",
)
@click.option(
    "--temperature", type=click.FloatRange(0, 2), default=0.0, show_default=True
)
@click.option(
    "--max-output-tokens",
    type=click.IntRange(min=1),
    default=1024,
    show_default=True,
    help="Her model yanıtı için azami çıktı tokenı",
)
@click.option(
    "--reasoning-effort",
    type=click.Choice(["none", "minimal", "low", "medium", "high", "xhigh", "max"]),
    default=None,
    help="Uyumlu yerel endpoint için reasoning eforu (varsayılan: none)",
)
@click.option(
    "--runs",
    "--repetitions",
    "repetitions",
    type=click.IntRange(min=1),
    default=None,
    help="LLM tekrar sayısı (varsayılan: 3)",
)
@click.option(
    "--request-delay",
    type=click.FloatRange(min=0),
    default=None,
    help="İstekler arası saniye (varsayılan: 0)",
)
@click.option("--max-retries", type=click.IntRange(min=0), default=3, show_default=True)
@click.option(
    "--retry-base-delay",
    type=click.FloatRange(min=0),
    default=1.0,
    show_default=True,
)
@click.option(
    "--retrieval-top-k", type=click.IntRange(min=1), default=8, show_default=True
)
@click.option("--input-cost-per-million", type=click.FloatRange(min=0), default=None)
@click.option("--output-cost-per-million", type=click.FloatRange(min=0), default=None)
@click.option("--output-dir", type=click.Path(path_type=Path), default=None)
@click.option("--compare-to", type=click.Path(path_type=Path), default=None)
@click.option(
    "--allow-remote-endpoint",
    is_flag=True,
    help="Depo bağlamının loopback dışındaki LLM endpoint'ine gönderilmesine izin ver",
)
def benchmark(
    path: str,
    questions: str,
    provider_base_url: str | None,
    model: str | None,
    judge_model: str | None,
    search_mode: str,
    embedding_model: Path | None,
    temperature: float,
    max_output_tokens: int,
    reasoning_effort: str | None,
    repetitions: int | None,
    request_delay: float | None,
    max_retries: int,
    retry_base_delay: float,
    retrieval_top_k: int,
    input_cost_per_million: float | None,
    output_cost_per_million: float | None,
    output_dir: Path | None,
    compare_to: Path | None,
    allow_remote_endpoint: bool,
) -> None:
    """Statik ve LLM benchmark çalıştırıp JSON/Markdown artefaktları üretir."""
    target_path = _target_directory(path)
    questions_path = _validated_question_file(questions)
    _validate_search_configuration(search_mode, embedding_model)
    _validate_pricing(input_cost_per_million, output_cost_per_million)
    compare_path = _benchmark_artifact_file(str(compare_to)) if compare_to else None
    _validate_questions_for_target(target_path, questions_path)
    _require_remote_endpoint_permission(provider_base_url, allow_remote_endpoint)
    resolved_repetitions = repetitions if repetitions is not None else 3
    resolved_request_delay = request_delay if request_delay is not None else 0.0
    resolved_reasoning_effort = reasoning_effort or "none"
    _, report_type = _load_llm_benchmark()
    try:
        report_file = report_type(
            target_path,
            questions_path,
            model_name=model,
            judge_model_name=judge_model,
            provider_name="local",
            provider_base_url=provider_base_url,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            reasoning_effort=resolved_reasoning_effort,
            search_mode=search_mode,
            embedding_model_path=embedding_model,
            repetitions=resolved_repetitions,
            request_delay=resolved_request_delay,
            max_retries=max_retries,
            retry_base_delay=retry_base_delay,
            retrieval_top_k=retrieval_top_k,
            input_cost_per_million=input_cost_per_million,
            output_cost_per_million=output_cost_per_million,
            output_dir=output_dir,
            compare_to=compare_path,
        ).generate_full_report()
    except Exception as exc:
        _runtime_failure(
            "BENCHMARK_REPORT_FAILED", "Benchmark raporu üretilemedi.", exc
        )
        return
    console.print(f"[bold green]JSON raporu:[/bold green] {report_file['json']}")
    console.print(
        f"[bold green]Markdown raporu:[/bold green] {report_file['markdown']}"
    )
    artifact = load_benchmark_artifact(report_file["json"])
    if artifact.get("benchmark_status") == "invalid":
        fail(
            "BENCHMARK_INVALID",
            "Benchmark artefaktı üretildi ancak altyapı hataları nedeniyle skorlar geçersiz.",
            details={
                "json": str(report_file["json"]),
                "markdown": str(report_file["markdown"]),
                "invalid_reasons": artifact.get("invalid_reasons", []),
            },
        )


@main.command(name="benchmark-compare")
@click.argument("current")
@click.argument("baseline")
def benchmark_compare(current: str, baseline: str) -> None:
    """İki geçmiş JSON benchmark koşusunu API çağrısı yapmadan karşılaştırır."""
    current_path = _benchmark_artifact_file(current)
    baseline_path = _benchmark_artifact_file(baseline)
    try:
        comparison = compare_benchmark_runs(
            load_benchmark_artifact(current_path),
            load_benchmark_artifact(baseline_path),
        )
    except ValueError as exc:
        _runtime_failure("BENCHMARK_COMPARE_FAILED", "Koşular karşılaştırılamadı.", exc)
        return
    console.print_json(json.dumps(comparison, ensure_ascii=False))


if __name__ == "__main__":
    main()
