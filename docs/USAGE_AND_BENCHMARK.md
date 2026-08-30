# Usage and benchmarking

## Generate repository context

```bash
air-context build --path /path/to/project
```

This writes `CLAUDE.md`, `repo_map.md`, and `dependency_graph.md` into the target
repository. Review generated files before sharing them because they summarize
source structure and symbols.

## Run offline checks

```bash
air-context benchmark-static --path /path/to/project
air-context benchmark-retrieval \
  --path /path/to/project \
  --evaluation /path/to/retrieval_questions.json
```

Static and retrieval benchmarks do not require an LLM.

## Run the LLM A/B benchmark

Start an OpenAI-compatible local endpoint, validate a repository-specific
question set, and run the comparison:

```bash
air-context validate-questions /path/to/questions.json --path /path/to/project

air-context benchmark-llm \
  --path /path/to/project \
  --questions /path/to/questions.json \
  --provider-base-url http://127.0.0.1:11434/v1 \
  --model qwen3:4b \
  --runs 3
```

The benchmark compares a raw-code prompt with generated macro context plus RAG.
It records model and prompt versions, token usage, latency, retries, errors,
retrieval evidence, repeat statistics, and optional judge results. Matching JSON
and Markdown reports are written under `.air-benchmark/`.

The bundled question file demonstrates the schema only. Reliable measurements
require questions and expected symbols reviewed against the target commit.
