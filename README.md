# Air Context Engine

Air Context Engine is a local-first CLI for turning a source repository into
compact, searchable context for developers and local LLMs.

It can:

- generate a repository overview, symbol map, and dependency graph;
- build a persistent lexical, semantic, or hybrid code-search index;
- measure static context quality and retrieval relevance;
- compare raw-code prompts with Air Context + RAG through an OpenAI-compatible
  local LLM endpoint.

No cloud provider or fallback is built in. Source code leaves the machine only
when you explicitly configure a non-loopback OpenAI-compatible endpoint.

## Install

Python 3.10–3.14 is supported.

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[benchmark]"
```

Optional local semantic search:

```bash
pip install -e ".[benchmark,semantic]"
```

## Quick start

```bash
# Generate CLAUDE.md, repo_map.md, and dependency_graph.md
air-context build --path /path/to/project

# Build and query the local index
air-context index --path /path/to/project
air-context query --path /path/to/project "where is user authentication handled?"

# Run offline quality checks
air-context benchmark-static --path /path/to/project
```

## Local LLM benchmark

Copy `.env.example` to `.env` and point it at Ollama, llama.cpp, vLLM, or another
OpenAI-compatible endpoint. API keys are optional for endpoints that do not
require authentication.

```dotenv
LOCAL_LLM_BASE_URL=http://127.0.0.1:11434/v1
LOCAL_LLM_API_KEY=
```

Create and review a project-specific question set, then run:

```bash
air-context validate-questions /path/to/questions.json --path /path/to/project

air-context benchmark-llm \
  --path /path/to/project \
  --questions /path/to/questions.json \
  --model qwen3:4b \
  --runs 3
```

The benchmark writes matching JSON and Markdown reports to `.air-benchmark/`.
Question validation is strict so a mismatched fixture cannot silently produce a
misleading score.

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Usage and benchmarking](docs/USAGE_AND_BENCHMARK.md)
- [Natural-language retrieval](docs/NATURAL_LANGUAGE_RETRIEVAL.md)
- [Roadmap](docs/ROADMAP.md)
- [Contributing](CONTRIBUTING.md)
- [Security](SECURITY.md)

## License

MIT
