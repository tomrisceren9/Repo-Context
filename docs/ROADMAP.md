# Roadmap

## Completed foundation

- Multi-language repository scanning and symbol extraction.
- Persistent lexical, semantic, and hybrid retrieval.
- Static and retrieval quality gates.
- Reproducible LLM A/B reports with retries, token usage, cost inputs, repeat
  statistics, and run comparison.
- Local-only OpenAI-compatible LLM integration with no cloud fallback.

## Next priorities

1. Expand synthetic multi-language retrieval fixtures and regression queries.
2. Add benchmark schema migration tests and report compatibility checks.
3. Improve large-monorepo indexing telemetry and cancellation UX.
4. Add release automation, signed artifacts, and dependency vulnerability
   scanning to CI.
5. Document stable extension points for custom embedding providers.

Success means a clean install on every supported Python version, deterministic
offline quality gates, and traceable benchmark runs for the same repository,
model, prompt, and question-set versions.
