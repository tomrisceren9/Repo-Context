# Changelog

## Unreleased

- Restricted LLM benchmarks to explicit OpenAI-compatible local endpoints with
  no cloud fallback.
- Added persistent local semantic and hybrid retrieval.
- Added strict benchmark question preflight, reproducible JSON/Markdown reports,
  retry metadata, token accounting, repeat statistics, and run comparison.
- Added natural-language retrieval evaluation with Recall@k and MRR.
- Improved Windows and restricted-drive path handling while preserving a
  fail-closed repository boundary.
- Removed repository-specific benchmark fixtures and internal measurements from
  the public distribution.

## 0.6.0 — 13 August 2026

- Added structured local search, incremental indexing, multi-language analysis,
  reproducible benchmarks, and package/CI foundations.

Earlier development versions were consolidated before the first public release.
