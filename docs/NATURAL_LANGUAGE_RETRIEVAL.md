# Natural-language retrieval benchmark

This benchmark measures whether a natural-language query retrieves the expected
file and symbol without leaking the target identifier into the query.

Each evaluation record contains a stable `id`, a natural-language `query`, the
expected `file` and `symbol`, and optional `query_style` and
`forbidden_identifiers` fields.

```bash
air-context benchmark-retrieval \
  --path /path/to/project \
  --evaluation /path/to/retrieval_questions.json \
  --top-k 5
```

The JSON output reports exact-symbol and snippet Recall@1/3/k, MRR, per-query
ranking details, and cold/warm index timing. The synthetic file in
`examples/retrieval_questions.json` documents the format; adapt it to the target
repository before using its results for decisions.
