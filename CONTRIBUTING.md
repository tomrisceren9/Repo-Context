# Contributing

1. Create a focused branch and add tests for behavioral changes.
2. Use synthetic fixtures; never commit proprietary repositories, credentials,
   generated indexes, or raw benchmark reports.
3. Install development dependencies and run the quality gates:

```bash
uv sync --locked --extra benchmark --extra dev
uv run ruff check .
uv run mypy .
uv run coverage run -m pytest -q
uv run coverage report --fail-under=80
```

Keep public APIs documented, preserve backward compatibility where practical,
and explain user-visible changes in `CHANGELOG.md`.
