# Contributing

## Required quality gates

- `ruff check .`
- `mypy packages/core/src apps/api/src`
- `pytest`

## Rules

- No strategy may call a broker directly.
- No floats in future money/accounting implementations.
- No hidden global mutable state.
- No untyped public APIs in the financial core.
- Add regression tests for every fixed bug.
- Keep modules cohesive; do not create catch-all `utils.py` files.
