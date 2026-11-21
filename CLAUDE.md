# CLAUDE.md

Read these first, in order: `GUARDRAILS.md` (hard limits), `AGENTS.md` (harness and coding
rules), `docs/HANDOFF.md` (where the work stands and what is next), `docs/MASTER_PLAN.md`.

## Commands

```bash
pip install -e '.[dev]'                 # from the repo root (Python 3.12+)
pytest -q                               # full suite; must stay green
cd packages/core && ruff check src tests && mypy src   # mypy baseline: 55 pre-existing errors
```

Run `ruff format` only on files you changed, never on whole folders: it reformats unrelated
files. Do not raise the mypy error count.

## Non-negotiables

- The Shariah gate, risk engine, edge-over-cost filter and protections sit in front of every
  order. Never weaken them to make a test pass; ask the owner instead.
- Futures, options, binaries, CFDs, perpetuals, margin and short selling stay blocked (checked
  against IIFA Resolution 63 and AAOIFI Standards 1 and 21).
- Paper trading only. Live trading stays behind the two-key interlock.
- Money is `Decimal`; timestamps are timezone-aware; no network calls in tests.
- Telegram is not usable for the owner: alerts go through ntfy, webhooks or email.
