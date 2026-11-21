# Agents and harness

How automated agents (trading agents and AI coding assistants) operate in this repository.
`GUARDRAILS.md` overrides everything here.

## Trading agents and decision authority

| Agent | Can do | Cannot do |
|---|---|---|
| Signal strategies | Propose a `TradePlan` (entry, stop, target, invalidation, confidence) | Size or send orders |
| Analyst council (local LLMs via Ollama) | Argue bull/bear cases; veto; shrink size to 0.25x-0.9x | Approve, create or enlarge a trade |
| Meta-labeller (LightGBM) | Score a proposal's probability of hitting its target | Override the gate or risk limits |
| Shariah gate | Final yes/no on instrument, token and order | Be overridden by any other stage |
| Protections and governor | Lock entries after loss streaks, drawdowns, frequency limits | Unlock early |
| Risk engine | Final size and limit check | Exceed leverage 1.0 |
| Exit manager | Close on stop, invalidation, target, time, flatten, trailing stop | Widen a stop |
| Knowledge chatbot / MCP server | Explain and calculate from cited sources | Trade, read secrets, write files |

Pipeline order (`EntryPipeline.decide`): plan -> sizing -> Shariah -> protections -> governor ->
council -> risk engine. Every decision, approved or blocked, goes to the audit log with the
stage that blocked it.

## Evaluation gates

- Strategies and retrained models: `docs/MASTER_PLAN.md` section 4 (Deflated Sharpe >= 0.95,
  PBO < 0.5, Monte Carlo, paper period). Champion/challenger: a new model replaces the old one
  only after beating it out of sample.
- Chatbot: answers must cite a retrieved source; uncited or invalid citations fall back to
  quoting passages.
- LLM analysts: structured JSON at temperature 0; invalid output = neutral.

## Working rules for AI coding assistants

1. Read `GUARDRAILS.md`, `docs/MASTER_PLAN.md` and the module's docs before editing.
2. Python 3.12, pydantic v2, strict mypy, ruff (line length 100). Money is `Decimal`, never float.
   Timestamps are timezone-aware.
3. Add tests with every change; run `cd packages/core && pytest -q`, `ruff check` and `mypy src`.
   Do not increase the mypy error count.
4. Never weaken a Shariah or risk rule to make a test pass. If a rule seems wrong, stop and ask
   the owner.
5. No network calls in tests; inject `fetch`/`generate` callables.
6. Do not commit secrets, data dumps, model binaries or machine-local tool folders.
7. Document user-visible changes in `CHANGELOG.md` and the relevant `docs/` file.
