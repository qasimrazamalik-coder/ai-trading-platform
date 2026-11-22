# Plugins, marketplace, strategy selection and signal intake

LLMs advise, code decides. Nothing on this page adds a path to a broker: every order still
passes the Shariah gate, the risk engine, the edge-over-cost filter and the protections in
`EntryPipeline.decide`, in paper mode.

## Plugins ("tentacles")

Package: `quant_os_core.plugins`.

| Kind            | Component                                             |
| --------------- | ----------------------------------------------------- |
| `strategy`      | `StrategyPlugin(metadata, factory)`                   |
| `indicator`     | any callable                                          |
| `data_source`   | any object                                            |
| `signal_source` | any object (feeds `SignalIntake`, never the broker)   |
| `alert_sink`    | any object (ntfy, webhook or email; never Telegram)   |

Every plugin declares a `PluginManifest`: `name` (lowercase, `[a-z0-9_.-]`), `version`,
`kind`, `halal_reviewed` and `permissions`.

Rules enforced by `PluginRegistry.register`:

- **Order placement is refused.** A manifest that requests `place_orders` is rejected:
  orders only go through the entry pipeline.
- **Unknown kinds or permissions are refused** (fail closed).
- **Not reviewed means research only.** `halal_reviewed=False` registers the plugin with
  status `research_only`. `registry.tradable()` never returns it; it can still be
  backtested. Flip the flag only after the owner (and a scholar, for Shariah questions)
  has reviewed it.
- Duplicate names are refused; strategies must be `StrategyPlugin`, indicators callable.

### Discovery

```python
from pathlib import Path
from quant_os_core.plugins import PluginRegistry, discover

registry = PluginRegistry()
report = discover(registry, directory=Path("plugins"))
print(report.loaded, report.refused)
```

`discover` loads, in order: the built-in example, installed packages that expose the entry
point group `quant_os.plugins`, then each `*.py` file in the local directory (files starting
with `_` are skipped). Each source must expose `register(registrar)`. A refusal or a broken
plugin becomes a line in `report.refused`; it never stops other plugins from loading.

Loading a plugin runs its code. Keep the local `plugins/` directory owner-controlled and
install third-party packages only after reading them.

Entry point in a third-party `pyproject.toml`:

```toml
[project.entry-points."quant_os.plugins"]
my_plugin = "my_package.plugin:register"
```

Minimal plugin module:

```python
from quant_os_core.plugins.registry import Permission, PluginKind, PluginManifest

def register(registrar):
    registrar.register(
        PluginManifest(
            name="acme.rsi",
            version="1.0.0",
            kind=PluginKind.INDICATOR,
            halal_reviewed=False,
            permissions=frozenset({Permission.READ_MARKET_DATA}),
        ),
        my_rsi_function,
    )
```

The built-in example `builtin.long_only_ma_crossover` (`plugins/builtin.py`) wraps the
moving-average crossover and clips exposure to `[0, 1]`: a bearish cross means flat, never
short. It ships research-only until the owner reviews it.

## Strategy marketplace

`agents/marketplace.py`. Providers: internal strategies, analyst-board sources and Discord
channels. No payments, no fees, no real money: providers are graded only on the paper
outcomes of their own signals.

Vetting (`VettingPolicy` defaults; owner can tighten):

| Rule                                 | Default |
| ------------------------------------ | ------- |
| Graded signals                       | >= 30   |
| Track record                         | >= 90 days |
| Deflated Sharpe probability          | >= 0.95 (deflated by the number of providers) |
| Hit rate                             | >= 50%  |
| Max drawdown (compounded)            | <= 20%  |
| Halal compliance (passed Shariah gate) | >= 95% |
| Re-vet interval                      | 7 days  |

Statuses: `candidate` -> `vetted` when every rule holds; `vetted` -> `suspended` when any rule
breaks on re-vet; `suspended` -> `vetted` only when every rule holds again. Every result keeps
its reasons. `suspend()` is a sticky manual hold until `reinstate()`. Run `revet_due(now)` on a
schedule.

## Strategy selection

`agents/strategy_selector.py`. Chooses among **vetted** strategies (candidate
`strategy_id` = marketplace `provider_id`) for the current regime (`regimes` snapshot; default
key `trend|volatility`). Walk-forward folds score each candidate in and out of sample on
periods with the same regime label. Right-fit guard:

- **Overfit**: in-sample to out-of-sample Sharpe degradation above 50% rejects.
- **No skill**: excess return over the simple baseline must beat noise, deflated for the
  number of candidates (probability >= 0.95).
- **Selection overfit**: probability of backtest overfitting above 0.5 rejects all.

The best surviving out-of-sample Sharpe wins; if none survives, nothing is selected and the
baseline stays. An optional LLM callable may write `explanation` only; it cannot change
`selected`. A failing or empty model reply falls back to the rule-based explanation.

## Signal intake

`agents/signal_intake.py` parses text such as `BUY BTC/USD 65000 SL 62000 TP 72000` into a
long-only `TradePlan`. Rejected with a reason: `SHORT`, leverage (`10x`, `x20`, `leverage`),
`PERP`, `futures`, `margin`, options, CFDs, binaries and opening `SELL` signals. These count
against the provider's halal compliance. Chatter and malformed plans are ignored, not graded.

The provider never sets size or confidence: confidence is a fixed prior (`0.60` default).
Each accepted candidate runs through `EntryPipeline.decide` in paper mode
(`SignalIntake(mode="paper")` is the only allowed mode). Missing market context blocks the
entry (unknown state). The outcome is recorded in the marketplace under the provider id;
call `Marketplace.grade()` with the realised paper return when the trade closes.

## Discord intake

`ops/discord_intake.py`. Reads one channel through the Discord REST API
(`GET /channels/{id}/messages`). Telegram is not supported (the owner cannot use it).

- Token: environment variable `QUANT_OS_DISCORD_BOT_TOKEN`; never a file in the repo.
- Dedupes by message id and keeps a cursor (`after=`).
- Rate-limited: 30 s minimum poll interval, at most 20 messages per poll, honours HTTP 429
  `retry_after`.
- Optional `allowed_author_ids` filter.
- Every message goes to `SignalIntake`; the reader cannot place orders.

```python
reader = DiscordChannelReader.from_env("123456789012345678", intake)
result = reader.poll()  # call from the paper loop's scheduler
```

## Owner decisions

- Vetting thresholds above (30 signals, 90 days, DSR 0.95, hit rate 50%, drawdown 20%,
  compliance 95%, weekly re-vet).
- Selector guard: 50% degradation, 0.95 skill probability, PBO 0.5.
- Signal confidence prior 0.60 (must clear the plan policy minimum of 0.55).
- When to mark the built-in plugin `halal_reviewed=True`.
