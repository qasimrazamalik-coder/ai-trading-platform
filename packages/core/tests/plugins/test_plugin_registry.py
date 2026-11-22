import numpy as np
import pytest

from quant_os_core.plugins import (
    FORBIDDEN_PERMISSIONS,
    Permission,
    PluginKind,
    PluginManifest,
    PluginRefusedError,
    PluginRegistry,
    PluginStatus,
    StrategyPlugin,
)
from quant_os_core.plugins.builtin import (
    LONG_ONLY_MA_METADATA,
    MANIFEST,
    long_only_ma_crossover,
)


def _manifest(**overrides: object) -> PluginManifest:
    params: dict[str, object] = {
        "name": "acme.sink",
        "version": "0.1.0",
        "kind": PluginKind.ALERT_SINK,
        "halal_reviewed": True,
        "permissions": frozenset({Permission.SEND_ALERTS}),
    }
    params.update(overrides)
    return PluginManifest(**params)  # type: ignore[arg-type]


def test_reviewed_plugin_is_active_and_tradable() -> None:
    registry = PluginRegistry()
    record = registry.register(_manifest(), object(), source="test")
    assert record.status is PluginStatus.ACTIVE
    assert record.tradable
    assert record.source == "test"
    assert "acme.sink" in registry
    assert len(registry) == 1
    assert registry.get("acme.sink") is record
    assert registry.tradable(PluginKind.ALERT_SINK) == (record,)
    assert registry.list(PluginKind.STRATEGY) == ()


def test_unreviewed_plugin_is_research_only() -> None:
    registry = PluginRegistry()
    record = registry.register(_manifest(halal_reviewed=False), object())
    assert record.status is PluginStatus.RESEARCH_ONLY
    assert not record.tradable
    assert registry.list() == (record,)
    assert registry.tradable() == ()


def test_order_placement_permission_is_refused() -> None:
    assert Permission.PLACE_ORDERS in FORBIDDEN_PERMISSIONS
    registry = PluginRegistry()
    bad = _manifest(permissions=frozenset({Permission.PLACE_ORDERS, Permission.READ_NETWORK}))
    with pytest.raises(PluginRefusedError, match="place_orders"):
        registry.register(bad, object())
    assert "acme.sink" not in registry


@pytest.mark.parametrize(
    ("overrides", "component", "message"),
    [
        ({"name": "Bad Name!"}, object(), "invalid plugin name"),
        ({"version": " "}, object(), "version is required"),
        ({"kind": "strategy"}, object(), "unknown plugin kind"),
        ({"permissions": frozenset({"root"})}, object(), "unknown permission"),
        ({"kind": PluginKind.STRATEGY}, lambda: None, "must be StrategyPlugin"),
        ({"kind": PluginKind.INDICATOR}, object(), "must be callable"),
    ],
)
def test_registry_rules(overrides: dict[str, object], component: object, message: str) -> None:
    with pytest.raises(PluginRefusedError, match=message):
        PluginRegistry().register(_manifest(**overrides), component)


def test_duplicates_and_unknown_lookups() -> None:
    registry = PluginRegistry()
    registry.register(_manifest(), object())
    with pytest.raises(PluginRefusedError, match="already registered"):
        registry.register(_manifest(), object())
    with pytest.raises(KeyError, match="unknown plugin"):
        registry.get("nope")


def test_manifest_build_parses_and_fails_closed() -> None:
    manifest = PluginManifest.build(
        name="acme.data",
        version="1",
        kind="data_source",
        halal_reviewed=False,
        permissions=["read_market_data", "read_network"],
    )
    assert manifest.kind is PluginKind.DATA_SOURCE
    assert manifest.permissions == {Permission.READ_MARKET_DATA, Permission.READ_NETWORK}
    with pytest.raises(PluginRefusedError, match="unknown plugin kind"):
        PluginManifest.build(name="a.b", version="1", kind="broker", halal_reviewed=True)
    with pytest.raises(PluginRefusedError, match="unknown permission"):
        PluginManifest.build(
            name="a.b", version="1", kind="indicator", halal_reviewed=True, permissions=["sudo"]
        )


def test_builtin_strategy_is_long_only() -> None:
    prices = np.concatenate([np.linspace(100, 150, 80), np.linspace(150, 90, 80)])
    signals = long_only_ma_crossover(prices, fast_window=5, slow_window=20)
    assert signals.metadata is LONG_ONLY_MA_METADATA
    assert signals.positions.min() >= 0.0
    assert signals.positions.max() == 1.0
    plugin = StrategyPlugin(LONG_ONLY_MA_METADATA, long_only_ma_crossover)
    again = plugin(prices, fast_window=5, slow_window=20)
    assert np.array_equal(again.positions, signals.positions)
    assert MANIFEST.kind is PluginKind.STRATEGY
    assert not MANIFEST.halal_reviewed
