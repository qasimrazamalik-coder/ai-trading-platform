from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from quant_os_core.plugins import (
    ENTRY_POINT_GROUP,
    PluginKind,
    PluginRegistry,
    PluginStatus,
    discover,
    load_directory,
    load_entry_points,
)
from quant_os_core.plugins.loader import EntryPointLike, PluginRegistrar
from quant_os_core.plugins.registry import Permission, PluginManifest

GOOD_PLUGIN = """
from quant_os_core.plugins.registry import Permission, PluginKind, PluginManifest


def _indicator(values):
    return values


def register(registrar):
    registrar.register(
        PluginManifest(
            name="local.echo",
            version="1.0",
            kind=PluginKind.INDICATOR,
            halal_reviewed=True,
            permissions=frozenset({Permission.READ_MARKET_DATA}),
        ),
        _indicator,
    )
    registrar.register(
        PluginManifest(
            name="local.trader",
            version="1.0",
            kind=PluginKind.SIGNAL_SOURCE,
            halal_reviewed=True,
            permissions=frozenset({Permission.PLACE_ORDERS}),
        ),
        object(),
    )
"""


def _write(directory: Path, name: str, body: str) -> None:
    (directory / name).write_text(body, encoding="utf-8")


def test_local_directory_loads_and_refuses(tmp_path: Path) -> None:
    _write(tmp_path, "good.py", GOOD_PLUGIN)
    _write(tmp_path, "broken.py", "raise RuntimeError('boom')\n")
    _write(tmp_path, "no_hook.py", "VALUE = 1\n")
    _write(tmp_path, "raises.py", "def register(registrar):\n    raise KeyError('x')\n")
    _write(tmp_path, "_private.py", "raise SystemExit('must never run')\n")
    registry = PluginRegistry()
    report = load_directory(registry, tmp_path)
    assert report.loaded == ["local.echo"]
    reasons = dict(report.refused)
    assert "place_orders" in reasons["local.trader"]
    assert reasons["local:broken.py"] == "import failed: RuntimeError"
    assert reasons["local:no_hook.py"] == "no callable register(registrar)"
    assert reasons["local:raises.py"] == "register failed: KeyError"
    assert registry.get("local.echo").source == "local:good.py"
    assert "local.trader" not in registry


def test_missing_directory_is_empty(tmp_path: Path) -> None:
    report = load_directory(PluginRegistry(), tmp_path / "absent")
    assert report.loaded == [] and report.refused == []


@dataclass
class _FakeEntryPoint:
    name: str
    hook: object
    fail: bool = False

    def load(self) -> object:
        if self.fail:
            raise ImportError("missing dependency")
        return self.hook


def _sink_hook(registrar: PluginRegistrar) -> None:
    registrar.register(
        PluginManifest(
            name="pkg.ntfy",
            version="2.0",
            kind=PluginKind.ALERT_SINK,
            halal_reviewed=False,
            permissions=frozenset({Permission.SEND_ALERTS}),
        ),
        object(),
    )


def test_entry_points_load_with_fake_metadata() -> None:
    calls: list[str] = []

    def fake(group: str) -> Iterable[EntryPointLike]:
        calls.append(group)
        return [
            _FakeEntryPoint("ntfy", _sink_hook),
            _FakeEntryPoint("bad", None, fail=True),
            _FakeEntryPoint("notcallable", 42),
        ]

    registry = PluginRegistry()
    report = load_entry_points(registry, entry_points=fake)
    assert calls == [ENTRY_POINT_GROUP]
    assert report.loaded == ["pkg.ntfy"]
    assert dict(report.refused) == {
        "entry_point:bad": "load failed: ImportError",
        "entry_point:notcallable": "no callable register(registrar)",
    }
    assert registry.get("pkg.ntfy").status is PluginStatus.RESEARCH_ONLY


def test_discover_combines_builtin_entry_points_and_directory(tmp_path: Path) -> None:
    _write(tmp_path, "good.py", GOOD_PLUGIN)
    registry = PluginRegistry()
    report = discover(registry, directory=tmp_path, entry_points=lambda group: [])
    assert report.loaded == ["builtin.long_only_ma_crossover", "local.echo"]
    builtin = registry.get("builtin.long_only_ma_crossover")
    assert builtin.source == "builtin"
    assert builtin.status is PluginStatus.RESEARCH_ONLY
    assert registry.tradable(PluginKind.STRATEGY) == ()


def test_discover_without_builtins_uses_installed_metadata() -> None:
    registry = PluginRegistry()
    report = discover(registry, include_builtins=False)
    assert "builtin.long_only_ma_crossover" not in registry
    assert isinstance(report.loaded, list)
