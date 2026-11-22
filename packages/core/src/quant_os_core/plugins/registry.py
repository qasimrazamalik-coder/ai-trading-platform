"""Plugin ("tentacle") registry.

A plugin adds a strategy, an indicator, a data source, a signal source or an alert sink. It
never adds a way to place orders: orders only leave through the entry pipeline (Shariah gate,
risk engine, edge filter, protections). The registry enforces that and two more rules:

* a plugin that asks for order placement, or for a permission this code does not know, is
  refused outright (fail closed);
* a plugin whose ``halal_reviewed`` flag is false is registered as *research only*: it can be
  backtested and inspected, but :meth:`PluginRegistry.tradable` never returns it.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import StrEnum

from quant_os_core.strategies.base import StrategyMetadata, StrategySignals


class PluginKind(StrEnum):
    STRATEGY = "strategy"
    INDICATOR = "indicator"
    DATA_SOURCE = "data_source"
    SIGNAL_SOURCE = "signal_source"
    ALERT_SINK = "alert_sink"


class Permission(StrEnum):
    READ_MARKET_DATA = "read_market_data"
    READ_NETWORK = "read_network"
    READ_SIGNALS = "read_signals"
    SEND_ALERTS = "send_alerts"
    WRITE_BOARD = "write_board"
    PLACE_ORDERS = "place_orders"


FORBIDDEN_PERMISSIONS: frozenset[Permission] = frozenset({Permission.PLACE_ORDERS})

_NAME = re.compile(r"^[a-z0-9][a-z0-9_.-]{1,63}$")
_CALLABLE_KINDS = frozenset({PluginKind.STRATEGY, PluginKind.INDICATOR})


class PluginRefusedError(ValueError):
    """Raised when a plugin breaks a registry rule; the plugin is not registered."""


class PluginStatus(StrEnum):
    ACTIVE = "active"
    RESEARCH_ONLY = "research_only"


@dataclass(frozen=True, slots=True)
class PluginManifest:
    name: str
    version: str
    kind: PluginKind
    halal_reviewed: bool
    permissions: frozenset[Permission] = frozenset()
    description: str = ""

    @classmethod
    def build(
        cls,
        *,
        name: str,
        version: str,
        kind: str,
        halal_reviewed: bool,
        permissions: Iterable[str] = (),
        description: str = "",
    ) -> PluginManifest:
        """Parse a manifest from plain strings, refusing anything unknown."""

        try:
            parsed_kind = PluginKind(kind)
        except ValueError as exc:
            raise PluginRefusedError(f"unknown plugin kind: {kind!r}") from exc
        parsed: set[Permission] = set()
        for item in permissions:
            try:
                parsed.add(Permission(item))
            except ValueError as exc:
                raise PluginRefusedError(f"unknown permission requested: {item!r}") from exc
        return cls(
            name=name,
            version=version,
            kind=parsed_kind,
            halal_reviewed=halal_reviewed,
            permissions=frozenset(parsed),
            description=description,
        )


@dataclass(frozen=True, slots=True)
class StrategyPlugin:
    """Component shape for ``PluginKind.STRATEGY``: metadata plus a signal factory."""

    metadata: StrategyMetadata
    factory: Callable[..., StrategySignals]

    def __call__(self, *args: object, **kwargs: object) -> StrategySignals:
        return self.factory(*args, **kwargs)


@dataclass(frozen=True, slots=True)
class RegisteredPlugin:
    manifest: PluginManifest
    component: object
    status: PluginStatus
    source: str

    @property
    def tradable(self) -> bool:
        return self.status is PluginStatus.ACTIVE


@dataclass(slots=True)
class PluginRegistry:
    _plugins: dict[str, RegisteredPlugin] = field(default_factory=dict)

    def register(
        self, manifest: PluginManifest, component: object, *, source: str = "builtin"
    ) -> RegisteredPlugin:
        if not _NAME.fullmatch(manifest.name):
            raise PluginRefusedError(f"invalid plugin name: {manifest.name!r}")
        if not manifest.version.strip():
            raise PluginRefusedError(f"{manifest.name}: version is required")
        if not isinstance(manifest.kind, PluginKind):
            raise PluginRefusedError(f"{manifest.name}: unknown plugin kind")
        unknown = [p for p in manifest.permissions if not isinstance(p, Permission)]
        if unknown:
            raise PluginRefusedError(f"{manifest.name}: unknown permission requested")
        forbidden = sorted(p.value for p in manifest.permissions & FORBIDDEN_PERMISSIONS)
        if forbidden:
            raise PluginRefusedError(
                f"{manifest.name}: requests {', '.join(forbidden)}; orders only go through "
                "the entry pipeline"
            )
        if manifest.name in self._plugins:
            raise PluginRefusedError(f"plugin already registered: {manifest.name}")
        if manifest.kind is PluginKind.STRATEGY and not isinstance(component, StrategyPlugin):
            raise PluginRefusedError(f"{manifest.name}: strategy plugins must be StrategyPlugin")
        if manifest.kind in _CALLABLE_KINDS and not callable(component):
            raise PluginRefusedError(f"{manifest.name}: {manifest.kind} must be callable")
        status = PluginStatus.ACTIVE if manifest.halal_reviewed else PluginStatus.RESEARCH_ONLY
        record = RegisteredPlugin(manifest, component, status, source)
        self._plugins[manifest.name] = record
        return record

    def get(self, name: str) -> RegisteredPlugin:
        try:
            return self._plugins[name]
        except KeyError as exc:
            raise KeyError(f"unknown plugin: {name}") from exc

    def list(self, kind: PluginKind | None = None) -> tuple[RegisteredPlugin, ...]:
        items = sorted(self._plugins.values(), key=lambda item: item.manifest.name)
        return tuple(p for p in items if kind is None or p.manifest.kind is kind)

    def tradable(self, kind: PluginKind | None = None) -> tuple[RegisteredPlugin, ...]:
        """Plugins allowed to feed paper decisions: halal-reviewed only."""

        return tuple(p for p in self.list(kind) if p.tradable)

    def __contains__(self, name: object) -> bool:
        return name in self._plugins

    def __len__(self) -> int:
        return len(self._plugins)
