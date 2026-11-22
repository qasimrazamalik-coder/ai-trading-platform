"""Plugin discovery: Python entry points and a local ``plugins/`` directory.

Both sources expose a ``register(registrar)`` callable. The registrar records which source
each plugin came from and turns a refusal into a report line instead of aborting discovery,
so one bad plugin cannot stop the others from loading (and cannot slip past the rules).

Loading a module runs its code. The local directory must be owner-controlled; discovery is
opt-in and never runs at import time.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Protocol

from quant_os_core.plugins.registry import (
    PluginManifest,
    PluginRefusedError,
    PluginRegistry,
    RegisteredPlugin,
)

ENTRY_POINT_GROUP = "quant_os.plugins"


class EntryPointLike(Protocol):
    name: str

    def load(self) -> object: ...


EntryPointsFn = Callable[[str], Iterable[EntryPointLike]]


def _installed_entry_points(group: str) -> Iterable[EntryPointLike]:
    return metadata.entry_points(group=group)


@dataclass(slots=True)
class LoadReport:
    loaded: list[str] = field(default_factory=list)
    refused: list[tuple[str, str]] = field(default_factory=list)

    def merge(self, other: LoadReport) -> LoadReport:
        self.loaded.extend(other.loaded)
        self.refused.extend(other.refused)
        return self


@dataclass(slots=True)
class PluginRegistrar:
    """What a plugin's ``register`` function receives."""

    registry: PluginRegistry
    source: str
    report: LoadReport

    def register(self, manifest: PluginManifest, component: object) -> RegisteredPlugin | None:
        try:
            record = self.registry.register(manifest, component, source=self.source)
        except PluginRefusedError as exc:
            self.report.refused.append((manifest.name, str(exc)))
            return None
        self.report.loaded.append(manifest.name)
        return record


def _run_register(hook: object, registrar: PluginRegistrar, label: str) -> None:
    if not callable(hook):
        registrar.report.refused.append((label, "no callable register(registrar)"))
        return
    try:
        hook(registrar)
    except Exception as exc:  # a broken plugin is reported, never trusted
        registrar.report.refused.append((label, f"register failed: {type(exc).__name__}"))


def load_entry_points(
    registry: PluginRegistry, *, entry_points: EntryPointsFn = _installed_entry_points
) -> LoadReport:
    report = LoadReport()
    for point in entry_points(ENTRY_POINT_GROUP):
        label = f"entry_point:{point.name}"
        try:
            hook = point.load()
        except Exception as exc:
            report.refused.append((label, f"load failed: {type(exc).__name__}"))
            continue
        _run_register(hook, PluginRegistrar(registry, label, report), label)
    return report


def load_directory(registry: PluginRegistry, directory: Path) -> LoadReport:
    report = LoadReport()
    if not directory.is_dir():
        return report
    for path in sorted(directory.glob("*.py")):
        if path.name.startswith("_"):
            continue
        label = f"local:{path.name}"
        spec = importlib.util.spec_from_file_location(f"quant_os_local_plugin_{path.stem}", path)
        if spec is None or spec.loader is None:  # pragma: no cover - only odd filesystems
            report.refused.append((label, "cannot import"))
            continue
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            report.refused.append((label, f"import failed: {type(exc).__name__}"))
            continue
        _run_register(
            getattr(module, "register", None), PluginRegistrar(registry, label, report), label
        )
    return report


def discover(
    registry: PluginRegistry,
    *,
    directory: Path | None = None,
    entry_points: EntryPointsFn = _installed_entry_points,
    include_builtins: bool = True,
) -> LoadReport:
    """Load built-ins, then installed entry points, then the local directory."""

    report = LoadReport()
    if include_builtins:
        from quant_os_core.plugins.builtin import register as register_builtin

        register_builtin(PluginRegistrar(registry, "builtin", report))
    report.merge(load_entry_points(registry, entry_points=entry_points))
    if directory is not None:
        report.merge(load_directory(registry, directory))
    return report
