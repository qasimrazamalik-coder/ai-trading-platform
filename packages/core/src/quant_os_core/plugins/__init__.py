"""Plugin ("tentacles") architecture. See docs/PLUGINS.md."""

from quant_os_core.plugins.loader import (
    ENTRY_POINT_GROUP,
    LoadReport,
    PluginRegistrar,
    discover,
    load_directory,
    load_entry_points,
)
from quant_os_core.plugins.registry import (
    FORBIDDEN_PERMISSIONS,
    Permission,
    PluginKind,
    PluginManifest,
    PluginRefusedError,
    PluginRegistry,
    PluginStatus,
    RegisteredPlugin,
    StrategyPlugin,
)

__all__ = [
    "ENTRY_POINT_GROUP",
    "FORBIDDEN_PERMISSIONS",
    "LoadReport",
    "Permission",
    "PluginKind",
    "PluginManifest",
    "PluginRefusedError",
    "PluginRegistrar",
    "PluginRegistry",
    "PluginStatus",
    "RegisteredPlugin",
    "StrategyPlugin",
    "discover",
    "load_directory",
    "load_entry_points",
]
