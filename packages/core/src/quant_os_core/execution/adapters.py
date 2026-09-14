"""Explicit broker/data adapter capability declarations.

These classes describe supported integration surfaces without pretending undocumented or
unavailable venue APIs are connected. Live credentials are never stored here.
"""

from __future__ import annotations

from dataclasses import dataclass

from quant_os_core.instruments import AssetClass


@dataclass(frozen=True, slots=True)
class AdapterCapability:
    adapter_id: str
    markets: tuple[AssetClass, ...]
    market_data: bool
    paper_trading: bool
    live_trading: bool
    websocket: bool
    notes: str


def default_adapter_catalog() -> tuple[AdapterCapability, ...]:
    return (
        AdapterCapability(
            "paper",
            tuple(AssetClass),
            True,
            True,
            False,
            True,
            "Built-in deterministic paper adapter; no credentials required.",
        ),
        AdapterCapability(
            "ccxt_style_crypto",
            (AssetClass.CRYPTO,),
            True,
            True,
            False,
            True,
            "Interface target for documented exchange APIs; live implementation is opt-in.",
        ),
        AdapterCapability(
            "mt5_style_fx_cfd",
            (AssetClass.FOREX, AssetClass.CFD, AssetClass.COMMODITY),
            True,
            True,
            False,
            True,
            "Interface target for MetaTrader-compatible brokers where API access is permitted.",
        ),
        AdapterCapability(
            "ibkr_style_global",
            (
                AssetClass.EQUITY,
                AssetClass.ETF,
                AssetClass.FUTURE,
                AssetClass.OPTION,
                AssetClass.FOREX,
            ),
            True,
            True,
            False,
            True,
            "Global multi-asset adapter contract; live trading requires user configuration.",
        ),
        AdapterCapability(
            "psx_broker",
            (AssetClass.EQUITY, AssetClass.INDEX),
            True,
            True,
            False,
            False,
            (
                "PSX execution remains broker-specific and disabled until "
                "a documented API is configured."
            ),
        ),
    )
