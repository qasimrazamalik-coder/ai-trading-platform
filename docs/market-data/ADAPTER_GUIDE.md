# Market Data Adapter Guide

Implement `MarketDataProvider`; do not expose an exchange SDK directly to Quant OS.

An adapter must:

1. map provider symbols into `InstrumentId(venue, symbol)`;
2. convert all timestamps to timezone-aware UTC;
3. emit canonical `Decimal` price/quantity fields;
4. declare instrument metadata and precision;
5. preserve provider trade/order-book sequence ids where available;
6. pass records through `MarketDataValidator` before research/live consumption;
7. never fill missing bars silently unless an explicit repair policy is requested.

Adapters for Binance, Bybit, MetaTrader/FX, IBKR/global equities and PSX can therefore
share the same research/backtest/risk interfaces without coupling the core to any vendor.
