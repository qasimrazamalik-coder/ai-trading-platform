"""Quant OS control-plane API."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from decimal import Decimal

from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator, model_validator

from quant_os_core.backtesting.models import OrderSide, OrderStatus, OrderType, TimeInForce
from quant_os_core.instruments import AssetClass, InstrumentId
from quant_os_core.logging import configure_logging
from quant_os_core.market_data.corporate_actions import CorporateActionType
from quant_os_core.market_data.models import MarketDataKind
from quant_os_core.execution import BrokerOrderRequest, default_adapter_catalog
from quant_os_core.execution.models import BrokerOrderState
from quant_os_core.intelligence import SourceTier
from quant_os_core.settings import get_settings
from quant_os_core.strategies import create_default_registry
from quant_os_api.paper_session import PaperTradingSession


class MarketDataCapabilitiesResponse(BaseModel):
    asset_classes: list[str]
    record_types: list[str]
    fixed_timeframe_units: list[str]
    corporate_actions: list[str]
    storage_targets: list[str]


class QuantCapabilitiesResponse(BaseModel):
    modules: dict[str, list[str]]
    conventions: dict[str, str]


class BacktestingCapabilitiesResponse(BaseModel):
    engines: list[str]
    order_types: list[str]
    time_in_force: list[str]
    execution_models: list[str]
    accounting: list[str]


class RightFitCapabilitiesResponse(BaseModel):
    validation: list[str]
    diagnostics: list[str]
    robustness: list[str]
    decisions: list[str]


class StrategyCapabilitiesResponse(BaseModel):
    registered: list[str]
    families: list[str]
    tooling: list[str]


class MLCapabilitiesResponse(BaseModel):
    classical: list[str]
    deep_learning: list[str]
    foundation_models: list[str]
    validation: list[str]
    governance: list[str]


class RegimeCapabilitiesResponse(BaseModel):
    deterministic: list[str]
    unsupervised: list[str]
    dynamics: list[str]
    allocation: list[str]


class PortfolioRiskCapabilitiesResponse(BaseModel):
    portfolio_construction: list[str]
    sizing: list[str]
    pre_trade_controls: list[str]
    portfolio_controls: list[str]
    decisions: list[str]


class AnalyticsCapabilitiesResponse(BaseModel):
    heatmaps: list[str]
    seasonality: list[str]
    distributions: list[str]
    series: list[str]
    surfaces: list[str]
    frontend: list[str]




class ExecutionCapabilitiesResponse(BaseModel):
    order_management: list[str]
    reconciliation: list[str]
    adapters: list[dict[str, object]]
    safety: list[str]


class IntelligenceCapabilitiesResponse(BaseModel):
    ingestion: list[str]
    credibility: list[str]
    nlp: list[str]
    outputs: list[str]


class TerminalCandle(BaseModel):
    time: int
    open: float
    high: float
    low: float
    close: float
    volume: float


class TerminalBookLevel(BaseModel):
    price: float
    quantity: float


class TerminalNews(BaseModel):
    headline: str
    source: str
    credibility: float = Field(ge=0, le=1)
    sentiment: float = Field(ge=-1, le=1)
    age: str


class TerminalPosition(BaseModel):
    symbol: str
    side: str
    quantity: float
    entry: float
    mark: float
    pnl: float


class TerminalStateResponse(BaseModel):
    mode: str
    synthetic: bool
    symbol: str
    venue: str
    timeframe: str
    last_price: float
    change_pct: float
    candles: list[TerminalCandle]
    bids: list[TerminalBookLevel]
    asks: list[TerminalBookLevel]
    news: list[TerminalNews]
    positions: list[TerminalPosition]
    registered_strategies: int
    risk_status: str
    live_trading_enabled: bool


class PaperOrderCommand(BaseModel):
    client_order_id: str = Field(min_length=1, max_length=128)
    symbol: str = Field(min_length=1, max_length=80)
    side: OrderSide
    quantity: Decimal = Field(gt=0)
    order_type: OrderType = OrderType.MARKET
    limit_price: Decimal | None = Field(default=None, gt=0)
    stop_price: Decimal | None = Field(default=None, gt=0)

    @field_validator("quantity", "limit_price", "stop_price", mode="before")
    @classmethod
    def reject_binary_float(cls, value: object) -> object:
        if isinstance(value, float):
            raise ValueError("paper-order financial values must be sent as strings or integers")
        return value

    @model_validator(mode="after")
    def validate_prices(self) -> "PaperOrderCommand":
        if self.order_type is OrderType.LIMIT and self.limit_price is None:
            raise ValueError("limit orders require limit_price")
        if self.order_type is OrderType.STOP and self.stop_price is None:
            raise ValueError("stop orders require stop_price")
        if self.order_type is not OrderType.LIMIT and self.limit_price is not None:
            raise ValueError("limit_price is only valid for limit orders")
        if self.order_type is not OrderType.STOP and self.stop_price is not None:
            raise ValueError("stop_price is only valid for stop orders")
        return self


class PaperOrderResponse(BaseModel):
    client_order_id: str
    venue_order_id: str
    symbol: str
    status: str
    submitted_quantity: str
    filled_quantity: str
    average_fill_price: str | None


class PaperAccountPosition(BaseModel):
    symbol: str
    quantity: str
    average_price: str
    mark_price: str


class PaperAccountResponse(BaseModel):
    cash: str
    equity: str
    buying_power: str
    positions: list[PaperAccountPosition]


class HealthResponse(BaseModel):
    status: str
    service: str
    live_trading_enabled: bool


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level)
    yield


app = FastAPI(
    title="Quant OS API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:3000",
        "http://localhost:3000",
        "http://127.0.0.1:8080",
        "http://localhost:8080",
        "http://127.0.0.1:4173",
        "http://localhost:4173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type"],
)

_paper_session = PaperTradingSession()


def _paper_instrument(symbol: str) -> InstrumentId:
    return InstrumentId(venue="PAPER", symbol=symbol)


def _paper_order_response(order: BrokerOrderState) -> PaperOrderResponse:
    return PaperOrderResponse(
        client_order_id=order.client_order_id,
        venue_order_id=order.venue_order_id,
        symbol=order.instrument_id.symbol,
        status=order.status.value,
        submitted_quantity=str(order.submitted_quantity),
        filled_quantity=str(order.filled_quantity),
        average_fill_price=(
            None if order.average_fill_price is None else str(order.average_fill_price)
        ),
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(
        status="ok",
        service=settings.service_name,
        live_trading_enabled=settings.live_trading_enabled,
    )


@app.get("/market-data/capabilities", response_model=MarketDataCapabilitiesResponse)
def market_data_capabilities() -> MarketDataCapabilitiesResponse:
    return MarketDataCapabilitiesResponse(
        asset_classes=[item.value for item in AssetClass],
        record_types=[item.value for item in MarketDataKind],
        fixed_timeframe_units=["seconds", "minutes", "hours", "days", "weeks"],
        corporate_actions=[item.value for item in CorporateActionType],
        storage_targets=["postgresql", "clickhouse", "parquet", "memory"],
    )


@app.get("/quant/capabilities", response_model=QuantCapabilitiesResponse)
def quant_capabilities() -> QuantCapabilitiesResponse:
    return QuantCapabilitiesResponse(
        modules={
            "returns": ["simple", "log", "wealth", "drawdown"],
            "statistics": ["variance", "covariance", "correlation", "zscore", "autocorrelation"],
            "volatility": ["annualized", "rolling", "ewma"],
            "performance": ["sharpe", "sortino", "calmar", "omega", "profit_factor"],
            "tail_risk": ["historical_var", "historical_cvar", "gaussian_var", "gaussian_cvar"],
            "regression": ["ols", "beta", "alpha", "tracking_error", "information_ratio"],
            "stat_arb": ["engle_granger", "mean_reversion_half_life"],
            "sizing": ["kelly", "fractional_kelly", "volatility_targeting", "fixed_fractional"],
            "portfolio": ["covariance", "correlation", "risk_contribution", "nearest_psd"],
        },
        conventions={
            "var_cvar": "positive loss fraction",
            "drawdown": "non-positive fraction from running wealth peak",
            "returns": "simple unless function name explicitly says log",
        },
    )


@app.get("/backtesting/capabilities", response_model=BacktestingCapabilitiesResponse)
def backtesting_capabilities() -> BacktestingCapabilitiesResponse:
    return BacktestingCapabilitiesResponse(
        engines=["vectorized", "event_driven_multi_asset"],
        order_types=[item.value for item in OrderType],
        time_in_force=[item.value for item in TimeInForce],
        execution_models=[
            "next_bar_latency",
            "spread",
            "fixed_slippage",
            "sqrt_volume_impact",
            "partial_fills",
            "fees",
            "financing",
        ],
        accounting=["decimal_cash", "long_short_positions", "mark_to_market", "equity_ledger"],
    )


@app.get("/right-fit/capabilities", response_model=RightFitCapabilitiesResponse)
def right_fit_capabilities() -> RightFitCapabilitiesResponse:
    return RightFitCapabilitiesResponse(
        validation=[
            "purged_kfold",
            "embargo",
            "walk_forward",
            "combinatorial_purged_cv",
            "label_horizon_purging",
        ],
        diagnostics=[
            "probabilistic_sharpe",
            "deflated_sharpe",
            "probability_backtest_overfitting",
            "parameter_surface_stability",
            "underfit_rightfit_overfit",
            "walk_forward_parameter_selection",
        ],
        robustness=["moving_block_bootstrap", "monte_carlo", "sidak_multiple_testing"],
        decisions=["research_pass", "review", "reject"],
    )


@app.get("/strategies/capabilities", response_model=StrategyCapabilitiesResponse)
def strategy_capabilities() -> StrategyCapabilitiesResponse:
    registry = create_default_registry()
    return StrategyCapabilitiesResponse(
        registered=[item.strategy_id for item in registry.list_metadata()],
        families=[
            "trend",
            "momentum",
            "mean_reversion",
            "breakout",
            "stat_arb",
            "carry_basis",
            "microstructure",
            "volatility",
            "market_making",
            "fundamental_screening",
            "event_news",
            "ensemble",
        ],
        tooling=[
            "indicator_library",
            "strategy_registry",
            "vectorized_research_bridge",
            "event_driven_target_adapter",
            "right_fit_diagnostic_bridge",
        ],
    )


@app.get("/ml/capabilities", response_model=MLCapabilitiesResponse)
def ml_capabilities() -> MLCapabilitiesResponse:
    return MLCapabilitiesResponse(
        classical=[
            "ridge",
            "logistic_regression",
            "random_forest",
            "xgboost",
            "lightgbm",
            "catboost",
        ],
        deep_learning=[
            "lstm",
            "gru",
            "temporal_convolution",
            "transformer",
            "patchtst_style",
            "itransformer_style",
        ],
        foundation_models=["chronos_adapter", "timesfm_adapter", "kronos_adapter"],
        validation=[
            "chronological_split",
            "walk_forward_tournament",
            "train_only_standardization",
            "baseline_comparison",
            "validation_early_stopping",
        ],
        governance=[
            "versioned_model_registry",
            "training_data_hash",
            "jsonl_experiment_tracking",
            "lazy_checkpoint_loading",
            "no_implicit_model_downloads",
        ],
    )


@app.get("/regimes/capabilities", response_model=RegimeCapabilitiesResponse)
def regime_capabilities() -> RegimeCapabilitiesResponse:
    return RegimeCapabilitiesResponse(
        deterministic=[
            "trend",
            "volatility",
            "risk_on_off",
            "liquidity",
            "correlation",
            "macro_growth_inflation",
        ],
        unsupervised=["diagonal_gaussian_mixture", "gaussian_hmm"],
        dynamics=[
            "cusum_change_points",
            "transition_matrix",
            "stationary_distribution",
            "entropy_confidence",
            "volatility_clustering",
        ],
        allocation=[
            "strategy_regime_compatibility",
            "dynamic_strategy_weights",
            "composite_regime_weights",
            "regime_strategy_heatmap_data",
        ],
    )


@app.get("/portfolio-risk/capabilities", response_model=PortfolioRiskCapabilitiesResponse)
def portfolio_risk_capabilities() -> PortfolioRiskCapabilitiesResponse:
    return PortfolioRiskCapabilitiesResponse(
        portfolio_construction=[
            "minimum_variance",
            "maximum_sharpe",
            "inverse_volatility",
            "risk_parity",
            "hierarchical_risk_parity",
            "hierarchical_equal_risk_contribution",
            "black_litterman",
            "cvar_minimization",
        ],
        sizing=["fractional_kelly", "volatility_targeting", "fixed_fractional"],
        pre_trade_controls=[
            "position_limit",
            "gross_leverage",
            "net_leverage",
            "group_concentration",
            "correlation_concentration",
            "shorting_policy",
        ],
        portfolio_controls=[
            "portfolio_heat",
            "daily_loss_brake",
            "drawdown_brake",
            "kill_switch",
            "stale_data_halt",
            "stress_scenarios",
        ],
        decisions=["approve", "resize", "reject", "halt"],
    )

@app.get("/analytics/capabilities", response_model=AnalyticsCapabilitiesResponse)
def analytics_capabilities() -> AnalyticsCapabilitiesResponse:
    return AnalyticsCapabilitiesResponse(
        heatmaps=[
            "asset_correlation",
            "strategy_correlation",
            "factor_exposure",
            "risk_contribution",
            "regime_strategy",
        ],
        seasonality=[
            "hour_of_day",
            "weekday",
            "month",
            "pnl_calendar",
        ],
        distributions=[
            "monte_carlo_terminal_returns",
            "drawdowns",
            "generic_histogram",
        ],
        series=[
            "drawdown",
            "rolling_volatility",
            "liquidity_score",
            "annualized_funding",
        ],
        surfaces=[
            "parameter_stability",
            "walk_forward_performance",
        ],
        frontend=[
            "responsive_terminal_shell",
            "typed_heatmap_component",
            "equity_chart",
            "pnl_calendar",
            "regime_sidecar",
            "explicit_demo_data_banner",
        ],
    )

@app.get("/execution/capabilities", response_model=ExecutionCapabilitiesResponse)
def execution_capabilities() -> ExecutionCapabilitiesResponse:
    catalog = default_adapter_catalog()
    return ExecutionCapabilitiesResponse(
        order_management=[
            "idempotent_client_order_id",
            "market_limit_stop",
            "gtc_ioc_contracts",
            "paper_fills",
            "decimal_accounting",
        ],
        reconciliation=[
            "order_status",
            "filled_quantity",
            "position_quantity",
            "halt_on_critical_mismatch",
        ],
        adapters=[
            {
                "adapter_id": item.adapter_id,
                "markets": [market.value for market in item.markets],
                "market_data": item.market_data,
                "paper_trading": item.paper_trading,
                "live_trading": item.live_trading,
                "websocket": item.websocket,
                "notes": item.notes,
            }
            for item in catalog
        ],
        safety=[
            "live_disabled_by_default",
            "no_embedded_credentials",
            "risk_gate_before_execution",
            "paper_first",
        ],
    )


@app.get("/intelligence/capabilities", response_model=IntelligenceCapabilitiesResponse)
def intelligence_capabilities() -> IntelligenceCapabilitiesResponse:
    return IntelligenceCapabilitiesResponse(
        ingestion=["typed_news_items", "entity_tags", "headline_deduplication"],
        credibility=[
            "source_tiers",
            "primary_source_bonus",
            "independent_corroboration",
            "contradiction_penalty",
        ],
        nlp=["transparent_lexicon_sentiment", "pluggable_learned_model_interface"],
        outputs=["credibility_score", "confidence", "sentiment", "risk_adjusted_impact"],
    )


@app.post("/paper/orders", response_model=PaperOrderResponse)
def submit_paper_order(command: PaperOrderCommand) -> PaperOrderResponse:
    request = BrokerOrderRequest(
        client_order_id=command.client_order_id,
        instrument_id=_paper_instrument(command.symbol),
        side=command.side,
        quantity=command.quantity,
        order_type=command.order_type,
        limit_price=command.limit_price,
        stop_price=command.stop_price,
    )
    try:
        state = _paper_session.submit(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _paper_order_response(state)


@app.get("/paper/orders", response_model=list[PaperOrderResponse])
def paper_orders() -> list[PaperOrderResponse]:
    return [_paper_order_response(state) for state in _paper_session.orders()]


@app.delete("/paper/orders/{client_order_id}", response_model=PaperOrderResponse)
def cancel_paper_order(client_order_id: str) -> PaperOrderResponse:
    try:
        state = _paper_session.cancel(client_order_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="paper order not found") from exc
    return _paper_order_response(state)


@app.get("/paper/account", response_model=PaperAccountResponse)
def paper_account() -> PaperAccountResponse:
    account = _paper_session.account()
    return PaperAccountResponse(
        cash=str(account.cash),
        equity=str(account.equity),
        buying_power=str(account.buying_power),
        positions=[
            PaperAccountPosition(
                symbol=position.instrument_id.symbol,
                quantity=str(position.quantity),
                average_price=str(position.average_price),
                mark_price=str(position.mark_price),
            )
            for position in account.positions
        ],
    )


def _terminal_demo_state() -> TerminalStateResponse:
    closes = [
        63810, 64020, 63940, 64360, 64610, 64520, 64980, 65220, 65110, 65570,
        65840, 65720, 66130, 66420, 66310, 66850, 67120, 66980, 67420, 67820,
        67640, 68110, 68480, 68260, 68740, 69020, 68880, 69240, 69510, 69780,
    ]
    candles: list[TerminalCandle] = []
    previous = float(closes[0] - 120)
    for index, close in enumerate(closes):
        open_price = previous
        close_price = float(close)
        high = max(open_price, close_price) + 120 + (index % 4) * 35
        low = min(open_price, close_price) - 95 - (index % 3) * 30
        candles.append(
            TerminalCandle(
                time=1_789_000_000 + index * 3600,
                open=open_price,
                high=high,
                low=low,
                close=close_price,
                volume=740 + ((index * 83) % 560),
            )
        )
        previous = close_price
    last = float(closes[-1])
    _paper_session.update_mark(_paper_instrument("BTCUSDT"), Decimal(str(last)))
    bids = [TerminalBookLevel(price=last - i * 18 - 2, quantity=1.4 + i * 0.42) for i in range(8)]
    asks = [TerminalBookLevel(price=last + i * 18 + 2, quantity=1.2 + i * 0.39) for i in range(8)]
    return TerminalStateResponse(
        mode="paper_demo",
        synthetic=True,
        symbol="BTCUSDT",
        venue="PAPER",
        timeframe="1h",
        last_price=last,
        change_pct=3.14,
        candles=candles,
        bids=bids,
        asks=asks,
        news=[
            TerminalNews(
                headline="ETF flow report shows another week of net digital-asset inflows",
                source="DemoWire",
                credibility=0.88,
                sentiment=0.54,
                age="8m",
            ),
            TerminalNews(
                headline="Macro desk flags higher real yields ahead of policy update",
                source="DemoMacro",
                credibility=0.81,
                sentiment=-0.31,
                age="21m",
            ),
            TerminalNews(
                headline=(
                    "Unverified social post claims exchange disruption; "
                    "no primary confirmation"
                ),
                source="Community demo",
                credibility=0.34,
                sentiment=-0.72,
                age="32m",
            ),
        ],
        positions=[
            TerminalPosition(
                symbol=position.instrument_id.symbol,
                side="LONG" if position.quantity > 0 else "SHORT",
                quantity=float(abs(position.quantity)),
                entry=float(position.average_price),
                mark=float(position.mark_price),
                pnl=float((position.mark_price - position.average_price) * position.quantity),
            )
            for position in _paper_session.account().positions
        ],
        registered_strategies=len(create_default_registry().list_metadata()),
        risk_status="NORMAL",
        live_trading_enabled=get_settings().live_trading_enabled,
    )


@app.get("/terminal/demo-state", response_model=TerminalStateResponse)
def terminal_demo_state() -> TerminalStateResponse:
    """Synthetic paper-terminal state for UI verification and tutorials."""
    return _terminal_demo_state()


@app.websocket("/ws/demo/market")
async def websocket_demo_market(websocket: WebSocket) -> None:
    await websocket.accept()
    state = _terminal_demo_state()
    try:
        for candle in state.candles[-5:]:
            await websocket.send_json(
                {
                    "type": "candle",
                    "symbol": state.symbol,
                    "timeframe": state.timeframe,
                    "data": candle.model_dump(),
                    "synthetic": True,
                }
            )
    finally:
        await websocket.close()

