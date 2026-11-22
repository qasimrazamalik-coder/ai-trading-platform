from fastapi.testclient import TestClient

from quant_os_api.main import app


def test_health_endpoint_is_safe_by_default() -> None:
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["live_trading_enabled"] is False


def test_market_data_capabilities_endpoint() -> None:
    with TestClient(app) as client:
        response = client.get("/market-data/capabilities")

    assert response.status_code == 200
    payload = response.json()
    assert "crypto" in payload["asset_classes"]
    assert "equity" in payload["asset_classes"]
    assert "candle" in payload["record_types"]
    assert "split" in payload["corporate_actions"]
    assert "clickhouse" in payload["storage_targets"]


def test_quant_capabilities_endpoint() -> None:
    with TestClient(app) as client:
        response = client.get("/quant/capabilities")

    assert response.status_code == 200
    payload = response.json()
    assert "sharpe" in payload["modules"]["performance"]
    assert "engle_granger" in payload["modules"]["stat_arb"]
    assert payload["conventions"]["var_cvar"] == "positive loss fraction"


def test_backtesting_capabilities() -> None:
    with TestClient(app) as client:
        response = client.get("/backtesting/capabilities")
    assert response.status_code == 200
    payload = response.json()
    assert "event_driven_multi_asset" in payload["engines"]
    assert "market" in payload["order_types"]
    assert "partial_fills" in payload["execution_models"]


def test_right_fit_capabilities() -> None:
    with TestClient(app) as client:
        response = client.get("/right-fit/capabilities")
    assert response.status_code == 200
    payload = response.json()
    assert "deflated_sharpe" in payload["diagnostics"]
    assert "combinatorial_purged_cv" in payload["validation"]
    assert "research_pass" in payload["decisions"]


def test_strategy_capabilities() -> None:
    with TestClient(app) as client:
        response = client.get("/strategies/capabilities")
    assert response.status_code == 200
    payload = response.json()
    assert "trend.ma_crossover" in payload["registered"]
    assert "stat_arb" in payload["families"]
    assert "event_driven_target_adapter" in payload["tooling"]


def test_ml_capabilities() -> None:
    with TestClient(app) as client:
        response = client.get("/ml/capabilities")
    assert response.status_code == 200
    payload = response.json()
    assert "xgboost" in payload["classical"]
    assert "patchtst_style" in payload["deep_learning"]
    assert "walk_forward_tournament" in payload["validation"]
    assert "no_implicit_model_downloads" in payload["governance"]


def test_regime_capabilities() -> None:
    with TestClient(app) as client:
        response = client.get("/regimes/capabilities")
    assert response.status_code == 200
    payload = response.json()
    assert "gaussian_hmm" in payload["unsupervised"]
    assert "dynamic_strategy_weights" in payload["allocation"]


def test_portfolio_risk_capabilities() -> None:
    with TestClient(app) as client:
        response = client.get("/portfolio-risk/capabilities")
    assert response.status_code == 200
    payload = response.json()
    assert "hierarchical_risk_parity" in payload["portfolio_construction"]
    assert "black_litterman" in payload["portfolio_construction"]
    assert "portfolio_heat" in payload["portfolio_controls"]
    assert "halt" in payload["decisions"]


def test_analytics_capabilities() -> None:
    client = TestClient(app)
    response = client.get("/analytics/capabilities")
    assert response.status_code == 200
    payload = response.json()
    assert "asset_correlation" in payload["heatmaps"]
    assert "parameter_stability" in payload["surfaces"]
    assert "explicit_demo_data_banner" in payload["frontend"]


def test_execution_capabilities() -> None:
    from fastapi.testclient import TestClient

    from quant_os_api.main import app

    client = TestClient(app)
    response = client.get("/execution/capabilities")
    assert response.status_code == 200
    data = response.json()
    assert "idempotent_client_order_id" in data["order_management"]
    assert all(not adapter["live_trading"] for adapter in data["adapters"])


def test_intelligence_capabilities() -> None:
    from fastapi.testclient import TestClient

    from quant_os_api.main import app

    client = TestClient(app)
    response = client.get("/intelligence/capabilities")
    assert response.status_code == 200
    data = response.json()
    assert "independent_corroboration" in data["credibility"]


def test_terminal_demo_is_explicitly_synthetic_and_live_off() -> None:
    from fastapi.testclient import TestClient

    from quant_os_api.main import app

    client = TestClient(app)
    response = client.get("/terminal/demo-state")
    assert response.status_code == 200
    data = response.json()
    assert data["synthetic"] is True
    assert data["mode"] == "paper_demo"
    assert data["live_trading_enabled"] is False
    assert len(data["candles"]) >= 20
    assert data["bids"] and data["asks"]


def test_demo_market_websocket_stream() -> None:
    from fastapi.testclient import TestClient

    from quant_os_api.main import app

    client = TestClient(app)
    with client.websocket_connect("/ws/demo/market") as socket:
        payload = socket.receive_json()
        assert payload["type"] == "candle"
        assert payload["synthetic"] is True
        assert payload["symbol"] == "BTCUSDT"


def test_paper_order_api_executes_real_paper_fill_and_is_idempotent() -> None:
    client = TestClient(app)
    terminal = client.get("/terminal/demo-state")
    assert terminal.status_code == 200
    mark = terminal.json()["last_price"]

    payload = {
        "client_order_id": "api-release-paper-1",
        "symbol": "BTCUSDT",
        "side": "buy",
        "quantity": "0.01",
        "order_type": "market",
    }
    first = client.post("/paper/orders", json=payload)
    second = client.post("/paper/orders", json=payload)
    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json() == second.json()
    data = first.json()
    assert data["status"] == "filled"
    assert data["filled_quantity"] == "0.01"
    assert float(data["average_fill_price"]) >= mark

    account = client.get("/paper/account")
    assert account.status_code == 200
    positions = account.json()["positions"]
    btc = next(item for item in positions if item["symbol"] == "BTCUSDT")
    assert float(btc["quantity"]) >= 0.01


def test_paper_order_api_rejects_binary_float_and_bad_order_shape() -> None:
    client = TestClient(app)
    float_payload = {
        "client_order_id": "api-release-float",
        "symbol": "BTCUSDT",
        "side": "buy",
        "quantity": 0.01,
        "order_type": "market",
    }
    assert client.post("/paper/orders", json=float_payload).status_code == 422

    bad_limit = {
        "client_order_id": "api-release-bad-limit",
        "symbol": "BTCUSDT",
        "side": "buy",
        "quantity": "0.01",
        "order_type": "limit",
    }
    response = client.post("/paper/orders", json=bad_limit)
    assert response.status_code == 422


def test_paper_limit_order_can_be_cancelled() -> None:
    client = TestClient(app)
    payload = {
        "client_order_id": "api-release-cancel-1",
        "symbol": "BTCUSDT",
        "side": "buy",
        "quantity": "0.01",
        "order_type": "limit",
        "limit_price": "1",
    }
    response = client.post("/paper/orders", json=payload)
    assert response.status_code == 200
    assert response.json()["status"] == "pending"
    cancelled = client.delete("/paper/orders/api-release-cancel-1")
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"
