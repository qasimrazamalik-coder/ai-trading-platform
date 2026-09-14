from quant_os_core.market_data.models import Candle
from quant_os_core.market_data.replay import MarketDataReplay
from quant_os_core.market_data.storage import InMemoryCandleStore


def test_store_filters_by_instrument(minute_candles: list[Candle], btc) -> None:
    store = InMemoryCandleStore()
    store.append(minute_candles)
    result = store.read(
        btc,
        "1m",
        start=minute_candles[1].timestamp,
        end=minute_candles[4].timestamp,
    )
    assert result == minute_candles[1:4]


def test_replay_orders_records_deterministically(minute_candles: list[Candle]) -> None:
    replay = MarketDataReplay(reversed(minute_candles))
    result = list(replay)
    assert result == minute_candles
    assert len(replay) == 5
