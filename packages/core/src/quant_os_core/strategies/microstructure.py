"""Order-flow and market-microstructure features."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from quant_os_core.quant._validation import as_float_array, paired_float_arrays


def order_book_imbalance(
    bid_sizes: npt.ArrayLike,
    ask_sizes: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    bids, asks = paired_float_arrays(
        bid_sizes,
        ask_sizes,
        left_name="bid_sizes",
        right_name="ask_sizes",
        min_size=1,
    )
    if np.any(bids < 0.0) or np.any(asks < 0.0):
        raise ValueError("book sizes cannot be negative")
    denominator = bids + asks
    result = np.zeros(bids.size, dtype=np.float64)
    valid = denominator > 0.0
    result[valid] = (bids[valid] - asks[valid]) / denominator[valid]
    return result


def trade_flow_imbalance(
    buy_volume: npt.ArrayLike,
    sell_volume: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    buys, sells = paired_float_arrays(
        buy_volume,
        sell_volume,
        left_name="buy_volume",
        right_name="sell_volume",
        min_size=1,
    )
    if np.any(buys < 0.0) or np.any(sells < 0.0):
        raise ValueError("trade volume cannot be negative")
    denominator = buys + sells
    result = np.zeros(buys.size, dtype=np.float64)
    valid = denominator > 0.0
    result[valid] = (buys[valid] - sells[valid]) / denominator[valid]
    return result


def microprice(
    bid_price: npt.ArrayLike,
    ask_price: npt.ArrayLike,
    bid_size: npt.ArrayLike,
    ask_size: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    bids, asks = paired_float_arrays(
        bid_price,
        ask_price,
        left_name="bid_price",
        right_name="ask_price",
        min_size=1,
    )
    bid_sizes, ask_sizes = paired_float_arrays(
        bid_size,
        ask_size,
        left_name="bid_size",
        right_name="ask_size",
        min_size=1,
    )
    if bid_sizes.size != bids.size:
        raise ValueError("prices and sizes must have equal length")
    if np.any(bids <= 0.0) or np.any(asks <= 0.0) or np.any(bids > asks):
        raise ValueError("quotes must be positive and non-crossed")
    if np.any(bid_sizes < 0.0) or np.any(ask_sizes < 0.0):
        raise ValueError("sizes cannot be negative")
    total = bid_sizes + ask_sizes
    midpoint = (bids + asks) / 2.0
    result = midpoint.copy()
    valid = total > 0.0
    result[valid] = (
        asks[valid] * bid_sizes[valid] + bids[valid] * ask_sizes[valid]
    ) / total[valid]
    return result


def vwap_deviation_signal(
    prices: npt.ArrayLike,
    vwap: npt.ArrayLike,
    *,
    threshold_fraction: float = 0.002,
    mean_reverting: bool = True,
) -> npt.NDArray[np.float64]:
    price_values, vwap_values = paired_float_arrays(
        prices,
        vwap,
        left_name="prices",
        right_name="vwap",
        min_size=1,
    )
    if np.any(price_values <= 0.0) or np.any(vwap_values <= 0.0):
        raise ValueError("prices and vwap must be > 0")
    threshold = float(threshold_fraction)
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError("threshold_fraction must be finite and non-negative")
    deviation = price_values / vwap_values - 1.0
    signal = np.zeros(price_values.size, dtype=np.float64)
    high = deviation > threshold
    low = deviation < -threshold
    signal[high] = -1.0 if mean_reverting else 1.0
    signal[low] = 1.0 if mean_reverting else -1.0
    return signal
