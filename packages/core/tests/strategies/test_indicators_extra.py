import numpy as np
import pytest

from quant_os_core.strategies import indicators
from quant_os_core.strategies.indicators_extra import (
    adx,
    atr,
    cci,
    donchian,
    hull_ma,
    ichimoku,
    kama,
    keltner,
    obv,
    parabolic_sar,
    stochastic,
    supertrend,
    vwap,
    williams_r,
    wma,
    zscore,
)


def _trend(n: int = 60, slope: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    close = 100.0 + slope * np.arange(n, dtype=float)
    return close + 1.0, close - 1.0, close


def test_atr_is_the_core_indicator() -> None:
    h, lo, c = _trend(20)
    np.testing.assert_allclose(atr(h, lo, c, 5), indicators.atr(h, lo, c, 5), equal_nan=True)


def test_wma_hand_values() -> None:
    out = wma([1, 2, 3, 4], 3)
    assert np.isnan(out[:2]).all()
    np.testing.assert_allclose(out[2:], [14 / 6, 20 / 6])
    assert np.isnan(wma([1, 2], 3)).all()


def test_hull_ma_matches_formula_and_warmup() -> None:
    data = np.array([1.0, 3.0, 2.0, 5.0, 4.0, 6.0, 8.0, 7.0])
    out = hull_ma(data, 4)
    raw = 2 * wma(data, 2) - wma(data, 4)
    expected = np.full(data.size, np.nan)
    expected[3:] = wma(raw[3:], 2)
    np.testing.assert_allclose(out, expected, equal_nan=True)
    assert np.isnan(out[:4]).all()
    assert np.isfinite(out[4:]).all()
    with pytest.raises(ValueError, match="at least 2"):
        hull_ma(data, 1)
    assert np.isnan(hull_ma([1.0, 2.0, 3.0], 9)).all()


def test_kama_trending_series_uses_fast_constant() -> None:
    data = np.arange(1.0, 13.0)
    out = kama(data, 10, 2, 30)
    assert np.isnan(out[:10]).all()
    assert out[10] == pytest.approx(11.0)
    assert out[11] == pytest.approx(11.0 + (2 / 3) ** 2 * 1.0)
    flat = kama(np.full(15, 5.0), 10)
    assert flat[-1] == pytest.approx(5.0)
    assert np.isnan(kama([1.0, 2.0], 10)).all()


def test_zscore_hand_values() -> None:
    out = zscore([1, 2, 3], 3)
    assert out[2] == pytest.approx(1 / np.sqrt(2 / 3))
    sample = zscore([1, 2, 3], 3, ddof=1)
    assert sample[2] == pytest.approx(1.0)
    assert zscore([4, 4, 4], 3)[2] == 0.0
    assert np.isnan(zscore([1, 2], 3)).all()
    with pytest.raises(ValueError, match="ddof"):
        zscore([1, 2, 3], 1, ddof=1)


def test_stochastic_and_williams_hand_values() -> None:
    k, d = stochastic([3, 4, 5, 6], [1, 2, 3, 4], [2, 3, 4, 4], 3, 2)
    assert np.isnan(k[:2]).all()
    assert k[2] == pytest.approx(75.0)  # (4 - 1) / (5 - 1)
    assert k[3] == pytest.approx(50.0)  # (4 - 2) / (6 - 2)
    assert np.isnan(d[:3]).all()
    assert d[3] == pytest.approx(62.5)
    wr = williams_r([3, 4, 5], [1, 2, 3], [2, 3, 4], 3)
    assert wr[2] == pytest.approx(-25.0)
    flat_k, _ = stochastic([2, 2, 2], [2, 2, 2], [2, 2, 2], 2, 1)
    assert flat_k[1] == 50.0
    short_k, short_d = stochastic([2, 3], [1, 2], [1.5, 2.5], 5, 3)
    assert np.isnan(short_k).all() and np.isnan(short_d).all()


def test_cci_hand_value() -> None:
    tp = np.array([1.0, 2.0, 3.0, 4.0])
    out = cci(tp, tp, tp, 3)
    assert out[2] == pytest.approx(100.0)  # (3 - 2) / (0.015 * 2/3)
    assert cci([5, 5, 5], [5, 5, 5], [5, 5, 5], 3)[2] == 0.0
    assert np.isnan(cci([1, 2], [1, 2], [1, 2], 3)).all()
    with pytest.raises(ValueError, match="constant"):
        cci(tp, tp, tp, 3, constant=0.0)


def test_obv_with_and_without_volume() -> None:
    np.testing.assert_allclose(obv([1, 2, 2, 1], [10, 20, 30, 40]), [0, 20, 20, -20])
    assert obv([1, 2, 3], None) is None
    with pytest.raises(ValueError, match="equal length"):
        obv([1, 2], [1])
    with pytest.raises(ValueError, match="negative"):
        obv([1, 2], [1, -1])


def test_vwap_cumulative_and_rolling() -> None:
    h, lo, c, v = [3, 6, 9], [1, 2, 3], [2, 4, 6], [1, 1, 2]
    np.testing.assert_allclose(vwap(h, lo, c, v), indicators.cumulative_vwap(h, lo, c, v))
    rolling = vwap(h, lo, c, v, window=2)
    assert np.isnan(rolling[0])
    assert rolling[1] == pytest.approx((2 * 1 + 4 * 1) / 2)
    assert rolling[2] == pytest.approx((4 * 1 + 6 * 2) / 3)
    assert np.isnan(vwap(h, lo, c, [0, 0, 0], window=2)[1])
    assert np.isnan(vwap(h, lo, c, v, window=5)).all()
    with pytest.raises(ValueError, match="volume must match"):
        vwap(h, lo, c, [1, 1], window=2)
    with pytest.raises(ValueError, match="negative"):
        vwap(h, lo, c, [1, -1, 1], window=2)


def test_adx_hand_computed_uptrend() -> None:
    h = [10.0, 11.0, 12.0, 13.0]
    lo = [9.0, 10.0, 11.0, 12.0]
    c = [9.5, 10.5, 11.5, 12.5]
    a, plus, minus = adx(h, lo, c, 2)
    assert np.isnan(plus[:2]).all()
    np.testing.assert_allclose(plus[2:], [200 / 3, 200 / 3])
    np.testing.assert_allclose(minus[2:], [0.0, 0.0])
    assert np.isnan(a[:3]).all()
    assert a[3] == pytest.approx(100.0)
    short = adx(h[:3], lo[:3], c[:3], 2)
    assert all(np.isnan(x).all() for x in short)


def test_adx_downtrend_and_flat() -> None:
    h, lo, c = _trend(40, slope=-1.0)
    a, plus, minus = adx(h, lo, c, 14)
    assert minus[-1] > plus[-1]
    assert a[-1] > 50
    flat = np.full(10, 5.0)
    fa, fp, fm = adx(flat, flat, flat, 3)
    assert fa[-1] == 0.0 and fp[-1] == 0.0 and fm[-1] == 0.0


def test_keltner_is_ema_plus_atr_band() -> None:
    h, lo, c = _trend(30)
    lower, mid, upper = keltner(h, lo, c, 10, 5, 2.0)
    np.testing.assert_allclose(mid, indicators.ema(c, 10), equal_nan=True)
    band = 2.0 * indicators.atr(h, lo, c, 5)
    ready = np.isfinite(mid) & np.isfinite(band)
    assert ready.sum() == 21
    np.testing.assert_allclose((upper - mid)[ready], band[ready])
    np.testing.assert_allclose((mid - lower)[ready], band[ready])
    assert all(np.isnan(x).all() for x in keltner(h[:3], lo[:3], c[:3], 10, 5))
    with pytest.raises(ValueError, match="multiplier"):
        keltner(h, lo, c, multiplier=-1.0)


def test_donchian_includes_current_bar() -> None:
    upper, middle, lower = donchian([1, 3, 2, 5], [0, 1, 1, 2], 2)
    np.testing.assert_allclose(upper, [np.nan, 3, 3, 5], equal_nan=True)
    np.testing.assert_allclose(lower, [np.nan, 0, 1, 1], equal_nan=True)
    np.testing.assert_allclose(middle, [np.nan, 1.5, 2, 3], equal_nan=True)


def test_ichimoku_spans_are_shifted_forward() -> None:
    h = np.array([2, 3, 4, 5, 6, 7, 8, 9], dtype=float)
    lo = h - 2
    ich = ichimoku(h, lo, conversion=2, base=3, span_b=4, displacement=2)
    assert ich.conversion[1] == pytest.approx((3 + 0) / 2)
    assert ich.base[2] == pytest.approx((4 + 0) / 2)
    raw_a = (ich.conversion + ich.base) / 2
    np.testing.assert_allclose(ich.span_a[2:], raw_a[:-2], equal_nan=True)
    assert np.isnan(ich.span_a[:4]).all()
    assert ich.span_b[5] == pytest.approx((5 + 0) / 2)  # built from bars 0..3
    huge = ichimoku(h, lo, 2, 3, 4, displacement=20)
    assert np.isnan(huge.span_a).all()
    with pytest.raises(ValueError, match="displacement"):
        ichimoku(h, lo, displacement=-1)


def test_parabolic_sar_hand_computed() -> None:
    highs = [10, 11, 12, 11, 9, 9, 13]
    lows = [9, 10, 11, 10, 8, 7, 12]
    sar, trend = parabolic_sar(highs, lows)
    assert np.isnan(sar[0]) and np.isnan(trend[0])
    np.testing.assert_allclose(sar[1:], [9.0, 9.0, 9.18, 12.0, 11.92, 7.0])
    np.testing.assert_allclose(trend[1:], [1, 1, 1, -1, -1, 1])
    one_sar, _ = parabolic_sar([5.0], [4.0])
    assert np.isnan(one_sar).all()
    with pytest.raises(ValueError, match="step"):
        parabolic_sar(highs, lows, step=0.3, max_step=0.2)


def test_supertrend_follows_trend_and_flips() -> None:
    up_h, up_l, up_c = _trend(30)
    crash = 129.0 - 4.0 * np.arange(1, 16)
    h = np.concatenate([up_h, crash + 1.0])
    lo = np.concatenate([up_l, crash - 1.0])
    c = np.concatenate([up_c, crash])
    line, direction = supertrend(h, lo, c, 10, 3.0)
    assert np.isnan(line[:9]).all()
    assert direction[25] == 1.0 and line[25] < c[25]
    assert direction[-1] == -1.0 and line[-1] > c[-1]
    short_line, _ = supertrend(h[:3], lo[:3], c[:3], 10)
    assert np.isnan(short_line).all()
    with pytest.raises(ValueError, match="multiplier"):
        supertrend(h, lo, c, multiplier=0.0)


def test_supertrend_starts_down_and_flips_up() -> None:
    dn_h, dn_l, dn_c = _trend(20, slope=-1.0)
    rally = dn_c[-1] + 5.0 * np.arange(1, 15)
    h = np.concatenate([dn_h, rally + 1.0])
    lo = np.concatenate([dn_l, rally - 1.0])
    c = np.concatenate([dn_c - 0.9, rally])
    _, direction = supertrend(h, lo, c, 5, 2.0)
    assert direction[4] == -1.0
    assert direction[-1] == 1.0


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda: stochastic([1, 2], [2, 1], [1, 1]), "below low"),
        (lambda: stochastic([1, 2, 3], [0, 1], [1, 1, 1]), "equal length"),
        (lambda: wma([1, 2, 3], 0), "positive integer"),
        (lambda: zscore([1, 2, 3], 1.5), "positive integer"),  # type: ignore[arg-type]
        (lambda: wma([1.0, float("nan")], 1), "NaN"),
    ],
)
def test_input_validation(call: object, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        call()  # type: ignore[operator]
