import importlib.util

import pytest

from quant_os_core.market_data.models import Candle
from quant_os_core.market_data.parquet import ParquetCandleWriter, ParquetUnavailableError


def test_parquet_adapter_has_clear_optional_dependency_error(
    tmp_path, minute_candles: list[Candle]
) -> None:
    if importlib.util.find_spec("pyarrow") is not None:
        pytest.skip("pyarrow is installed in this environment")
    with pytest.raises(ParquetUnavailableError):
        ParquetCandleWriter().write(tmp_path / "candles.parquet", minute_candles)
