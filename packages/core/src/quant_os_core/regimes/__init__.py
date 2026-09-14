"""Market-regime intelligence for strategy selection and risk context."""

from quant_os_core.regimes.base import (
    CorrelationRegime,
    LiquidityRegime,
    MacroRegime,
    RegimeEstimate,
    RegimeSnapshot,
    RiskRegime,
    TrendRegime,
    VolatilityRegime,
)
from quant_os_core.regimes.change_points import ChangePoint, cusum_change_points
from quant_os_core.regimes.classification import (
    average_absolute_correlation,
    classify_correlation,
    classify_liquidity,
    classify_macro,
    classify_risk,
    classify_trend,
    classify_volatility,
    volatility_clustering_score,
)
from quant_os_core.regimes.compatibility import (
    StrategyRegimeMatrix,
    StrategyRegimeScore,
    composite_strategy_weights,
    dynamic_strategy_weights,
)
from quant_os_core.regimes.engine import regime_snapshot
from quant_os_core.regimes.gmm import DiagonalGaussianMixture, GaussianMixtureFit
from quant_os_core.regimes.hmm import GaussianHMMFit, GaussianHMMRegimeDetector
from quant_os_core.regimes.transitions import (
    confidence_from_probabilities,
    normalized_entropy,
    stationary_distribution,
    transition_matrix,
)

__all__ = [
    "ChangePoint",
    "CorrelationRegime",
    "DiagonalGaussianMixture",
    "GaussianHMMFit",
    "GaussianHMMRegimeDetector",
    "GaussianMixtureFit",
    "LiquidityRegime",
    "MacroRegime",
    "RegimeEstimate",
    "RegimeSnapshot",
    "RiskRegime",
    "StrategyRegimeMatrix",
    "StrategyRegimeScore",
    "TrendRegime",
    "VolatilityRegime",
    "average_absolute_correlation",
    "classify_correlation",
    "classify_liquidity",
    "classify_macro",
    "classify_risk",
    "classify_trend",
    "classify_volatility",
    "composite_strategy_weights",
    "confidence_from_probabilities",
    "cusum_change_points",
    "dynamic_strategy_weights",
    "normalized_entropy",
    "regime_snapshot",
    "stationary_distribution",
    "transition_matrix",
    "volatility_clustering_score",
]
