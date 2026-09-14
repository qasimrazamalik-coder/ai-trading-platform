"""Right-fit validation and anti-overfitting diagnostics."""

from quant_os_core.right_fit.leakage import (
    assert_monotonic_unique_timestamps,
    interval_overlap_mask,
    purge_label_overlap,
    train_test_overlap,
)
from quant_os_core.right_fit.multiple_testing import (
    DeflatedSharpeResult,
    deflated_sharpe_ratio,
    expected_maximum_sharpe,
    probabilistic_sharpe_ratio,
    sidak_adjusted_pvalue,
)
from quant_os_core.right_fit.pbo import PBOResult, probability_backtest_overfitting
from quant_os_core.right_fit.report import (
    RightFitDecision,
    RightFitPolicy,
    RightFitReport,
    evaluate_right_fit,
)
from quant_os_core.right_fit.selection import (
    FitDiagnosis,
    WalkForwardSelectionResult,
    diagnose_fit,
    walk_forward_parameter_selection,
)
from quant_os_core.right_fit.robustness import (
    MonteCarloRobustnessResult,
    monte_carlo_robustness,
    moving_block_bootstrap,
)
from quant_os_core.right_fit.splits import (
    TimeSeriesSplit,
    combinatorial_purged_splits,
    purged_kfold,
    walk_forward_splits,
)
from quant_os_core.right_fit.stability import (
    ParameterStabilityResult,
    complexity_penalized_score,
    parameter_surface_stability,
    relative_performance_degradation,
)

__all__ = [
    "DeflatedSharpeResult",
    "FitDiagnosis",
    "MonteCarloRobustnessResult",
    "PBOResult",
    "ParameterStabilityResult",
    "RightFitDecision",
    "RightFitPolicy",
    "RightFitReport",
    "TimeSeriesSplit",
    "WalkForwardSelectionResult",
    "assert_monotonic_unique_timestamps",
    "combinatorial_purged_splits",
    "complexity_penalized_score",
    "deflated_sharpe_ratio",
    "diagnose_fit",
    "evaluate_right_fit",
    "expected_maximum_sharpe",
    "interval_overlap_mask",
    "monte_carlo_robustness",
    "moving_block_bootstrap",
    "parameter_surface_stability",
    "probabilistic_sharpe_ratio",
    "probability_backtest_overfitting",
    "purge_label_overlap",
    "purged_kfold",
    "relative_performance_degradation",
    "sidak_adjusted_pvalue",
    "train_test_overlap",
    "walk_forward_parameter_selection",
    "walk_forward_splits",
]
