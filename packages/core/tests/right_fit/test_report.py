from quant_os_core.right_fit import RightFitDecision, RightFitPolicy, evaluate_right_fit


def test_right_fit_passes_strong_candidate() -> None:
    result = evaluate_right_fit(
        train_score=1.3,
        test_score=1.1,
        pbo=0.20,
        deflated_sharpe_probability=0.98,
        parameter_stability=0.80,
        bootstrap_positive_probability=0.90,
        leakage_detected=False,
    )
    assert result.decision is RightFitDecision.RESEARCH_PASS


def test_right_fit_rejects_leakage() -> None:
    result = evaluate_right_fit(
        train_score=1.3,
        test_score=1.1,
        pbo=0.20,
        deflated_sharpe_probability=0.98,
        parameter_stability=0.80,
        bootstrap_positive_probability=0.90,
        leakage_detected=True,
    )
    assert result.decision is RightFitDecision.REJECT
    assert any("leakage" in reason for reason in result.reasons)


def test_right_fit_rejects_high_pbo() -> None:
    result = evaluate_right_fit(
        train_score=1.3,
        test_score=1.1,
        pbo=0.75,
        deflated_sharpe_probability=0.98,
        parameter_stability=0.80,
        bootstrap_positive_probability=0.90,
        leakage_detected=False,
    )
    assert result.decision is RightFitDecision.REJECT


def test_right_fit_marks_unstable_surface_for_review() -> None:
    result = evaluate_right_fit(
        train_score=1.3,
        test_score=1.1,
        pbo=0.20,
        deflated_sharpe_probability=0.98,
        parameter_stability=0.30,
        bootstrap_positive_probability=0.90,
        leakage_detected=False,
    )
    assert result.decision is RightFitDecision.REVIEW


def test_policy_is_configurable() -> None:
    policy = RightFitPolicy(max_pbo=0.8, min_deflated_sharpe_probability=0.8)
    result = evaluate_right_fit(
        train_score=1.0,
        test_score=0.9,
        pbo=0.7,
        deflated_sharpe_probability=0.9,
        parameter_stability=0.8,
        bootstrap_positive_probability=0.9,
        leakage_detected=False,
        policy=policy,
    )
    assert result.decision is RightFitDecision.RESEARCH_PASS
