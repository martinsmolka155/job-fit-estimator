"""Unit tests for src/eval_metrics.py — pure metric functions.

All tests use hand-built prediction/label pairs. No LLM, no API calls,
no file I/O. The live eval script (scripts/eval.py) is NOT invoked here.

Test coverage:
- ISCO: exact match, prefix-3 match, total mismatch
- Seniority: exact match, off-by-one, off-by-two (not counted as obo)
- Salary: within-15%, outside-15%, true inside predicted range, MAE/MAPE math
- Warning stats
- Aggregate metrics via aggregate_metrics()
- Helper formatters
- SalaryLabel.from_raw() scalar and dict
"""

from __future__ import annotations

import pytest

from src.eval_metrics import (
    SENIORITY_ORDER,
    EvalResults,
    Label,
    Prediction,
    SalaryLabel,
    SalaryPrediction,
    aggregate_metrics,
    compute_isco_metrics,
    compute_salary_metrics,
    compute_seniority_metrics,
    compute_warning_stats,
    format_czk,
    format_pct,
    mape_is_finite,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pred(
    cv_name: str = "test.pdf",
    isco: str | None = "2512",
    seniority: str | None = "senior",
    sal_low: int = 100_000,
    sal_high: int = 150_000,
    warnings: int = 0,
) -> Prediction:
    salary = SalaryPrediction(low=sal_low, high=sal_high)
    return Prediction(
        cv_name=cv_name,
        isco_code=isco,
        seniority=seniority,  # type: ignore[arg-type]
        salary=salary,
        warnings_emitted=warnings,
    )


def _label(
    cv_name: str = "test.pdf",
    isco: str = "2512",
    seniority: str = "senior",
    sal_low: int = 100_000,
    sal_high: int = 150_000,
    is_real: bool = False,
) -> Label:
    salary = SalaryLabel(low=sal_low, high=sal_high)
    return Label(
        cv_name=cv_name,
        isco_code=isco,
        seniority=seniority,
        salary=salary,
        is_real=is_real,
    )


# ---------------------------------------------------------------------------
# SalaryLabel.from_raw
# ---------------------------------------------------------------------------


class TestSalaryLabelFromRaw:
    def test_scalar_int(self) -> None:
        label = SalaryLabel.from_raw(80_000)
        assert label.low == 80_000
        assert label.high == 80_000

    def test_dict(self) -> None:
        label = SalaryLabel.from_raw({"low": 60_000, "high": 90_000})
        assert label.low == 60_000
        assert label.high == 90_000

    def test_dict_string_values(self) -> None:
        # JSON may supply numeric strings in rare cases
        label = SalaryLabel.from_raw({"low": 50_000, "high": 80_000})
        assert isinstance(label.low, int)
        assert isinstance(label.high, int)


# ---------------------------------------------------------------------------
# ISCO metrics
# ---------------------------------------------------------------------------


class TestISCOMetrics:
    def test_exact_match(self) -> None:
        p = [_pred("a.pdf", isco="2512")]
        lb = [_label("a.pdf", isco="2512")]
        exact, p3, results = compute_isco_metrics(p, lb)
        assert exact == 1.0
        assert p3 == 1.0
        assert len(results) == 1
        assert results[0].exact_match is True

    def test_prefix3_match_not_exact(self) -> None:
        # 2511 vs 2512 — same 3-digit prefix (251), different 4th digit
        p = [_pred("a.pdf", isco="2511")]
        lb = [_label("a.pdf", isco="2512")]
        exact, p3, results = compute_isco_metrics(p, lb)
        assert exact == 0.0
        assert p3 == 1.0
        assert results[0].exact_match is False
        assert results[0].prefix3_match is True

    def test_total_mismatch(self) -> None:
        # 5120 (cooks) vs 2512 (IT dev) — no prefix overlap
        p = [_pred("a.pdf", isco="5120")]
        lb = [_label("a.pdf", isco="2512")]
        exact, p3, results = compute_isco_metrics(p, lb)
        assert exact == 0.0
        assert p3 == 0.0
        assert results[0].exact_match is False
        assert results[0].prefix3_match is False

    def test_two_predictions_one_match(self) -> None:
        preds = [_pred("a.pdf", isco="2512"), _pred("b.pdf", isco="5120")]
        labels = [_label("a.pdf", isco="2512"), _label("b.pdf", isco="2512")]
        exact, p3, results = compute_isco_metrics(preds, labels)
        assert exact == pytest.approx(0.5)

    def test_todo_labels_excluded(self) -> None:
        p = [_pred("a.pdf", isco="2512")]
        lb = [_label("a.pdf", isco="TODO", is_real=True)]
        exact, p3, results = compute_isco_metrics(p, lb)
        assert exact == 0.0
        assert results == []

    def test_no_labels(self) -> None:
        exact, p3, results = compute_isco_metrics([], [])
        assert exact == 0.0
        assert p3 == 0.0
        assert results == []

    def test_predicted_none(self) -> None:
        p = [_pred("a.pdf", isco=None)]
        lb = [_label("a.pdf", isco="2512")]
        exact, p3, results = compute_isco_metrics(p, lb)
        assert exact == 0.0
        assert p3 == 0.0

    def test_three_cvs_mixed(self) -> None:
        preds = [
            _pred("a.pdf", isco="2512"),  # exact
            _pred("b.pdf", isco="2511"),  # prefix-3 only
            _pred("c.pdf", isco="5120"),  # total miss
        ]
        labels = [
            _label("a.pdf", isco="2512"),
            _label("b.pdf", isco="2512"),
            _label("c.pdf", isco="2512"),
        ]
        exact, p3, results = compute_isco_metrics(preds, labels)
        assert exact == pytest.approx(1 / 3)
        assert p3 == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# Seniority metrics
# ---------------------------------------------------------------------------


class TestSeniorityMetrics:
    def test_exact_match(self) -> None:
        p = [_pred("a.pdf", seniority="senior")]
        lb = [_label("a.pdf", seniority="senior")]
        exact, obo, details = compute_seniority_metrics(p, lb)
        assert exact == 1.0
        assert obo == 1.0

    def test_off_by_one(self) -> None:
        # medior predicted vs senior true — adjacent bands
        p = [_pred("a.pdf", seniority="medior")]
        lb = [_label("a.pdf", seniority="senior")]
        exact, obo, details = compute_seniority_metrics(p, lb)
        assert exact == 0.0
        assert obo == 1.0
        assert details[0].off_by_one is True

    def test_off_by_two_not_counted(self) -> None:
        # junior (rank 0) vs senior (rank 2) — delta = 2
        p = [_pred("a.pdf", seniority="junior")]
        lb = [_label("a.pdf", seniority="senior")]
        exact, obo, details = compute_seniority_metrics(p, lb)
        assert exact == 0.0
        assert obo == 0.0
        assert details[0].off_by_one is False

    def test_lead_vs_senior_is_obo(self) -> None:
        # lead (3) vs senior (2) — delta = 1
        p = [_pred("a.pdf", seniority="lead")]
        lb = [_label("a.pdf", seniority="senior")]
        exact, obo, _ = compute_seniority_metrics(p, lb)
        assert obo == 1.0

    def test_todo_excluded(self) -> None:
        p = [_pred("a.pdf", seniority="senior")]
        lb = [_label("a.pdf", seniority="TODO", is_real=True)]
        exact, obo, details = compute_seniority_metrics(p, lb)
        assert exact == 0.0
        assert details == []

    def test_predicted_none(self) -> None:
        p = [_pred("a.pdf", seniority=None)]
        lb = [_label("a.pdf", seniority="senior")]
        exact, obo, details = compute_seniority_metrics(p, lb)
        assert exact == 0.0
        assert obo == 0.0

    def test_seniority_order_covers_all_bands(self) -> None:
        expected = {"junior": 0, "medior": 1, "senior": 2, "lead": 3, "principal": 4}
        assert expected == SENIORITY_ORDER


# ---------------------------------------------------------------------------
# Salary metrics
# ---------------------------------------------------------------------------


class TestSalaryMetrics:
    def test_within_15pct(self) -> None:
        # pred mid = 100k, true mid = 100k — 0% error
        p = [_pred("a.pdf", sal_low=90_000, sal_high=110_000)]  # mid=100k
        lb = [_label("a.pdf", sal_low=90_000, sal_high=110_000)]  # true mid=100k
        mae, mape, w15, cov, results = compute_salary_metrics(p, lb)
        assert w15 == 1.0
        assert mape == pytest.approx(0.0)

    def test_outside_15pct(self) -> None:
        # pred mid = 120k, true mid = 90k — 33% error
        p = [_pred("a.pdf", sal_low=110_000, sal_high=130_000)]  # mid=120k
        lb = [_label("a.pdf", sal_low=80_000, sal_high=100_000)]  # true mid=90k
        mae, mape, w15, cov, results = compute_salary_metrics(p, lb)
        assert w15 == 0.0
        assert mape == pytest.approx(abs(120_000 - 90_000) / 90_000, rel=1e-3)

    def test_true_inside_predicted_range(self) -> None:
        # pred: 80k-130k, true: 90k-110k — true range fully inside pred range
        p = [_pred("a.pdf", sal_low=80_000, sal_high=130_000)]
        lb = [_label("a.pdf", sal_low=90_000, sal_high=110_000)]
        mae, mape, w15, cov, results = compute_salary_metrics(p, lb)
        assert cov == 1.0
        assert results[0].true_inside_range is True

    def test_true_outside_predicted_range(self) -> None:
        # pred: 40k-50k, true: 90k-110k — no overlap
        p = [_pred("a.pdf", sal_low=40_000, sal_high=50_000)]
        lb = [_label("a.pdf", sal_low=90_000, sal_high=110_000)]
        mae, mape, w15, cov, results = compute_salary_metrics(p, lb)
        assert cov == 0.0
        assert results[0].true_inside_range is False

    def test_mae_calculation(self) -> None:
        # pred mid = 120k, true mid = 100k => MAE = 20k
        p = [_pred("a.pdf", sal_low=110_000, sal_high=130_000)]  # mid=120k
        lb = [_label("a.pdf", sal_low=90_000, sal_high=110_000)]  # mid=100k
        mae, mape, w15, cov, results = compute_salary_metrics(p, lb)
        assert mae == pytest.approx(20_000.0)

    def test_mape_math(self) -> None:
        # pred mid = 80k, true mid = 100k => MAPE = 20/100 = 0.20
        p = [_pred("a.pdf", sal_low=70_000, sal_high=90_000)]  # mid=80k
        lb = [_label("a.pdf", sal_low=90_000, sal_high=110_000)]  # mid=100k
        mae, mape, w15, cov, results = compute_salary_metrics(p, lb)
        assert mape == pytest.approx(0.20, rel=1e-3)

    def test_partial_overlap_counts_as_coverage(self) -> None:
        # pred: 80k-100k, true: 90k-120k — partial overlap at 90-100k
        p = [_pred("a.pdf", sal_low=80_000, sal_high=100_000)]
        lb = [_label("a.pdf", sal_low=90_000, sal_high=120_000)]
        mae, mape, w15, cov, results = compute_salary_metrics(p, lb)
        assert cov == 1.0  # they overlap

    def test_skip_todo_salary(self) -> None:
        # Labels with low=0 high=0 are TODO placeholders — skip
        p = [_pred("a.pdf", sal_low=80_000, sal_high=100_000)]
        lb = [_label("a.pdf", sal_low=0, sal_high=0)]
        mae, mape, w15, cov, results = compute_salary_metrics(p, lb)
        assert results == []

    def test_skip_is_real_label(self) -> None:
        p = [_pred("a.pdf")]
        lb = [_label("a.pdf", is_real=True)]
        mae, mape, w15, cov, results = compute_salary_metrics(p, lb)
        assert results == []

    def test_no_salary_prediction(self) -> None:
        p = [Prediction(cv_name="a.pdf", isco_code="2512", seniority="senior", salary=None)]
        lb = [_label("a.pdf", sal_low=90_000, sal_high=110_000)]
        mae, mape, w15, cov, results = compute_salary_metrics(p, lb)
        assert results == []

    def test_two_cvs_average(self) -> None:
        # CV1: pred mid=100k, true mid=100k (0% error, within 15%, in range)
        # CV2: pred mid=120k, true mid=90k (33% error, outside 15%, no overlap pred=110-130 true=80-100)
        preds = [
            _pred("a.pdf", sal_low=90_000, sal_high=110_000),  # mid=100k
            _pred("b.pdf", sal_low=110_000, sal_high=130_000),  # mid=120k
        ]
        labels = [
            _label("a.pdf", sal_low=90_000, sal_high=110_000),  # mid=100k
            _label("b.pdf", sal_low=80_000, sal_high=100_000),  # mid=90k
        ]
        mae, mape, w15, cov, results = compute_salary_metrics(preds, labels)
        # MAE = (0 + 30k) / 2 = 15k
        assert mae == pytest.approx(15_000.0, rel=1e-3)
        # MAPE = (0.0 + 0.333) / 2 ≈ 0.1667
        assert mape == pytest.approx(30_000 / 90_000 / 2, rel=1e-2)
        # Within 15%: only CV1 → 0.5
        assert w15 == pytest.approx(0.5)
        # Coverage: CV1 yes, CV2 no (110-130 vs 80-100: no overlap) → 0.5
        assert cov == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Warning stats
# ---------------------------------------------------------------------------


class TestWarningStats:
    def test_no_predictions(self) -> None:
        total, per_cv = compute_warning_stats([])
        assert total == 0
        assert per_cv == 0.0

    def test_single_prediction(self) -> None:
        p = [_pred("a.pdf", warnings=3)]
        total, per_cv = compute_warning_stats(p)
        assert total == 3
        assert per_cv == 3.0

    def test_multiple_predictions(self) -> None:
        preds = [_pred("a.pdf", warnings=2), _pred("b.pdf", warnings=4)]
        total, per_cv = compute_warning_stats(preds)
        assert total == 6
        assert per_cv == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# aggregate_metrics
# ---------------------------------------------------------------------------


class TestAggregateMetrics:
    def test_all_correct(self) -> None:
        preds = [
            _pred("a.pdf", isco="2512", seniority="senior", sal_low=90_000, sal_high=110_000),
        ]
        labels = [
            _label("a.pdf", isco="2512", seniority="senior", sal_low=90_000, sal_high=110_000),
        ]
        result = aggregate_metrics(preds, labels)
        assert isinstance(result, EvalResults)
        assert result.n_labeled == 1
        assert result.isco_exact_rate == pytest.approx(1.0)
        assert result.seniority_exact_rate == pytest.approx(1.0)
        assert result.salary_within_15pct_rate == pytest.approx(1.0)
        assert result.isco_mismatches == []

    def test_all_wrong(self) -> None:
        preds = [
            _pred("a.pdf", isco="5120", seniority="junior", sal_low=20_000, sal_high=30_000),
        ]
        labels = [
            _label("a.pdf", isco="2512", seniority="senior", sal_low=90_000, sal_high=110_000),
        ]
        result = aggregate_metrics(preds, labels)
        assert result.isco_exact_rate == pytest.approx(0.0)
        assert result.seniority_exact_rate == pytest.approx(0.0)
        assert len(result.isco_mismatches) == 1
        assert result.isco_mismatches[0].predicted == "5120"
        assert result.isco_mismatches[0].true == "2512"

    def test_todo_not_counted_in_labeled(self) -> None:
        preds = [_pred("real.pdf")]
        labels = [_label("real.pdf", is_real=True)]
        result = aggregate_metrics(preds, labels)
        assert result.n_labeled == 0
        assert result.n_total == 1

    def test_empty(self) -> None:
        result = aggregate_metrics([], [])
        assert result.n_total == 0
        assert result.isco_exact_rate == 0.0
        assert result.salary_mae == 0.0


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------


class TestFormatters:
    def test_format_pct_zero(self) -> None:
        assert format_pct(0.0) == "0.0%"

    def test_format_pct_one(self) -> None:
        assert format_pct(1.0) == "100.0%"

    def test_format_pct_half(self) -> None:
        assert format_pct(0.5) == "50.0%"

    def test_format_czk(self) -> None:
        assert format_czk(45_000) == "45,000 CZK"

    def test_format_czk_large(self) -> None:
        result = format_czk(150_000)
        assert "150,000" in result

    def test_mape_is_finite_normal(self) -> None:
        assert mape_is_finite(0.15) is True

    def test_mape_is_finite_nan(self) -> None:
        assert mape_is_finite(float("nan")) is False

    def test_mape_is_finite_inf(self) -> None:
        assert mape_is_finite(float("inf")) is False
