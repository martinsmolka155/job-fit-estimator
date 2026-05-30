"""Pure metric functions for the Job Fit Estimator evaluation harness.

No LLM, no I/O, no side effects. Given predictions + ground-truth labels,
compute ISCO accuracy, seniority agreement, salary error, and warning stats.

All functions are deterministic and unit-testable in isolation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

# ---------------------------------------------------------------------------
# Input types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SalaryPrediction:
    """Salary prediction from the pipeline."""

    low: int
    high: int
    mid: int | None = None


@dataclass(frozen=True)
class SalaryLabel:
    """Ground-truth salary value or range."""

    low: int
    high: int

    @classmethod
    def from_raw(cls, raw: int | dict[str, int]) -> SalaryLabel:
        """Parse label from JSON value: int scalar or {low, high} dict.

        Args:
            raw: Either a single int (treated as low == high) or a dict with
                 ``low`` and ``high`` keys.

        Returns:
            SalaryLabel instance.
        """
        if isinstance(raw, int):
            return cls(low=raw, high=raw)
        return cls(low=int(raw["low"]), high=int(raw["high"]))


SENIORITY_ORDER: dict[str, int] = {
    "junior": 0,
    "medior": 1,
    "senior": 2,
    "lead": 3,
    "principal": 4,
}


@dataclass
class Prediction:
    """Single pipeline prediction for one CV."""

    cv_name: str
    isco_code: str | None
    seniority: Literal["junior", "medior", "senior", "lead", "principal"] | None
    salary: SalaryPrediction | None
    warnings_emitted: int = 0


@dataclass
class Label:
    """Single ground-truth label for one CV."""

    cv_name: str
    isco_code: str
    seniority: str
    salary: SalaryLabel | None
    is_real: bool = False  # True when salary_source contains "TODO_real" or isco_code is "TODO"


# ---------------------------------------------------------------------------
# Intermediate results
# ---------------------------------------------------------------------------


@dataclass
class ISCOResult:
    """Per-pair ISCO accuracy result."""

    cv_name: str
    predicted: str | None
    true: str
    exact_match: bool
    prefix3_match: bool  # 3-digit prefix matches


@dataclass
class SeniorityResult:
    """Per-pair seniority accuracy result."""

    cv_name: str
    predicted: str | None
    true: str
    exact_match: bool
    off_by_one: bool  # adjacent bands (e.g. medior predicted vs senior true)


@dataclass
class SalaryResult:
    """Per-pair salary error result."""

    cv_name: str
    predicted_low: int
    predicted_high: int
    true_low: int
    true_high: int
    mid_error_abs: float  # |predicted_mid - true_mid|
    mape: float  # Mean Absolute Percentage Error vs true mid
    within_15pct: bool  # predicted mid within ±15% of true mid
    true_inside_range: bool  # true label's low-high overlaps predicted low-high


# ---------------------------------------------------------------------------
# Result summary
# ---------------------------------------------------------------------------


@dataclass
class EvalResults:
    """Aggregated evaluation metrics across all labeled CVs."""

    # ---- counts ----
    n_total: int = 0
    n_labeled: int = 0  # excludes TODO entries
    n_with_salary: int = 0

    # ---- ISCO ----
    isco_exact_rate: float = 0.0
    isco_prefix3_rate: float = 0.0
    isco_mismatches: list[ISCOResult] = field(default_factory=list)

    # ---- seniority ----
    seniority_exact_rate: float = 0.0
    seniority_off_by_one_rate: float = 0.0  # includes exact matches
    seniority_details: list[SeniorityResult] = field(default_factory=list)

    # ---- salary ----
    salary_mae: float = 0.0
    salary_mape: float = 0.0
    salary_within_15pct_rate: float = 0.0
    salary_range_coverage_rate: float = 0.0  # true value inside predicted range
    salary_details: list[SalaryResult] = field(default_factory=list)

    # ---- warnings ----
    warnings_total: int = 0
    warnings_per_cv: float = 0.0


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------


def compute_isco_metrics(
    predictions: list[Prediction],
    labels: list[Label],
) -> tuple[float, float, list[ISCOResult]]:
    """Compute ISCO exact-match rate, 3-digit prefix match rate, and mismatch list.

    Only pairs where label is not a TODO placeholder are included.

    Args:
        predictions: Pipeline predictions, one per CV.
        labels: Ground-truth labels, one per CV.

    Returns:
        Tuple of (exact_rate, prefix3_rate, list_of_results).
        Rates are floats in [0.0, 1.0]; 0.0 when no labeled pairs available.
    """
    pred_by_name = {p.cv_name: p for p in predictions}
    results: list[ISCOResult] = []

    for label in labels:
        if label.is_real or label.isco_code == "TODO":
            continue
        pred = pred_by_name.get(label.cv_name)
        predicted_code = pred.isco_code if pred else None
        true_code = label.isco_code

        exact = predicted_code == true_code
        prefix3 = (
            predicted_code is not None
            and len(predicted_code) >= 3
            and len(true_code) >= 3
            and predicted_code[:3] == true_code[:3]
        )
        results.append(
            ISCOResult(
                cv_name=label.cv_name,
                predicted=predicted_code,
                true=true_code,
                exact_match=exact,
                prefix3_match=prefix3 or exact,
            )
        )

    if not results:
        return 0.0, 0.0, []

    exact_rate = sum(1 for r in results if r.exact_match) / len(results)
    prefix3_rate = sum(1 for r in results if r.prefix3_match) / len(results)
    return exact_rate, prefix3_rate, results


def compute_seniority_metrics(
    predictions: list[Prediction],
    labels: list[Label],
) -> tuple[float, float, list[SeniorityResult]]:
    """Compute seniority exact-match rate and off-by-one rate.

    Off-by-one means |predicted_rank - true_rank| <= 1.

    Args:
        predictions: Pipeline predictions, one per CV.
        labels: Ground-truth labels, one per CV.

    Returns:
        Tuple of (exact_rate, off_by_one_rate, details).
        Off-by-one rate includes exact matches.
    """
    pred_by_name = {p.cv_name: p for p in predictions}
    results: list[SeniorityResult] = []

    for label in labels:
        if label.is_real or label.seniority == "TODO":
            continue
        pred = pred_by_name.get(label.cv_name)
        predicted = pred.seniority if pred else None
        true = label.seniority

        exact = predicted == true
        if predicted is None:
            off_by_one = False
        else:
            pred_rank = SENIORITY_ORDER.get(predicted, -99)
            true_rank = SENIORITY_ORDER.get(true, -99)
            off_by_one = abs(pred_rank - true_rank) <= 1

        results.append(
            SeniorityResult(
                cv_name=label.cv_name,
                predicted=predicted,
                true=true,
                exact_match=exact,
                off_by_one=off_by_one,
            )
        )

    if not results:
        return 0.0, 0.0, []

    exact_rate = sum(1 for r in results if r.exact_match) / len(results)
    obo_rate = sum(1 for r in results if r.off_by_one) / len(results)
    return exact_rate, obo_rate, results


def _safe_mape(predicted_mid: float, true_mid: float) -> float:
    """Compute MAPE for a single pair, returning 0.0 when true_mid is 0.

    Args:
        predicted_mid: Predicted salary midpoint.
        true_mid: True salary midpoint.

    Returns:
        Absolute percentage error (0.0 to 1.0+); 0.0 when true_mid == 0.
    """
    if true_mid == 0:
        return 0.0
    return abs(predicted_mid - true_mid) / true_mid


def compute_salary_metrics(
    predictions: list[Prediction],
    labels: list[Label],
) -> tuple[float, float, float, float, list[SalaryResult]]:
    """Compute salary MAE, MAPE, within-15% rate, and range-coverage rate.

    Uses midpoints: predicted_mid = (low + high) / 2, true_mid = (low + high) / 2.
    Range coverage = true label's [low, high] overlaps with predicted [low, high].

    Args:
        predictions: Pipeline predictions.
        labels: Ground-truth labels.

    Returns:
        Tuple of (mae, mape, within_15pct_rate, range_coverage_rate, details).
        All rates in [0.0, 1.0]; 0.0 when no salary-labeled pairs exist.
    """
    pred_by_name = {p.cv_name: p for p in predictions}
    results: list[SalaryResult] = []

    for label in labels:
        if label.is_real or label.salary is None:
            continue
        pred = pred_by_name.get(label.cv_name)
        if pred is None or pred.salary is None:
            continue
        # Skip placeholder salary labels (both low and high == 0 means TODO)
        if label.salary.low == 0 and label.salary.high == 0:
            continue

        pred_mid = (pred.salary.low + pred.salary.high) / 2
        true_mid = (label.salary.low + label.salary.high) / 2

        mae = abs(pred_mid - true_mid)
        mape = _safe_mape(pred_mid, true_mid)
        within_15 = mape <= 0.15
        # Range coverage: check if the intervals overlap
        # i.e. NOT (pred.high < true.low OR pred.low > true.high)
        range_overlap = not (
            pred.salary.high < label.salary.low or pred.salary.low > label.salary.high
        )

        results.append(
            SalaryResult(
                cv_name=label.cv_name,
                predicted_low=pred.salary.low,
                predicted_high=pred.salary.high,
                true_low=label.salary.low,
                true_high=label.salary.high,
                mid_error_abs=mae,
                mape=mape,
                within_15pct=within_15,
                true_inside_range=range_overlap,
            )
        )

    if not results:
        return 0.0, 0.0, 0.0, 0.0, []

    n = len(results)
    mae_mean = sum(r.mid_error_abs for r in results) / n
    mape_mean = sum(r.mape for r in results) / n
    within_15_rate = sum(1 for r in results if r.within_15pct) / n
    coverage_rate = sum(1 for r in results if r.true_inside_range) / n
    return mae_mean, mape_mean, within_15_rate, coverage_rate, results


def compute_warning_stats(predictions: list[Prediction]) -> tuple[int, float]:
    """Compute total warnings emitted and average per CV.

    Args:
        predictions: All pipeline predictions (including TODO entries).

    Returns:
        Tuple of (total_warnings, warnings_per_cv). Per-CV is 0.0 if no predictions.
    """
    if not predictions:
        return 0, 0.0
    total = sum(p.warnings_emitted for p in predictions)
    per_cv = total / len(predictions)
    return total, per_cv


def aggregate_metrics(
    predictions: list[Prediction],
    labels: list[Label],
) -> EvalResults:
    """Compute all metrics and return a single EvalResults summary.

    Args:
        predictions: All pipeline outputs (one per CV run).
        labels: All ground-truth labels (one per CV in eval_labels.json).

    Returns:
        EvalResults with all metrics populated.
    """
    n_labeled = sum(
        1 for lb in labels if not lb.is_real and lb.isco_code != "TODO" and lb.seniority != "TODO"
    )
    n_with_salary = sum(
        1
        for lb in labels
        if not lb.is_real
        and lb.salary is not None
        and not (lb.salary.low == 0 and lb.salary.high == 0)
    )

    isco_exact, isco_prefix3, isco_details = compute_isco_metrics(predictions, labels)
    sen_exact, sen_obo, sen_details = compute_seniority_metrics(predictions, labels)
    s_mae, s_mape, s_within15, s_coverage, s_details = compute_salary_metrics(predictions, labels)
    w_total, w_per_cv = compute_warning_stats(predictions)

    # Collect mismatches only (exact matches are not included in the mismatch list)
    isco_mismatches = [r for r in isco_details if not r.exact_match]

    return EvalResults(
        n_total=len(labels),
        n_labeled=n_labeled,
        n_with_salary=n_with_salary,
        isco_exact_rate=isco_exact,
        isco_prefix3_rate=isco_prefix3,
        isco_mismatches=isco_mismatches,
        seniority_exact_rate=sen_exact,
        seniority_off_by_one_rate=sen_obo,
        seniority_details=sen_details,
        salary_mae=s_mae,
        salary_mape=s_mape,
        salary_within_15pct_rate=s_within15,
        salary_range_coverage_rate=s_coverage,
        salary_details=s_details,
        warnings_total=w_total,
        warnings_per_cv=w_per_cv,
    )


def format_pct(value: float) -> str:
    """Format a 0-1 float as a percentage string, e.g. 0.75 -> '75.0%'.

    Args:
        value: Float in [0.0, 1.0].

    Returns:
        String with one decimal place and percent sign.
    """
    return f"{value * 100:.1f}%"


def format_czk(value: float) -> str:
    """Format a CZK amount with thousands separator, e.g. 45000 -> '45,000 CZK'.

    Args:
        value: CZK amount.

    Returns:
        Formatted string.
    """
    return f"{int(value):,} CZK"


def mape_is_finite(mape: float) -> bool:
    """Return True when MAPE is a finite, non-NaN value.

    Args:
        mape: MAPE value to check.

    Returns:
        True if the value is usable in a report.
    """
    return math.isfinite(mape)
