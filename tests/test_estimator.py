"""Tests for salary estimator (ISPV-backed implementation).

All tests mock ISPVLoader — no real XLSX files required.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from src.estimator import SalaryEstimator, _score_to_seniority
from src.salary_ispv import ISPVLookupError
from src.schemas import (
    Experience,
    Resume,
    SalaryBand,
    SalaryData,
    ScoreComponent,
    SeniorityScore,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_score(total: float) -> SeniorityScore:
    return SeniorityScore(
        total=total,
        components=[ScoreComponent(name="Test", score=total, weight=1.0, reasoning="test")],
        confidence=1.0,
    )


def _make_salary_data(isco_code: str = "2512") -> SalaryData:
    """Build a minimal valid SalaryData fixture backed by synthetic ISPV values."""
    band_junior = SalaryBand(low=40_000, mid=50_000, high=60_000)
    band_medior = SalaryBand(low=60_000, mid=75_000, high=90_000)
    band_senior = SalaryBand(low=90_000, mid=110_000, high=130_000)
    band_lead = SalaryBand(low=130_000, mid=145_000, high=165_000)
    band_principal = SalaryBand(low=160_000, mid=185_000, high=220_000)
    return SalaryData(
        role_name="Software Developer",
        role_slug=f"isco_{isco_code}",
        researched_at=datetime(2025, 3, 1, tzinfo=UTC),
        ttl_days=90,
        junior=band_junior,
        medior=band_medior,
        senior=band_senior,
        lead=band_lead,
        principal=band_principal,
        isco_code=isco_code,
        isco_match_level="4digit",
        dataset_age_months=2,
        inflation_factor_applied=1.02,
        source_dataset_version="ISPV-2025-annual",
        confidence="high",
        warnings=[],
    )


def _make_resume_with_isco(
    isco_code: str | None = "2512",
    location: str | None = None,
    has_management: bool = False,
    occupation_family: str | None = "it",
) -> Resume:
    """Build a minimal Resume with one Experience entry."""
    exp = Experience(
        role_title="Software Developer",
        isco_code=isco_code,
        occupation_family=occupation_family,  # type: ignore[arg-type]
        company="Test Corp",
        start_year=2018,
        end_year=2024,
        seniority_level="senior",
        is_management=has_management,
    )
    return Resume(
        full_name="Test User",
        location=location,
        experiences=[exp],
        raw_text_length=1000,
    )


def _make_mock_loader(salary_data: SalaryData | None = None) -> MagicMock:
    """Build a mock ISPVLoader that returns the given SalaryData on lookup().

    known_codes() is pre-stubbed to frozenset({"2512"}) so that MagicMock.__contains__
    does not silently route every test through the whitelist-fallback branch.
    Tests that need different whitelist behavior must override this explicitly.
    """
    loader = MagicMock()
    loader.is_loaded.return_value = True
    # Stub known_codes() with a real frozenset so that ISCO membership tests work correctly.
    # Without this, `isco_code not in loader.known_codes()` evaluates to True for all codes
    # because MagicMock.__contains__ returns False, triggering unintended whitelist-fallback paths.
    loader.known_codes.return_value = frozenset({"2512"})
    if salary_data is not None:
        loader.lookup.return_value = salary_data
    return loader


# ---------------------------------------------------------------------------
# _score_to_seniority
# ---------------------------------------------------------------------------


class TestScoreToSeniority:
    def test_below_30_is_junior(self) -> None:
        assert _score_to_seniority(0.0) == "junior"
        assert _score_to_seniority(29.9) == "junior"

    def test_30_to_59_is_medior(self) -> None:
        assert _score_to_seniority(30.0) == "medior"
        assert _score_to_seniority(59.9) == "medior"

    def test_60_to_79_is_senior(self) -> None:
        assert _score_to_seniority(60.0) == "senior"
        assert _score_to_seniority(79.9) == "senior"

    def test_80_to_91_is_lead(self) -> None:
        assert _score_to_seniority(80.0) == "lead"
        assert _score_to_seniority(91.9) == "lead"

    def test_92_and_above_is_principal(self) -> None:
        assert _score_to_seniority(92.0) == "principal"
        assert _score_to_seniority(100.0) == "principal"


# ---------------------------------------------------------------------------
# SalaryEstimator — no loader (hard fail path)
# ---------------------------------------------------------------------------


class TestSalaryEstimatorNoLoader:
    def test_raises_when_loader_is_none(self) -> None:
        """estimate() must raise ISPVLookupError when no loader provided."""
        estimator = SalaryEstimator(ispv_loader=None)
        resume = _make_resume_with_isco(isco_code="2512")
        score = _make_score(70.0)

        with pytest.raises(ISPVLookupError, match="ISPV data nejsou načtená"):
            estimator.estimate(resume, score)

    def test_raises_when_loader_not_loaded(self) -> None:
        """estimate() must raise ISPVLookupError when loader.is_loaded() returns False."""
        loader = MagicMock()
        loader.is_loaded.return_value = False
        estimator = SalaryEstimator(ispv_loader=loader)
        resume = _make_resume_with_isco(isco_code="2512")
        score = _make_score(70.0)

        with pytest.raises(ISPVLookupError, match="ISPV data nejsou načtená"):
            estimator.estimate(resume, score)

    def test_raises_when_no_isco_code_in_resume(self) -> None:
        """estimate() must raise ISPVLookupError when resume has no ISCO code."""
        loader = _make_mock_loader(_make_salary_data())
        estimator = SalaryEstimator(ispv_loader=loader)
        resume = _make_resume_with_isco(isco_code=None)
        score = _make_score(70.0)

        with pytest.raises(ISPVLookupError, match="ISCO"):
            estimator.estimate(resume, score)


# ---------------------------------------------------------------------------
# SalaryEstimator — happy path
# ---------------------------------------------------------------------------


class TestSalaryEstimatorHappyPath:
    def test_senior_returns_czk_estimate(self) -> None:
        """Senior score → senior band values returned as CZK estimate."""
        salary_data = _make_salary_data()
        loader = _make_mock_loader(salary_data)
        estimator = SalaryEstimator(ispv_loader=loader)

        resume = _make_resume_with_isco(isco_code="2512", location=None)
        score = _make_score(70.0)  # senior band

        result = estimator.estimate(resume, score)

        assert result.currency == "CZK"
        assert result.low > 0
        assert result.mid >= result.low
        assert result.high >= result.mid
        # Senior band base: 90k–130k (no multipliers)
        assert result.low >= 85_000, f"Senior low should be >= 85k, got {result.low}"
        assert result.high <= 140_000, f"Senior high should be <= 140k, got {result.high}"

    def test_junior_score_uses_junior_band(self) -> None:
        """Junior score → junior band applied (lower values than senior)."""
        salary_data = _make_salary_data()
        loader = _make_mock_loader(salary_data)
        estimator = SalaryEstimator(ispv_loader=loader)

        resume = _make_resume_with_isco(isco_code="2512", location=None)
        score_junior = _make_score(20.0)
        score_senior = _make_score(70.0)

        result_junior = estimator.estimate(resume, score_junior)
        result_senior = estimator.estimate(resume, score_senior)

        assert result_junior.mid < result_senior.mid, (
            f"Junior mid ({result_junior.mid}) should be lower than senior mid ({result_senior.mid})"
        )

    def test_result_contains_required_assumptions(self) -> None:
        """SalaryEstimate must contain seniority, ISCO, location, management, dataset fields."""
        salary_data = _make_salary_data()
        loader = _make_mock_loader(salary_data)
        estimator = SalaryEstimator(ispv_loader=loader)

        resume = _make_resume_with_isco(isco_code="2512")
        result = estimator.estimate(resume, _make_score(70.0))

        assumption_text = " | ".join(result.assumptions)
        assert "senior" in assumption_text.lower() or "Seniority" in assumption_text
        assert "2512" in assumption_text
        assert "ISPV" in assumption_text or "ispv" in assumption_text.lower()


# ---------------------------------------------------------------------------
# Location multipliers
# ---------------------------------------------------------------------------


class TestLocationMultiplier:
    def test_praha_multiplier_applied(self) -> None:
        """Praha location must yield higher estimate than no location (×1.15)."""
        salary_data = _make_salary_data()
        loader = _make_mock_loader(salary_data)
        estimator = SalaryEstimator(ispv_loader=loader)
        score = _make_score(70.0)

        result_no_loc = estimator.estimate(
            _make_resume_with_isco(isco_code="2512", location=None), score
        )
        result_praha = estimator.estimate(
            _make_resume_with_isco(isco_code="2512", location="Praha"), score
        )

        assert result_praha.mid > result_no_loc.mid, (
            f"Praha mid ({result_praha.mid}) should exceed no-location mid ({result_no_loc.mid})"
        )
        # Praha × 1.15 — verify ratio is roughly right
        ratio = result_praha.mid / result_no_loc.mid
        assert 1.10 <= ratio <= 1.20, f"Praha multiplier ratio out of expected range: {ratio:.3f}"

    def test_brno_multiplier_neutral(self) -> None:
        """Brno location is at national baseline (×1.00)."""
        salary_data = _make_salary_data()
        loader = _make_mock_loader(salary_data)
        estimator = SalaryEstimator(ispv_loader=loader)
        score = _make_score(55.0)

        result_no_loc = estimator.estimate(
            _make_resume_with_isco(isco_code="2512", location=None), score
        )
        result_brno = estimator.estimate(
            _make_resume_with_isco(isco_code="2512", location="Brno"), score
        )

        # Brno at ×1.00 means Brno mid equals or is very close to no-location mid
        assert result_brno.mid == result_no_loc.mid

    def test_ostrava_multiplier_reduces_salary(self) -> None:
        """Ostrava location must yield lower estimate than no location (×0.95)."""
        salary_data = _make_salary_data()
        loader = _make_mock_loader(salary_data)
        estimator = SalaryEstimator(ispv_loader=loader)
        score = _make_score(70.0)

        result_no_loc = estimator.estimate(
            _make_resume_with_isco(isco_code="2512", location=None), score
        )
        result_ostrava = estimator.estimate(
            _make_resume_with_isco(isco_code="2512", location="Ostrava"), score
        )

        assert result_ostrava.mid < result_no_loc.mid, (
            f"Ostrava mid ({result_ostrava.mid}) should be below no-location mid ({result_no_loc.mid})"
        )

    def test_plain_remote_no_multiplier(self) -> None:
        """Plain 'remote' (no country qualifier) is treated as CZ employer with ×1.0.

        The previous Remote-EU / Remote-US multipliers were retired — they
        violated the README's CZ-only contract. Country-qualified remote
        locations now hard-fail at the non-CZ guard (covered separately).
        """
        salary_data = _make_salary_data()
        loader = _make_mock_loader(salary_data)
        estimator = SalaryEstimator(ispv_loader=loader)
        score = _make_score(70.0)

        result_no_loc = estimator.estimate(
            _make_resume_with_isco(isco_code="2512", location=None), score
        )
        result_remote = estimator.estimate(
            _make_resume_with_isco(isco_code="2512", location="remote"), score
        )

        assert result_remote.mid == result_no_loc.mid

    def test_unknown_location_no_multiplier(self) -> None:
        """CZ city not in the multiplier table should default to ×1.00."""
        salary_data = _make_salary_data()
        loader = _make_mock_loader(salary_data)
        estimator = SalaryEstimator(ispv_loader=loader)
        score = _make_score(70.0)

        result_none = estimator.estimate(
            _make_resume_with_isco(isco_code="2512", location=None), score
        )
        result_unknown = estimator.estimate(
            _make_resume_with_isco(isco_code="2512", location="Kladno"), score
        )

        assert result_unknown.mid == result_none.mid, (
            "CZ city not in multiplier table should not change mid salary"
        )

    def test_zlin_regional_discount(self) -> None:
        """Regional CZ city (Zlín) should yield lower estimate than unknown CZ city (×0.92)."""
        salary_data = _make_salary_data()
        loader = _make_mock_loader(salary_data)
        estimator = SalaryEstimator(ispv_loader=loader)
        score = _make_score(70.0)

        result_none = estimator.estimate(
            _make_resume_with_isco(isco_code="2512", location=None), score
        )
        result_zlin = estimator.estimate(
            _make_resume_with_isco(isco_code="2512", location="Zlín"), score
        )

        ratio = result_zlin.mid / result_none.mid
        # Zlín multiplier = ×0.85 — regional calibration vs national baseline
        assert 0.80 <= ratio <= 0.90, f"Zlín multiplier ratio out of expected range: {ratio:.3f}"


# ---------------------------------------------------------------------------
# Management multiplier
# ---------------------------------------------------------------------------


class TestManagementMultiplier:
    def test_management_role_yields_higher_salary(self) -> None:
        """is_management=True must produce higher estimate than non-management (×1.10)."""
        salary_data = _make_salary_data()
        loader = _make_mock_loader(salary_data)
        estimator = SalaryEstimator(ispv_loader=loader)
        score = _make_score(70.0)

        result_mgmt = estimator.estimate(
            _make_resume_with_isco(isco_code="2512", has_management=True), score
        )
        result_no_mgmt = estimator.estimate(
            _make_resume_with_isco(isco_code="2512", has_management=False), score
        )

        assert result_mgmt.mid > result_no_mgmt.mid, (
            f"Management mid ({result_mgmt.mid}) should exceed non-management ({result_no_mgmt.mid})"
        )

    def test_management_multiplier_compounds_with_location(self) -> None:
        """Praha + management should compound (×1.15 × ×1.10 = ×1.265)."""
        salary_data = _make_salary_data()
        loader = _make_mock_loader(salary_data)
        estimator = SalaryEstimator(ispv_loader=loader)
        score = _make_score(70.0)

        result_base = estimator.estimate(
            _make_resume_with_isco(isco_code="2512", location=None, has_management=False), score
        )
        result_compound = estimator.estimate(
            _make_resume_with_isco(isco_code="2512", location="Praha", has_management=True), score
        )

        ratio = result_compound.mid / result_base.mid
        # Expected: 1.15 (Praha) × 1.10 (mgmt) = 1.265 → allow ±5 % rounding
        assert 1.20 <= ratio <= 1.33, f"Compound multiplier ratio out of range: {ratio:.3f}"

    def test_management_multiplier_skipped_for_lead_band(self) -> None:
        """Lead/principal bands extrapolate above D9 (×1.10/1.20/1.35) and already
        absorb the management upside; stacking ×1.10 on top double-counts and
        overshoots reality. Regression for the eval golden-set principal CV that
        produced 691 k CZK before the fix.
        """
        salary_data = _make_salary_data()
        loader = _make_mock_loader(salary_data)
        estimator = SalaryEstimator(ispv_loader=loader)
        # score 85 → lead band per _score_to_seniority
        lead_score = _make_score(85.0)

        result_lead_no_mgmt = estimator.estimate(
            _make_resume_with_isco(isco_code="2512", has_management=False), lead_score
        )
        result_lead_with_mgmt = estimator.estimate(
            _make_resume_with_isco(isco_code="2512", has_management=True), lead_score
        )

        assert result_lead_with_mgmt.mid == result_lead_no_mgmt.mid, (
            "Lead band must NOT add management multiplier — D9×1.20 already absorbs it"
        )

    def test_management_multiplier_skipped_for_principal_band(self) -> None:
        """Same as lead — principal band uses D9 × 1.30/1.50/1.75; mgmt stacking
        would push estimates well past any plausible CZ market level.
        """
        salary_data = _make_salary_data()
        loader = _make_mock_loader(salary_data)
        estimator = SalaryEstimator(ispv_loader=loader)
        # score 95 → principal band
        principal_score = _make_score(95.0)

        result_no_mgmt = estimator.estimate(
            _make_resume_with_isco(isco_code="2512", has_management=False), principal_score
        )
        result_with_mgmt = estimator.estimate(
            _make_resume_with_isco(isco_code="2512", has_management=True), principal_score
        )

        assert result_with_mgmt.high == result_no_mgmt.high, (
            "Principal band must NOT add management multiplier — D9×1.75 already absorbs it"
        )


# ---------------------------------------------------------------------------
# last_salary_source / last_salary_data
# ---------------------------------------------------------------------------


class TestEstimatorProperties:
    def test_last_salary_source_is_ispv_after_estimate(self) -> None:
        """last_salary_source must return 'ispv' after a successful estimate."""
        loader = _make_mock_loader(_make_salary_data())
        estimator = SalaryEstimator(ispv_loader=loader)
        estimator.estimate(_make_resume_with_isco("2512"), _make_score(70.0))

        assert estimator.last_salary_source == "ispv"

    def test_last_salary_data_populated_after_estimate(self) -> None:
        """last_salary_data must be non-None after a successful estimate."""
        salary_data = _make_salary_data()
        loader = _make_mock_loader(salary_data)
        estimator = SalaryEstimator(ispv_loader=loader)
        estimator.estimate(_make_resume_with_isco("2512"), _make_score(70.0))

        assert estimator.last_salary_data is not None
        assert estimator.last_salary_data.isco_code == "2512"

    def test_last_salary_source_generic_fallback_before_estimate(self) -> None:
        """Before any estimate(), last_salary_source default depends on loader presence."""
        estimator_no_loader = SalaryEstimator(ispv_loader=None)
        assert estimator_no_loader.last_salary_source == "generic_fallback"

        loader = _make_mock_loader(_make_salary_data())
        estimator_with_loader = SalaryEstimator(ispv_loader=loader)
        assert estimator_with_loader.last_salary_source == "ispv"


# ---------------------------------------------------------------------------
# Bug 1: ISCO whitelist — family fallback (tests B)
# ---------------------------------------------------------------------------


class TestISCOWhitelistFallback:
    """Bug 1 regression: whitelist miss must trigger family fallback, not hard error."""

    def test_whitelist_miss_triggers_family_fallback(self) -> None:
        """When isco_code absent from known_codes(), best_code_for_family() provides fallback."""
        salary_data = _make_salary_data(isco_code="2512")
        loader = _make_mock_loader(salary_data)
        # Simulate: "9999" not in whitelist, but family "it" has a best code "2512"
        loader.known_codes.return_value = frozenset({"2512"})  # "9999" NOT in set
        loader.best_code_for_family.return_value = "2512"

        resume = _make_resume_with_isco(isco_code="9999", occupation_family="it")
        estimator = SalaryEstimator(ispv_loader=loader)

        result = estimator.estimate(resume, _make_score(70.0))

        # Must not raise — should succeed via family fallback
        assert result.currency == "CZK"
        assert result.low > 0
        # best_code_for_family must have been called with "it"
        loader.best_code_for_family.assert_called_once_with("it")

    def test_whitelist_miss_no_family_raises(self) -> None:
        """When isco_code not in whitelist and occupation_family is None → ISPVLookupError."""
        loader = _make_mock_loader(_make_salary_data())
        loader.known_codes.return_value = frozenset({"2512"})  # "9999" NOT in set

        # No occupation_family on the experience
        resume = _make_resume_with_isco(isco_code="9999", occupation_family=None)
        estimator = SalaryEstimator(ispv_loader=loader)

        with pytest.raises(ISPVLookupError, match="occupation_family"):
            estimator.estimate(resume, _make_score(70.0))

    def test_whitelist_miss_no_family_data_raises(self) -> None:
        """When isco_code not in whitelist and best_code_for_family returns None → ISPVLookupError."""
        loader = _make_mock_loader(_make_salary_data())
        loader.known_codes.return_value = frozenset({"2512"})  # "9999" NOT in set
        loader.best_code_for_family.return_value = None  # no family data available

        resume = _make_resume_with_isco(isco_code="9999", occupation_family="it")
        estimator = SalaryEstimator(ispv_loader=loader)

        with pytest.raises(ISPVLookupError, match="occupation_family"):
            estimator.estimate(resume, _make_score(70.0))


# ---------------------------------------------------------------------------
# Bug 3: Non-CZ location hard error (tests C)
# ---------------------------------------------------------------------------


class TestNonCZLocationGuard:
    """Bug 3 regression: non-CZ location must hard-fail with ISPVLookupError."""

    def test_non_cz_location_berlin_raises(self) -> None:
        """location='Berlin' (German city) must raise ISPVLookupError with 'CZ trh'."""
        loader = _make_mock_loader(_make_salary_data())
        estimator = SalaryEstimator(ispv_loader=loader)
        resume = _make_resume_with_isco(isco_code="2512", location="Berlin")

        with pytest.raises(ISPVLookupError, match="CZ trh"):
            estimator.estimate(resume, _make_score(70.0))

    def test_non_cz_location_london_raises(self) -> None:
        """location='London, UK' must raise ISPVLookupError."""
        loader = _make_mock_loader(_make_salary_data())
        estimator = SalaryEstimator(ispv_loader=loader)
        resume = _make_resume_with_isco(isco_code="2512", location="London, UK")

        with pytest.raises(ISPVLookupError, match="CZ trh"):
            estimator.estimate(resume, _make_score(70.0))

    def test_cz_praha_passes(self) -> None:
        """location='Praha' must pass the CZ location check (may fail later for other reasons)."""
        loader = _make_mock_loader(_make_salary_data())
        estimator = SalaryEstimator(ispv_loader=loader)
        resume = _make_resume_with_isco(isco_code="2512", location="Praha")

        # Must NOT raise ISPVLookupError about CZ trh — Praha is a known CZ city
        result = estimator.estimate(resume, _make_score(70.0))
        assert result.currency == "CZK"

    def test_cz_olomouc_passes(self) -> None:
        """location='Olomouc' must pass the CZ location check."""
        loader = _make_mock_loader(_make_salary_data())
        estimator = SalaryEstimator(ispv_loader=loader)
        resume = _make_resume_with_isco(isco_code="2512", location="Olomouc")

        result = estimator.estimate(resume, _make_score(70.0))
        assert result.currency == "CZK"

    def test_empty_location_passes(self) -> None:
        """location=None must pass (unknown → assume CZ)."""
        loader = _make_mock_loader(_make_salary_data())
        estimator = SalaryEstimator(ispv_loader=loader)
        resume = _make_resume_with_isco(isco_code="2512", location=None)

        result = estimator.estimate(resume, _make_score(70.0))
        assert result.currency == "CZK"

    def test_remote_location_passes(self) -> None:
        """location='remote' must pass (remote → assumed CZ employer)."""
        loader = _make_mock_loader(_make_salary_data())
        estimator = SalaryEstimator(ispv_loader=loader)
        resume = _make_resume_with_isco(isco_code="2512", location="remote")

        result = estimator.estimate(resume, _make_score(70.0))
        assert result.currency == "CZK"

    def test_szczecin_poland_raises(self) -> None:
        """Regression: 'Szczecin, Poland' must NOT pass via 'cz' substring match."""
        loader = _make_mock_loader(_make_salary_data())
        estimator = SalaryEstimator(ispv_loader=loader)
        resume = _make_resume_with_isco(isco_code="2512", location="Szczecin, Poland")

        with pytest.raises(ISPVLookupError, match="CZ trh"):
            estimator.estimate(resume, _make_score(70.0))

    def test_remote_us_raises(self) -> None:
        """Regression: 'Remote US' must raise — non-CZ country indicator overrides 'remote'."""
        loader = _make_mock_loader(_make_salary_data())
        estimator = SalaryEstimator(ispv_loader=loader)
        resume = _make_resume_with_isco(isco_code="2512", location="Remote US")

        with pytest.raises(ISPVLookupError, match="CZ trh"):
            estimator.estimate(resume, _make_score(70.0))

    def test_eu_token_alone_does_not_raise(self) -> None:
        """Bare 'EU'/'europe' tokens are too weak to hard-fail the lookup.

        Lots of legitimate CZ profiles describe themselves as 'EU citizen,
        Praha-based' or carry strings like 'European Engineering Center,
        Brno' in the location field. A specific country/city match (Berlin,
        Bratislava, …) is still required to flip to non-CZ.
        """
        loader = _make_mock_loader(_make_salary_data())
        estimator = SalaryEstimator(ispv_loader=loader)
        for location in ("Remote EU", "Europe remote", "European Center, Brno"):
            resume = _make_resume_with_isco(isco_code="2512", location=location)
            result = estimator.estimate(resume, _make_score(70.0))
            assert result.currency == "CZK", (
                f"{location!r} must not be classified as non-CZ on the EU token alone"
            )

    def test_bratislava_raises(self) -> None:
        """Bratislava (Slovakia) must raise — neighboring country, not CZ."""
        loader = _make_mock_loader(_make_salary_data())
        estimator = SalaryEstimator(ispv_loader=loader)
        resume = _make_resume_with_isco(isco_code="2512", location="Bratislava")

        with pytest.raises(ISPVLookupError, match="CZ trh"):
            estimator.estimate(resume, _make_score(70.0))

    def test_smaller_cz_cities_pass(self) -> None:
        """Cities outside the multiplier table (Tábor, Karlovy Vary, Mladá Boleslav)
        must pass the location check — default is CZ unless an explicit non-CZ
        country indicator is present.
        """
        loader = _make_mock_loader(_make_salary_data())
        estimator = SalaryEstimator(ispv_loader=loader)
        for city in ("Tábor", "Karlovy Vary", "Mladá Boleslav", "Náchod"):
            resume = _make_resume_with_isco(isco_code="2512", location=city)
            result = estimator.estimate(resume, _make_score(70.0))
            assert result.currency == "CZK", f"{city} must not be classified as non-CZ"


class TestExperienceRecencySelection:
    """Verify that ISCO is taken from the candidate's CURRENT primary role,
    not whichever experience started most recently.
    """

    def test_short_side_gig_does_not_override_current_role(self) -> None:
        """Long-running current role must beat a short side gig that started later."""
        from src.schemas import Experience, Resume

        current_main = Experience(
            role_title="Senior Software Engineer",
            isco_code="2512",
            occupation_family="it",
            company="MainCorp",
            start_year=2020,
            end_year=None,  # ongoing
        )
        short_side_gig = Experience(
            role_title="Cook (weekend)",
            isco_code="5120",
            occupation_family="services",
            company="SideRestaurant",
            start_year=2023,
            end_year=2024,
        )
        resume = Resume(
            full_name="Test",
            location=None,
            experiences=[current_main, short_side_gig],
            raw_text_length=1000,
        )

        loader = _make_mock_loader(_make_salary_data(isco_code="2512"))
        estimator = SalaryEstimator(ispv_loader=loader)
        estimator.estimate(resume, _make_score(70.0))

        # Verify the loader was queried for the current main role's ISCO (2512),
        # not the short side gig's ISCO (5120).
        loader.lookup.assert_called_with("2512")

    def test_longer_ongoing_role_wins_over_later_started_ongoing(self) -> None:
        """Two ongoing roles → the one started earlier (longer tenure) wins.

        Regression for the recency_key tie-break: previously sorted ongoing
        roles by start_year descending, so a fresh side gig started in 2024
        outranked the 2018 main job. The fixed key flips the tie-break.
        """
        from src.schemas import Experience, Resume

        long_main = Experience(
            role_title="Backend Lead",
            isco_code="2512",
            occupation_family="it",
            company="MainCorp",
            start_year=2018,
            end_year=None,
        )
        short_freelance = Experience(
            role_title="Freelance Cook",
            isco_code="5120",
            occupation_family="services",
            company="WeekendKitchen",
            start_year=2024,
            end_year=None,
            is_contractor=True,
        )
        resume = Resume(
            full_name="Test",
            location=None,
            experiences=[long_main, short_freelance],
            raw_text_length=1000,
        )

        loader = _make_mock_loader(_make_salary_data(isco_code="2512"))
        estimator = SalaryEstimator(ispv_loader=loader)
        estimator.estimate(resume, _make_score(70.0))

        loader.lookup.assert_called_with("2512")
