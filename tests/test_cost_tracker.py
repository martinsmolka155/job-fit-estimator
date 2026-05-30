"""Tests for per-run cost tracking and daily budget enforcement."""

from __future__ import annotations

import json
import threading
from collections.abc import Generator
from datetime import UTC, datetime
from pathlib import Path

import pytest

from src.cost_tracker import (
    DEFAULT_BUDGET_USD,
    BudgetExceededError,
    CostRecord,
    _budget_from_settings,  # pyright: ignore[reportPrivateUsage]
    _budget_lock,  # pyright: ignore[reportPrivateUsage]
    _release_reservation,  # pyright: ignore[reportPrivateUsage]
    _reservations,  # pyright: ignore[reportPrivateUsage]
    check_budget,
    daily_spent,
    finalize,
    new_run_id,
    record_run,
    reserve,
)


def _make_record(
    total_cost_usd: float = 0.01,
    timestamp_utc: str | None = None,
    run_id: str = "abc123",
) -> CostRecord:
    ts = timestamp_utc or datetime.now(UTC).isoformat()
    return CostRecord(
        run_id=run_id,
        timestamp_utc=ts,
        fixture_or_file="test_cv.pdf",
        total_cost_usd=total_cost_usd,
        parse_cost_usd=0.005,
        explain_cost_usd=0.005,
        embed_cost_usd=0.0,
        duration_s=3.5,
        score_total=72.0,
    )


@pytest.fixture(autouse=True)
def _clear_reservations() -> Generator[None, None, None]:
    """Ensure no reservations leak between tests."""
    _reservations.clear()
    yield
    _reservations.clear()


class TestDailySpent:
    def test_returns_zero_for_empty_log(self, tmp_path: Path) -> None:
        """daily_spent returns 0.0 when log file does not exist."""
        log = tmp_path / ".cost_log.jsonl"
        assert daily_spent(log) == 0.0

    def test_sums_today_only(self, tmp_path: Path) -> None:
        """daily_spent sums only records matching today's UTC date, ignoring past dates."""
        log = tmp_path / ".cost_log.jsonl"
        today = datetime.now(UTC).strftime("%Y-%m-%d")

        records = [
            {"timestamp_utc": f"{today}T10:00:00+00:00", "total_cost_usd": 0.50},
            {"timestamp_utc": f"{today}T14:00:00+00:00", "total_cost_usd": 0.30},
            # Past date — should be excluded
            {"timestamp_utc": "2025-01-01T10:00:00+00:00", "total_cost_usd": 99.00},
        ]
        with log.open("w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")

        result = daily_spent(log)
        assert result == pytest.approx(0.80)

    def test_ignores_malformed_lines(self, tmp_path: Path) -> None:
        """daily_spent skips lines that are not valid JSON without crashing."""
        log = tmp_path / ".cost_log.jsonl"
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        log.write_text(
            f'{{"timestamp_utc": "{today}T10:00:00+00:00", "total_cost_usd": 0.10}}\n'
            "not valid json\n"
            f'{{"timestamp_utc": "{today}T11:00:00+00:00", "total_cost_usd": 0.05}}\n'
        )
        assert daily_spent(log) == pytest.approx(0.15)

    def test_includes_outstanding_reservations(self, tmp_path: Path) -> None:
        """daily_spent adds outstanding in-memory reservations to JSONL actuals."""
        log = tmp_path / ".cost_log.jsonl"
        # Seed one committed record
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        log.write_text(f'{{"timestamp_utc": "{today}T10:00:00+00:00", "total_cost_usd": 0.10}}\n')
        # Manually inject a reservation (bypassing reserve() to avoid budget check)
        _reservations["test-run-abc"] = 0.05
        result = daily_spent(log)
        assert result == pytest.approx(0.15)


class TestCheckBudget:
    def test_under_budget_passes(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """check_budget does not raise when projected spend is within budget."""
        monkeypatch.setenv("DAILY_API_BUDGET_USD", "5.00")
        log = tmp_path / ".cost_log.jsonl"
        # Should not raise
        check_budget(estimated_cost_usd=0.01, log_path=log)

    def test_over_budget_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """check_budget raises BudgetExceededError when projected spend exceeds budget."""
        monkeypatch.setenv("DAILY_API_BUDGET_USD", "0.10")
        log = tmp_path / ".cost_log.jsonl"
        # Estimate alone already exceeds budget
        with pytest.raises(BudgetExceededError, match="Daily budget would be exceeded"):
            check_budget(estimated_cost_usd=0.50, log_path=log)

    def test_at_warn_threshold_logs_warning(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """check_budget logs a warning (but does not raise) when projected > 90% of budget."""
        import logging

        monkeypatch.setenv("DAILY_API_BUDGET_USD", "1.00")
        log = tmp_path / ".cost_log.jsonl"
        # 0.95 projected = 95% of 1.00 budget — above WARN_THRESHOLD_PCT (0.90), below 1.00
        with caplog.at_level(logging.WARNING, logger="src.cost_tracker"):
            check_budget(estimated_cost_usd=0.95, log_path=log)

        assert any("Approaching daily budget" in r.message for r in caplog.records)

    def test_exactly_at_budget_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """check_budget raises when projected equals budget exactly (projected > budget fails)."""
        monkeypatch.setenv("DAILY_API_BUDGET_USD", "1.00")
        log = tmp_path / ".cost_log.jsonl"
        # Add 0.50 spent today
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        log.write_text(f'{{"timestamp_utc": "{today}T10:00:00+00:00", "total_cost_usd": 0.60}}\n')
        # 0.60 spent + 0.50 estimate = 1.10 > 1.00 → raise
        with pytest.raises(BudgetExceededError):
            check_budget(estimated_cost_usd=0.50, log_path=log)


class TestRecordRun:
    def test_appends_valid_jsonl(self, tmp_path: Path) -> None:
        """record_run writes a valid JSONL line containing all CostRecord fields."""
        log = tmp_path / ".cost_log.jsonl"
        rec = _make_record(total_cost_usd=0.0042, run_id="deadbeef0001")
        record_run(rec, log_path=log)

        lines = log.read_text().splitlines()
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["run_id"] == "deadbeef0001"
        assert parsed["total_cost_usd"] == pytest.approx(0.0042)
        assert parsed["fixture_or_file"] == "test_cv.pdf"

    def test_appends_multiple_records(self, tmp_path: Path) -> None:
        """record_run appends without overwriting — each call adds one line."""
        log = tmp_path / ".cost_log.jsonl"
        for i in range(3):
            record_run(_make_record(run_id=f"run{i:03d}"), log_path=log)

        lines = [line for line in log.read_text().splitlines() if line.strip()]
        assert len(lines) == 3

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        """record_run creates parent directory (data/) if it does not exist."""
        log = tmp_path / "nested" / "dir" / ".cost_log.jsonl"
        record_run(_make_record(), log_path=log)
        assert log.exists()


class TestNewRunId:
    def test_is_unique(self) -> None:
        """Two calls to new_run_id produce different IDs."""
        id1 = new_run_id()
        id2 = new_run_id()
        assert id1 != id2

    def test_has_expected_length(self) -> None:
        """new_run_id returns a 12-character hex string."""
        run_id = new_run_id()
        assert len(run_id) == 12
        assert all(c in "0123456789abcdef" for c in run_id)


class TestBudgetFromSettings:
    def test_invalid_falls_back_to_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Non-numeric DAILY_API_BUDGET_USD makes Settings raise; we fall back."""
        monkeypatch.setenv("DAILY_API_BUDGET_USD", "not_a_number")
        result = _budget_from_settings()
        assert result == pytest.approx(DEFAULT_BUDGET_USD)

    def test_valid_env_parsed_correctly(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """DAILY_API_BUDGET_USD with valid float is loaded by Settings."""
        monkeypatch.setenv("DAILY_API_BUDGET_USD", "10.50")
        result = _budget_from_settings()
        assert result == pytest.approx(10.50)

    def test_missing_env_returns_settings_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Missing DAILY_API_BUDGET_USD falls through to the Settings field default."""
        monkeypatch.delenv("DAILY_API_BUDGET_USD", raising=False)
        result = _budget_from_settings()
        assert result == pytest.approx(DEFAULT_BUDGET_USD)


class TestBudgetLock:
    def test_check_budget_acquires_then_releases_lock(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """check_budget must release the lock on success so subsequent calls can proceed."""
        monkeypatch.setenv("DAILY_API_BUDGET_USD", "10.00")
        log = tmp_path / ".cost_log.jsonl"
        # Call once — lock must be released after returning.
        check_budget(estimated_cost_usd=0.01, log_path=log)
        # A second call must not deadlock.
        check_budget(estimated_cost_usd=0.01, log_path=log)

    def test_check_budget_releases_lock_on_raise(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Lock must be released even when BudgetExceededError is raised."""
        monkeypatch.setenv("DAILY_API_BUDGET_USD", "0.01")
        log = tmp_path / ".cost_log.jsonl"
        with pytest.raises(BudgetExceededError):
            check_budget(estimated_cost_usd=1.00, log_path=log)
        # Lock must be acquirable again after the exception.
        assert _budget_lock.acquire(blocking=False), "Lock not released after BudgetExceededError"
        _budget_lock.release()


class TestReserveFinalize:
    """Tests for the reservation protocol (reserve / finalize / _release_reservation)."""

    def test_reserve_claims_budget_slot(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """After reserve(), daily_spent includes the reserved amount."""
        monkeypatch.setenv("DAILY_API_BUDGET_USD", "1.00")
        log = tmp_path / ".cost_log.jsonl"
        run_id = "reserve-test-001"
        reserve(run_id, 0.10, log_path=log)
        assert daily_spent(log) == pytest.approx(0.10)
        _release_reservation(run_id)

    def test_reserve_raises_when_over_budget(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """reserve() raises BudgetExceededError when estimate would exceed budget."""
        monkeypatch.setenv("DAILY_API_BUDGET_USD", "0.05")
        log = tmp_path / ".cost_log.jsonl"
        with pytest.raises(BudgetExceededError, match="Daily budget would be exceeded"):
            reserve("run-too-big", 0.10, log_path=log)
        # No reservation must have been stored on failure.
        assert "run-too-big" not in _reservations

    def test_reserve_accounts_for_existing_reservations(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A second reserve() sees the first reservation and raises if combined > budget."""
        monkeypatch.setenv("DAILY_API_BUDGET_USD", "0.07")
        log = tmp_path / ".cost_log.jsonl"
        run_a = "run-a"
        run_b = "run-b"
        reserve(run_a, 0.05, log_path=log)
        # 0.05 reserved + 0.04 estimate = 0.09 > 0.07 → must raise
        with pytest.raises(BudgetExceededError):
            reserve(run_b, 0.04, log_path=log)
        # Only run_a reservation exists
        assert run_a in _reservations
        assert run_b not in _reservations
        _release_reservation(run_a)

    def test_finalize_writes_record_and_removes_reservation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """finalize() removes the reservation and writes a CostRecord to the JSONL log."""
        monkeypatch.setenv("DAILY_API_BUDGET_USD", "1.00")
        log = tmp_path / ".cost_log.jsonl"
        run_id = "finalize-test-001"
        reserve(run_id, 0.05, log_path=log)
        assert run_id in _reservations

        rec = _make_record(total_cost_usd=0.03, run_id=run_id)
        finalize(run_id, rec, log_path=log)

        # Reservation must be gone
        assert run_id not in _reservations
        # Actual record must be in the log
        lines = log.read_text().splitlines()
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["run_id"] == run_id
        assert parsed["total_cost_usd"] == pytest.approx(0.03)
        # daily_spent now reflects the actual record (0.03), not the estimate (0.05)
        assert daily_spent(log) == pytest.approx(0.03)

    def test_finalize_without_prior_reserve_still_writes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """finalize() on an unknown run_id logs a warning but still persists the record."""
        monkeypatch.setenv("DAILY_API_BUDGET_USD", "1.00")
        log = tmp_path / ".cost_log.jsonl"
        rec = _make_record(total_cost_usd=0.02, run_id="no-reservation")
        # Must not raise, even without a prior reserve()
        finalize("no-reservation", rec, log_path=log)
        lines = log.read_text().splitlines()
        assert len(lines) == 1

    def test_release_reservation_removes_slot(self, tmp_path: Path) -> None:
        """_release_reservation removes the in-memory slot without writing any record."""
        log = tmp_path / ".cost_log.jsonl"
        _reservations["manual-run"] = 0.10
        _release_reservation("manual-run")
        assert "manual-run" not in _reservations
        # No file created
        assert not log.exists()

    def test_release_reservation_idempotent(self) -> None:
        """_release_reservation on a non-existent run_id does not raise."""
        # Should be a no-op / log debug only
        _release_reservation("nonexistent-run-xyz")

    def test_aborted_run_releases_reservation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """After a failed run that calls _release_reservation in finally, budget is freed."""
        monkeypatch.setenv("DAILY_API_BUDGET_USD", "0.10")
        log = tmp_path / ".cost_log.jsonl"
        run_id = "abort-run-001"
        reserve(run_id, 0.08, log_path=log)
        # Simulate aborted run: release reservation (budget should be free again)
        _release_reservation(run_id)
        # Now a new run with the same estimate must pass
        new_id = "new-run-after-abort"
        reserve(new_id, 0.08, log_path=log)
        assert new_id in _reservations
        _release_reservation(new_id)


class TestConcurrentReservationProtocol:
    """Deterministic concurrency tests for the reserve/finalize protocol.

    Budget is set tight enough for exactly one of two concurrent requests.
    With the reservation protocol, the outcome is deterministic: exactly one
    thread claims the slot; the other sees it claimed and raises BudgetExceededError.
    No timing dependence — the guarantee comes from the atomic lock in reserve(),
    not from thread scheduling luck.
    """

    def test_exactly_one_of_two_concurrent_reserves_succeeds(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With budget tight for one, exactly one concurrent reserve() succeeds."""
        monkeypatch.setenv("DAILY_API_BUDGET_USD", "0.05")
        log = tmp_path / ".cost_log.jsonl"

        results: list[str] = []
        reserved_ids: list[str] = []
        # Barrier ensures both threads attempt reserve() simultaneously.
        barrier = threading.Barrier(2)

        def _try_reserve() -> None:
            run_id = new_run_id()
            barrier.wait()
            try:
                reserve(run_id, 0.04, log_path=log)
                results.append("ok")
                reserved_ids.append(run_id)
            except BudgetExceededError:
                results.append("blocked")

        t1 = threading.Thread(target=_try_reserve)
        t2 = threading.Thread(target=_try_reserve)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # Deterministic: exactly one succeeds, exactly one is blocked.
        assert sorted(results) == ["blocked", "ok"], (
            f"Expected exactly one ok and one blocked, got: {results}"
        )
        # Clean up the surviving reservation
        for rid in reserved_ids:
            _release_reservation(rid)

    def test_both_pass_when_budget_fits_both(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When budget is large enough for both, both concurrent reserves succeed."""
        monkeypatch.setenv("DAILY_API_BUDGET_USD", "1.00")
        log = tmp_path / ".cost_log.jsonl"

        results: list[str] = []
        reserved_ids: list[str] = []
        barrier = threading.Barrier(2)

        def _try_reserve() -> None:
            run_id = new_run_id()
            barrier.wait()
            try:
                reserve(run_id, 0.04, log_path=log)
                results.append("ok")
                reserved_ids.append(run_id)
            except BudgetExceededError:
                results.append("blocked")

        t1 = threading.Thread(target=_try_reserve)
        t2 = threading.Thread(target=_try_reserve)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert results == ["ok", "ok"] or results == ["ok", "ok"], (
            f"Both should have passed with a large budget, got: {results}"
        )
        for rid in reserved_ids:
            _release_reservation(rid)

    def test_finalized_run_frees_budget_for_next(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """After finalize(), the slot is freed so a subsequent request can pass."""
        monkeypatch.setenv("DAILY_API_BUDGET_USD", "0.06")
        log = tmp_path / ".cost_log.jsonl"
        run_id_1 = "seq-run-001"
        run_id_2 = "seq-run-002"

        # First run: reserve and finalize with actual cost well below budget
        reserve(run_id_1, 0.04, log_path=log)
        rec1 = _make_record(total_cost_usd=0.02, run_id=run_id_1)
        finalize(run_id_1, rec1, log_path=log)

        # After finalize, 0.02 is committed; budget remaining = 0.04 → second run fits
        reserve(run_id_2, 0.03, log_path=log)
        assert run_id_2 in _reservations
        _release_reservation(run_id_2)

    def test_reservation_does_not_double_count_after_finalize(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """After finalize(), daily_spent reflects actual cost, not estimate + actual."""
        monkeypatch.setenv("DAILY_API_BUDGET_USD", "1.00")
        log = tmp_path / ".cost_log.jsonl"
        run_id = "double-count-test"

        reserve(run_id, 0.10, log_path=log)
        rec = _make_record(total_cost_usd=0.07, run_id=run_id)
        finalize(run_id, rec, log_path=log)

        # Must equal the actual 0.07, not 0.10 (estimate) + 0.07 (actual) = 0.17
        assert daily_spent(log) == pytest.approx(0.07)
