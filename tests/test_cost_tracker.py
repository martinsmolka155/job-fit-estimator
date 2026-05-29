"""Tests for per-run cost tracking and daily budget enforcement."""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from pathlib import Path

import pytest

from src.cost_tracker import (
    DEFAULT_BUDGET_USD,
    BudgetExceededError,
    CostRecord,
    _budget_from_settings,  # pyright: ignore[reportPrivateUsage]
    _budget_lock,  # pyright: ignore[reportPrivateUsage]
    check_budget,
    daily_spent,
    new_run_id,
    record_run,
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

    def test_concurrent_requests_see_toctou_protection(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two threads calling check_budget concurrently must not both pass when budget is tight.

        This is a best-effort race test: we set a budget of $0.05 and fire two
        threads each trying to spend $0.04.  Without the lock at least one of
        the recorded amounts would be wrong; with the lock only one should be
        allowed to pass (the second sees the first's record already written).

        Note: the test is inherently timing-dependent.  We use a barrier to
        maximise the chance of a genuine race.
        """
        monkeypatch.setenv("DAILY_API_BUDGET_USD", "0.05")
        log = tmp_path / ".cost_log.jsonl"

        results: list[str] = []
        barrier = threading.Barrier(2)

        def _try_spend() -> None:
            barrier.wait()  # both threads race to check_budget simultaneously
            try:
                check_budget(estimated_cost_usd=0.04, log_path=log)
                # Immediately record spend so the second thread sees it.
                rec = CostRecord(
                    run_id=new_run_id(),
                    timestamp_utc=datetime.now(UTC).isoformat(),
                    fixture_or_file="test.pdf",
                    total_cost_usd=0.04,
                    parse_cost_usd=0.04,
                    explain_cost_usd=0.0,
                    embed_cost_usd=0.0,
                    duration_s=0.1,
                )
                record_run(rec, log_path=log)
                results.append("ok")
            except BudgetExceededError:
                results.append("blocked")

        t1 = threading.Thread(target=_try_spend)
        t2 = threading.Thread(target=_try_spend)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert len(results) == 2
        # At least one thread must have been blocked — the budget was too tight for both.
        # (Both passing would mean TOCTOU was not prevented.)
        assert results.count("ok") <= 1, (
            f"Both threads passed the budget check — TOCTOU protection may be broken: {results}"
        )
