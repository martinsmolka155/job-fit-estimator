"""Shared pytest fixtures for the job-fit-estimator test suite."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def ispv_sample_xlsx(tmp_path: Path) -> Path:
    """Generate a synthetic ISPV M8r XLSX matching the real-world layout.

    Sheet name MZS-M8r. Rows 0-7 = merged header band (filler). Row 8 = empty
    separator. Data rows start at row 9 (0-indexed row 8 is the first data row
    after skipping the header band of rows 0-7).

    Layout (positional columns):
        Col 0 = "ISCO code + space + role name" label
        Col 1 = POCET in thousands of persons
        Col 2 = MEDIAN  (CZK/month)
        Col 3 = D1
        Col 4 = Q1
        Col 5 = Q3
        Col 6 = D9
        Col 7 = PRUMER

    Indexed entries (4-digit ISCO codes, no leading space):
        2512 — software developers         (POCET=0.450  → 450 persons)
        2211 — medical doctors             (POCET=0.200  → 200 persons)
        2612 — lawyers                     (POCET=0.025  → 25 persons — low sample)
        3112 — civil engineering techs     (POCET=0.150  → 150 persons)
        9112 — elementary cleaners         (POCET=0.300  → 300 persons)

    Skipped entry (5-digit ISCO sub-category, leading space in label):
        " 11201 Subgroup test"             (must be filtered out by loader)

    Returns:
        Path to the generated XLSX file inside tmp_path.
    """
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    assert ws is not None
    ws.title = "MZS-M8r"

    # Rows 0-7: merged-cell title + multi-level header band — any content, skipped by loader.
    for _ in range(8):
        ws.append(["HEADER", None, None, None, None, None, None, None])

    # Data rows: label, POCET (thousands), MEDIAN, D1, Q1, Q3, D9, PRUMER
    ws.append(["2512 Vývojáři softwaru", 0.450, 78000, 45000, 60000, 100000, 130000, 82000])
    ws.append(["2211 Lékaři specialisté", 0.200, 90000, 55000, 72000, 115000, 150000, 95000])
    # 2612: POCET=0.025 → 25 persons — intentionally below _HIGH_CONFIDENCE_MIN_COUNT=30
    ws.append(["2612 Právníci", 0.025, 65000, 40000, 52000, 85000, 110000, 70000])
    ws.append(["3112 Stavební technici", 0.150, 55000, 35000, 44000, 68000, 85000, 58000])
    # 5-digit sub-category row — leading space, must be skipped by loader.
    ws.append([" 11201 Subgroup test", 0.020, 50000, 30000, 40000, 60000, 75000, 52000])
    ws.append(["9112 Pomocníci úklid", 0.300, 30000, 20000, 25000, 36000, 45000, 31000])

    path = tmp_path / "ispv_sample.xlsx"
    wb.save(path)
    return path
