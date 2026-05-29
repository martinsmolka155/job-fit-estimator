"""Tests for PDF/DOCX text extractor.

Fixtures are generated programmatically — no LLM calls.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from docx import Document
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from src.extractor import UnsupportedFormatError, extract_text


def _create_digital_pdf(path: Path, content: str = "") -> None:
    """Create a minimal digital PDF with readable text using reportlab."""
    if not content:
        content = (
            "Jan Novák\n"
            "junior@example.cz | +420 123 456 789 | Praha\n\n"
            "EXPERIENCE\n"
            "Software Developer at Acme Corp, Praha, 2024 - Present\n"
            "Implemented REST APIs in Python, worked with PostgreSQL\n\n"
            "EDUCATION\n"
            "BSc Computer Science, CVUT FIT, 2024\n\n"
            "SKILLS\n"
            "Python, SQL, Git, Docker, FastAPI\n"
        )
    c = canvas.Canvas(str(path), pagesize=A4)
    text_object = c.beginText(50, 750)
    text_object.setFont("Helvetica", 11)
    for line in content.split("\n"):
        text_object.textLine(line)
    c.drawText(text_object)
    c.save()


def _create_sparse_pdf(path: Path) -> None:
    """Create a PDF with very little text to simulate scanned document detection."""
    c = canvas.Canvas(str(path), pagesize=A4)
    # Only a few characters — below MIN_DIGITAL_CHARS threshold
    c.setFont("Helvetica", 11)
    c.drawString(50, 750, "ab")
    c.save()


def _create_docx(path: Path, content: str = "") -> None:
    """Create a minimal DOCX file with python-docx."""
    if not content:
        content = (
            "Jan Novák\n"
            "junior@example.cz | Praha\n\n"
            "Software Developer at Acme Corp 2024 - Present\n"
            "BSc Computer Science CVUT FIT 2024\n"
            "Skills: Python, SQL, Git, Docker"
        )
    doc = Document()
    for line in content.split("\n"):
        doc.add_paragraph(line)
    doc.save(str(path))


class TestExtractPDF:
    def test_digital_pdf_returns_text(self) -> None:
        """A normal digital PDF should extract readable text."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            _create_digital_pdf(tmp_path)
            result = extract_text(tmp_path)

            assert result.source_format == "pdf"
            assert result.extraction_method == "pymupdf"
            assert result.is_scanned is False
            assert result.char_count > 0
            assert len(result.text) >= 200
            # Verify we got actual content
            assert "Jan" in result.text or "Novák" in result.text or "Python" in result.text
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_sparse_pdf_returns_is_scanned_true(self) -> None:
        """A PDF with very little text (simulating scanned) should return is_scanned=True."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            _create_sparse_pdf(tmp_path)
            result = extract_text(tmp_path)

            assert result.is_scanned is True
            assert result.extraction_method == "scanned_unsupported"
            assert result.text == ""
            assert result.char_count == 0
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_minimal_legit_cv_is_not_flagged_as_scanned(self) -> None:
        """A short but legitimate one-page CV (~80 chars) must NOT be flagged as scanned.

        Regression for the previous _MIN_DIGITAL_CHARS=200 threshold which
        incorrectly rejected sparse digital PDFs (e.g. a junior CV with
        name + contact + one role).
        """
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            # Approximately 80 readable characters — enough to identify a candidate,
            # would have been rejected by the previous 200-char threshold.
            _create_digital_pdf(
                tmp_path,
                content="Jan Novák\nPython developer\nPraha, 2024-Present\nSkills: Python, SQL\n",
            )
            result = extract_text(tmp_path)

            assert result.is_scanned is False, (
                f"Sparse but legitimate CV must not be flagged as scanned "
                f"(extracted {result.char_count} chars)"
            )
            assert result.extraction_method == "pymupdf"
            assert result.char_count >= 30
        finally:
            tmp_path.unlink(missing_ok=True)


class TestExtractDOCX:
    def test_docx_returns_text(self) -> None:
        """A DOCX file should extract readable text."""
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            _create_docx(tmp_path)
            result = extract_text(tmp_path)

            assert result.source_format == "docx"
            assert result.extraction_method == "docx"
            assert result.is_scanned is False
            assert result.char_count > 0
            assert "Jan" in result.text or "Python" in result.text
        finally:
            tmp_path.unlink(missing_ok=True)


class TestUnsupportedFormat:
    def test_txt_raises_unsupported_format_error(self) -> None:
        """Non-PDF, non-DOCX files must raise UnsupportedFormatError."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            tmp.write(b"some content")
            tmp_path = Path(tmp.name)

        try:
            with pytest.raises(UnsupportedFormatError, match="Unsupported file format"):
                extract_text(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_csv_raises_unsupported_format_error(self) -> None:
        """CSV files must raise UnsupportedFormatError."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp.write(b"name,email\nJan,jan@example.cz")
            tmp_path = Path(tmp.name)

        try:
            with pytest.raises(UnsupportedFormatError):
                extract_text(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)


class TestIsGibberish:
    def test_gibberish_detection(self) -> None:
        """_is_gibberish should detect text with >30% non-alphanumeric chars."""
        from src.extractor import _is_gibberish

        # Clean text — should not be gibberish
        assert _is_gibberish("Hello World Python Developer 2024") is False

        # Mostly symbols — should be gibberish
        assert _is_gibberish("!@#$%^&*()!@#$%^&*()abc") is True

        # Empty string — gibberish
        assert _is_gibberish("") is True


class TestPageLimitGuard:
    def test_oversized_pdf_raises_value_error(self) -> None:
        """PDFs exceeding _MAX_PDF_PAGES must raise ValueError (maps to HTTP 422)."""
        from src.extractor import _MAX_PDF_PAGES

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            # Build a PDF with _MAX_PDF_PAGES + 1 pages.
            c = canvas.Canvas(str(tmp_path), pagesize=A4)
            for i in range(_MAX_PDF_PAGES + 1):
                c.setFont("Helvetica", 11)
                c.drawString(50, 750, f"Page {i + 1} — Jan Novák Python developer Praha")
                c.showPage()
            c.save()

            with pytest.raises(ValueError, match="pages"):
                extract_text(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_pdf_at_page_limit_is_accepted(self) -> None:
        """A PDF with exactly _MAX_PDF_PAGES pages must NOT be rejected."""
        from src.extractor import _MAX_PDF_PAGES

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            c = canvas.Canvas(str(tmp_path), pagesize=A4)
            for i in range(_MAX_PDF_PAGES):
                c.setFont("Helvetica", 11)
                c.drawString(50, 750, f"Page {i + 1} — Jan Novák Python developer Praha")
                c.showPage()
            c.save()

            result = extract_text(tmp_path)
            # Should not raise; must extract text
            assert result.is_scanned is False
            assert result.page_count == _MAX_PDF_PAGES
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_corrupt_pdf_returns_scanned_sentinel(self) -> None:
        """A corrupt (unreadable) PDF must return is_scanned=True, not crash."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(b"this is not a valid PDF file")
            tmp_path = Path(tmp.name)

        try:
            result = extract_text(tmp_path)
            assert result.is_scanned is True
            assert result.text == ""
            assert result.extraction_method == "scanned_unsupported"
        finally:
            tmp_path.unlink(missing_ok=True)
