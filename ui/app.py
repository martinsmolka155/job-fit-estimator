"""Streamlit web UI for Job Fit Estimator.

Cache: @st.cache_data receives file_bytes: bytes (from uploaded_file.getvalue()).
NEVER pass the file object itself — its hash is unstable across re-uploads.

HTML safety: all CV-derived and LLM-generated strings are passed through
html.escape() before being interpolated into st.markdown(unsafe_allow_html=True)
templates. CSS class wrappers stay literal in the source.
"""

from __future__ import annotations

import html
import logging
import re
import tempfile
from pathlib import Path
from typing import Any, cast

import httpx
import streamlit as st

from src.pipeline import Pipeline
from src.salary_ispv import (
    ISPVDataMissingError,
    ISPVLookupError,
    MissingISCOError,
    NonCZLocationError,
)
from src.schemas import PipelineResult, SalaryData
from ui.styles import inject_css

ISPV_INDEX_URL = "https://www.ispv.cz/cz/vysledky-setreni/aktualni.aspx"
ISPV_TARGET_PATH = Path("data/ispv_2025.xlsx")

logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Job Fit Estimator",
    page_icon="📄",
    layout="wide",
)
st.markdown(inject_css(), unsafe_allow_html=True)


@st.cache_data
def run_pipeline(file_bytes: bytes, file_name: str, provider: str) -> PipelineResult:
    """Run pipeline on uploaded CV bytes.

    CRITICAL: Takes file_bytes (bytes), NOT the file object.
    Streamlit caches by argument hash — file objects have unstable hashes across re-uploads.
    file.getvalue() returns stable bytes that hash correctly.
    """
    suffix = "." + file_name.rsplit(".", 1)[-1].lower()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = Path(tmp.name)

    try:
        from src.config import settings

        pipeline = Pipeline(
            provider,
            parser_model=settings.parser_model,
            explainer_model=settings.explainer_model,
        )
        return pipeline.run(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def render_sidebar() -> tuple[str, bytes | None, str]:
    """Render sidebar and return (provider, file_bytes, file_name)."""
    st.sidebar.markdown('<div class="jfe-logo">⚡ Job Fit</div>', unsafe_allow_html=True)
    st.sidebar.markdown(
        '<div class="jfe-tagline">AI analýza životopisů</div>', unsafe_allow_html=True
    )

    provider = "openai"

    from src.config import settings

    if not settings.openai_api_key:
        st.sidebar.error("OPENAI_API_KEY není nastaven v .env — pipeline selže za běhu.")

    uploaded_file = st.sidebar.file_uploader(
        "Nahrát CV",
        type=["pdf", "docx"],
        help="Nahrajte digitální PDF nebo DOCX. Skenované PDF nejsou podporovány.",
    )

    st.sidebar.divider()
    _render_ispv_download()
    st.sidebar.markdown(
        '<div style="color:#6B7280;font-size:12px;margin-top:24px;">MVP demo. OCR není podporováno.</div>',
        unsafe_allow_html=True,
    )

    if uploaded_file is not None:
        return provider, uploaded_file.getvalue(), uploaded_file.name  # type: ignore[return-value]

    return provider, None, ""


def _render_ispv_download() -> None:
    """Sidebar button: scrape ispv.cz and download ISPV M8r XLSX."""
    if st.sidebar.button("🔄 Stáhnout ISPV") and _download_ispv():
        st.rerun()

    if ISPV_TARGET_PATH.exists():
        size_kb = ISPV_TARGET_PATH.stat().st_size // 1024
        st.sidebar.caption(f"ISPV data: {size_kb:,} kB")
    else:
        st.sidebar.warning("ISPV data nestažena — pipeline selže.")


def _download_ispv() -> bool:
    """Download ISPV M8r XLSX. Returns True on success (caller can st.rerun)."""
    with st.spinner("Hledám M8r odkaz na ispv.cz..."):
        try:
            r = httpx.get(
                ISPV_INDEX_URL,
                follow_redirects=True,
                timeout=30,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            r.raise_for_status()
        except Exception as e:
            st.sidebar.error(f"Index fetch failed: {e}")
            return False

        # ISPV publishes XLSX as .aspx download endpoints, e.g.
        # /getattachment/<guid>/CR_254_MZS_M8r-xlsx.aspx?disposition=attachment
        # MZS = mzdová sféra (private), PLS = platová sféra (gov). M8r = 4-digit ISCO breakdown.
        links = re.findall(r'href="([^"]+)"', r.text)
        candidates = [
            link
            for link in links
            if "MZS_M8r-xlsx" in link or ("MZS" in link and "M8r" in link and "xlsx" in link)
        ]
        if not candidates:
            st.sidebar.error("M8r XLSX odkaz nenalezen — HTML ispv.cz se mohl změnit.")
            return False

        xlsx_url = candidates[0]
        if not xlsx_url.startswith("http"):
            xlsx_url = "https://www.ispv.cz" + xlsx_url

        try:
            resp = httpx.get(
                xlsx_url,
                follow_redirects=True,
                timeout=60,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            resp.raise_for_status()
            ISPV_TARGET_PATH.parent.mkdir(parents=True, exist_ok=True)
            ISPV_TARGET_PATH.write_bytes(resp.content)
            st.cache_data.clear()
            st.toast(f"ISPV staženo ({len(resp.content):,} B)", icon="✅")
            return True
        except Exception as e:
            st.sidebar.error(f"Download failed: {e}")
            return False


def render_profile_tab(result: PipelineResult) -> None:
    """Render the Profil tab."""
    resume = result.resume

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="jfe-stat-label">Jméno</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div style="font-size:18px;font-weight:600;color:#111827;">{html.escape(resume.full_name or "—")}</div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown('<div class="jfe-stat-label">Lokace</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div style="font-size:18px;font-weight:600;color:#111827;">{html.escape(resume.location or "—")}</div>',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown('<div class="jfe-stat-label">Email</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div style="font-size:18px;font-weight:600;color:#111827;">{html.escape(resume.email or "—")}</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # Experience timeline
    if resume.experiences:
        st.subheader("Praxe")
        for exp in resume.experiences:
            end = str(exp.end_year) if exp.end_year else "Současnost"
            contractor_badge = (
                " <span style='color:#6B7280;font-size:12px;'>[OSVČ/Freelance]</span>"
                if exp.is_contractor
                else ""
            )
            mgmt_badge = (
                " <span style='color:#6B7280;font-size:12px;'>[Management]</span>"
                if exp.is_management
                else ""
            )
            desc_html = (
                f'<div style="color:#6B7280;font-size:13px;margin-top:8px;">{html.escape(exp.description[:300])}</div>'
                if exp.description
                else ""
            )
            role_title_safe = html.escape(exp.role_title or "")
            company_safe = html.escape(exp.company or "")
            end_safe = html.escape(str(end))
            start_safe = html.escape(str(exp.start_year))
            st.markdown(
                f'<div class="jfe-card">'
                f'<div style="display:flex;justify-content:space-between;align-items:baseline;">'
                f'<div><strong style="color:#111827;font-size:16px;">{role_title_safe}</strong>{contractor_badge}{mgmt_badge}<br>'
                f'<span style="color:#6B7280;font-size:14px;">{company_safe}</span></div>'
                f'<span style="color:#6B7280;font-size:13px;">{start_safe}–{end_safe}</span>'
                f"</div>"
                f"{desc_html}"
                f"</div>",
                unsafe_allow_html=True,
            )

    # Education
    if resume.educations:
        st.subheader("Vzdělání")
        for edu in resume.educations:
            year = f", {edu.year_completed}" if edu.year_completed else ""
            field = f" — {edu.field}" if edu.field else ""
            st.write(f"**{edu.degree.title()}**{field} @ {edu.institution or '—'}{year}")

    # Skills as chips
    if resume.skills:
        st.subheader("Dovednosti")
        skills_html = "".join(
            f'<span class="jfe-skill-chip">{html.escape(s.name)}</span>' for s in resume.skills
        )
        st.markdown(f"<div>{skills_html}</div>", unsafe_allow_html=True)

    # Validation flags
    if result.validation_flags:
        st.subheader("Varování")
        for flag in result.validation_flags:
            if flag.severity == "error":
                st.error(f"**{flag.field_path}:** {flag.issue}")
            elif flag.severity == "warning":
                st.warning(f"**{flag.field_path}:** {flag.issue}")
            else:
                st.info(f"**{flag.field_path}:** {flag.issue}")


def _salary_source_badge_html(salary_source: str) -> str:
    """Return HTML pill for the salary data source label.

    Args:
        salary_source: One of ispv / generic_fallback.

    Returns:
        HTML string with an inline-styled pill badge.
    """
    config: dict[str, tuple[str, str]] = {
        "ispv": ("ISPV 2025", "jfe-badge-green"),
        "generic_fallback": ("Obecný odhad", "jfe-badge-amber"),
    }
    label, css_class = config.get(salary_source, (salary_source, "jfe-badge-gray"))
    return f'<span class="{css_class}">{label}</span>'


def _confidence_badge_html(confidence: str) -> str:
    """Return HTML pill for SalaryData confidence level.

    Args:
        confidence: One of high / medium / low.

    Returns:
        HTML string with an inline-styled pill badge.
    """
    config: dict[str, tuple[str, str]] = {
        "high": ("Spolehlivost: vysoká", "jfe-badge-green"),
        "medium": ("Spolehlivost: střední", "jfe-badge-amber"),
        "low": ("Spolehlivost: nízká — viz varování", "jfe-badge-red"),
    }
    label, css_class = config.get(confidence, (confidence, "jfe-badge-gray"))
    return f'<span class="{css_class}">{label}</span>'


def _parse_salary_data(meta: dict[str, Any]) -> SalaryData | None:
    """Parse SalaryData from pipeline meta dict if present.

    Args:
        meta: Pipeline result meta dict.

    Returns:
        SalaryData instance or None when unavailable.
    """
    raw = meta.get("salary_data")
    if not isinstance(raw, dict):
        return None
    try:
        return SalaryData.model_validate(raw)
    except Exception:
        return None


def render_score_tab(result: PipelineResult) -> None:
    """Render the Skóre & Mzda tab."""
    score = result.score
    salary = result.salary

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f'<div class="jfe-card">'
            f'<div class="jfe-stat-label">Skóre seniority</div>'
            f'<div class="jfe-stat-value">{score.total:.1f}<span style="font-size:18px;color:#6B7280;">/100</span></div>'
            f"</div>",
            unsafe_allow_html=True,
        )
        st.progress(score.total / 100)
    with col2:
        confidence_color = (
            "#059669"
            if score.confidence >= 0.8
            else ("#D97706" if score.confidence >= 0.5 else "#DC2626")
        )
        st.markdown(
            f'<div class="jfe-card">'
            f'<div class="jfe-stat-label">Spolehlivost</div>'
            f'<div class="jfe-stat-value" style="color:{confidence_color};">{score.confidence:.0%}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )
    with col3:
        flags_count = len(result.validation_flags)
        flags_color = "#059669" if flags_count == 0 else "#D97706"
        st.markdown(
            f'<div class="jfe-card">'
            f'<div class="jfe-stat-label">Varování</div>'
            f'<div class="jfe-stat-value" style="color:{flags_color};">{flags_count}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )

    st.divider()

    # Score component breakdown
    st.subheader("Rozpis skóre")
    component_data = [
        {
            "Komponenta": c.name.replace("Component", ""),
            "Skóre": f"{c.score:.1f}",
            "Váha": f"{c.weight:.0%}",
            "Příspěvek": f"{c.score * c.weight:.1f}",
            "Odůvodnění": c.reasoning,
        }
        for c in score.components
    ]
    st.dataframe(component_data, use_container_width=True, hide_index=True)

    st.divider()

    # Salary estimate
    st.subheader("Odhad mzdy (Kč/měsíc hrubého)")
    sal_col1, sal_col2, sal_col3 = st.columns(3)
    with sal_col1:
        st.markdown(
            f'<div class="jfe-card">'
            f'<div class="jfe-stat-label">Nízký</div>'
            f'<div class="jfe-stat-value" style="font-size:24px;">{salary.low:,} Kč</div>'
            f"</div>",
            unsafe_allow_html=True,
        )
    with sal_col2:
        st.markdown(
            f'<div class="jfe-card" style="background:#EFF6FF;border-color:#2563EB;">'
            f'<div class="jfe-stat-label">Střední (nejpravděpodobnější)</div>'
            f'<div class="jfe-stat-value" style="color:#2563EB;">{salary.mid:,} Kč</div>'
            f"</div>",
            unsafe_allow_html=True,
        )
    with sal_col3:
        st.markdown(
            f'<div class="jfe-card">'
            f'<div class="jfe-stat-label">Vysoký</div>'
            f'<div class="jfe-stat-value" style="font-size:24px;">{salary.high:,} Kč</div>'
            f"</div>",
            unsafe_allow_html=True,
        )

    # ── Salary source badge + confidence + ISPV freshness disclosure ────────
    # PipelineResult.meta is typed as bare dict (no type params) — cast to typed for pyright.
    meta_typed: dict[str, Any] = cast("dict[str, Any]", result.meta)  # type: ignore[redundant-cast]
    salary_source: str = str(meta_typed.get("salary_source", "generic_fallback"))
    salary_data_obj = _parse_salary_data(meta_typed)

    badge_html = _salary_source_badge_html(salary_source)
    if salary_data_obj is not None:
        badge_html += " " + _confidence_badge_html(salary_data_obj.confidence)

    st.markdown(
        f'<div style="margin-top:16px;margin-bottom:8px;">{badge_html}</div>',
        unsafe_allow_html=True,
    )

    # ISPV freshness disclosure card — dataset age, inflation correction, match level.
    if salary_data_obj is not None:
        isco = html.escape(salary_data_obj.isco_code or "N/A")
        match_level = html.escape(salary_data_obj.isco_match_level)
        age = salary_data_obj.dataset_age_months
        factor = salary_data_obj.inflation_factor_applied
        conf = html.escape(salary_data_obj.confidence)

        badge_html_detail = f"""
        <div class="jfe-card" style="background:#F9FAFB;border-left:3px solid #6B7280;">
        <div style="font-size:12px;color:#6B7280;font-weight:600;">Zdroj mzdových dat</div>
        <div style="font-size:13px;color:#374151;margin-top:4px;">
            ISPV M8r 2025 ({match_level} ISCO {isco}) — <strong>HPP gross měsíčně</strong><br>
            Stáří dat: {age} měsíců → inflační korekce × {factor:.3f}<br>
            Spolehlivost: {conf}
        </div>
        <div style="font-size:11px;color:#9CA3AF;margin-top:8px;font-style:italic;">
            Estimate je HPP zaměstnanecká mzda. IČO/freelance ekvivalent ~× 1.4–1.6
            (zahrnuje sociální/zdravotní + neplacené volno + nemoc).
            Sektor firmy (banking/FAANG/agency/gov) může posunout o ±15 % — model neuvažuje.
        </div>
        </div>
        """
        st.markdown(badge_html_detail, unsafe_allow_html=True)

        # Warnings from SalaryData (low sample count, imprecise ISCO match, etc.)
        if salary_data_obj.warnings:
            with st.expander("Varování ke mzdovým datům"):
                for w in salary_data_obj.warnings:
                    st.warning(w)

    st.markdown(
        f'<div style="color:#6B7280;font-size:13px;margin-top:16px;">Odůvodnění: {salary.reasoning}</div>',
        unsafe_allow_html=True,
    )
    with st.expander("Předpoklady"):
        for assumption in salary.assumptions:
            st.caption(f"• {assumption}")


def render_recommendations_tab(result: PipelineResult) -> None:
    """Render the Doporučení tab."""
    explanation = result.explanation

    st.subheader("Shrnutí")
    st.write(explanation.summary)

    col_str, col_gap = st.columns(2)
    with col_str:
        st.subheader("Silné stránky")
        for s in explanation.strengths:
            st.markdown(
                f'<div class="jfe-card jfe-card-strength">'
                f'<svg width="16" height="16" viewBox="0 0 24 24" fill="none" style="vertical-align:middle;margin-right:8px;">'
                f'<path d="M5 13l4 4L19 7" stroke="#059669" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>'
                f"</svg>{html.escape(s)}"
                f"</div>",
                unsafe_allow_html=True,
            )

    with col_gap:
        st.subheader("Slabiny")
        for g in explanation.gaps:
            st.markdown(
                f'<div class="jfe-card jfe-card-gap">'
                f'<svg width="16" height="16" viewBox="0 0 24 24" fill="none" style="vertical-align:middle;margin-right:8px;">'
                f'<path d="M12 9v2m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" stroke="#D97706" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>'
                f"</svg>{html.escape(g)}"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.divider()
    st.subheader("Doporučení")
    for i, rec in enumerate(explanation.recommendations, 1):
        with st.expander(
            f"{i}. {rec.title} (+{rec.estimated_salary_impact_pct:.0f} %, {rec.timeframe_months} měs.)"
        ):
            st.markdown(
                f'<div style="margin-bottom:12px;">'
                f'<strong style="color:#0F172A;">Proč je to důležité:</strong> '
                f'<span style="color:#374151;">{html.escape(rec.why_it_matters)}</span>'
                f"</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div style="margin-bottom:12px;">'
                f'<strong style="color:#0F172A;">První krok:</strong> '
                f'<span style="color:#374151;">{html.escape(rec.first_action)}</span>'
                f"</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="jfe-stat-label">Odhadovaný dopad na mzdu: +{rec.estimated_salary_impact_pct:.0f}%</div>',
                unsafe_allow_html=True,
            )


def main() -> None:
    """Main Streamlit app entry point."""
    provider, file_bytes, file_name = render_sidebar()

    # Single Streamlit slot reused for welcome → spinner → results.
    # Explicit .empty() clear prevents Streamlit's stale-frame ghost overlay
    # during @st.cache_data computations (FINDING 1 from Phase 16 self-audit).
    welcome_slot = st.empty()

    if file_bytes is None:
        # Welcome screen — centered, no spinners, no bare Streamlit widgets
        welcome_slot.markdown(
            '<div class="jfe-welcome">'
            '<div class="jfe-welcome-headline">Job Fit &amp; Salary Estimator</div>'
            '<div class="jfe-welcome-subhead">'
            "Nahrajte CV (PDF nebo DOCX) v postranním panelu pro získání skóre seniority, "
            "odhadu mzdy a 3 doporučení pro kariérní růst."
            "</div>"
            '<div class="jfe-welcome-note">'
            "<strong>Omezení:</strong> Skenované PDF nejsou podporovány. "
            "Platové rozsahy jsou ručně kurátorovaná demo data, ne živá tržní data."
            "</div>"
            "</div>",
            unsafe_allow_html=True,
        )
        return

    # CV uploaded — clear welcome slot before spinner so Streamlit has nothing
    # to replay as ghost overlay during cache miss.
    welcome_slot.empty()

    # Spinner first, then results (never both visible simultaneously)
    try:
        with st.spinner(f"Analyzuji {file_name}..."):
            result = run_pipeline(file_bytes, file_name, provider)
    except NonCZLocationError as e:
        st.warning(f"⚠️ {e}")
        st.info(
            "ISPV data pokrývají jen český trh. Nahraj CV s lokací v CZ "
            "(libovolné město), nebo lokaci v CV uprav."
        )
        st.stop()
    except ISPVDataMissingError as e:
        st.warning(f"⚠️ {e}")
        st.info(
            "Klikni na **🔄 Stáhnout ISPV** v levém panelu — stáhne se "
            "`data/ispv_2025.xlsx` (~150 kB) z ispv.cz. Po stažení nahraj CV znovu."
        )
        st.stop()
    except MissingISCOError as e:
        st.warning(f"⚠️ {e}")
        st.info(
            "Parser ze CV nedokázal určit profesi (CZ-ISCO-08 kód). "
            "Možné příčiny: CV bez jasných pracovních titulů, příliš obecné popisy rolí, "
            "neobvyklé/ezoterické povolání. Můžeš zkusit CV upravit nebo nahrát jiné."
        )
        st.stop()
    except ISPVLookupError as e:
        st.error(f"Salary estimate selhal: {e}")
        st.stop()
    except ValueError as e:
        st.error(f"Pipeline selhal: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Neočekávaná chyba: {e}")
        logger.exception("Pipeline failed for %s", file_name)
        st.stop()

    # Display results in 3 tabs
    st.markdown(
        '<p style="color:#6B7280;font-size:13px;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:0;">Výsledky analýzy</p>',
        unsafe_allow_html=True,
    )
    display_name = html.escape(result.resume.full_name or "Unknown")
    st.markdown(
        f'<h1 style="font-size:36px;font-weight:600;color:#0F172A;margin-top:0;margin-bottom:24px;">{display_name}</h1>',
        unsafe_allow_html=True,
    )

    tab_profile, tab_score, tab_recs = st.tabs(["Profil", "Skóre & Mzda", "Doporučení"])

    with tab_profile:
        render_profile_tab(result)

    with tab_score:
        render_score_tab(result)

    with tab_recs:
        render_recommendations_tab(result)

    # Footer
    cost = result.meta.get("total_cost_usd", 0)
    duration = result.meta.get("total_duration_s", 0)
    model = result.meta.get("model", "n/a")
    st.markdown(
        f'<div class="jfe-footer">'
        f'<span class="jfe-stat-label">Náklady:</span> '
        f'<span class="jfe-footer-mono">${cost:.4f}</span>'
        f"&nbsp;&nbsp;&nbsp;"
        f'<span class="jfe-stat-label">Trvání:</span> '
        f'<span class="jfe-footer-mono">{duration:.1f}s</span>'
        f"&nbsp;&nbsp;&nbsp;"
        f'<span class="jfe-stat-label">Model:</span> '
        f'<span class="jfe-footer-mono">{model}</span>'
        f"</div>",
        unsafe_allow_html=True,
    )


main()
