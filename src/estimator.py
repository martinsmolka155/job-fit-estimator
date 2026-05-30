"""Salary estimator — ISPV-backed implementation.

Produces SalaryEstimate (CZK range) from Resume + SeniorityScore using
ISPV M8r 2025 dataset via ISPVLoader. Hard error if ISPV data is not loaded
or no ISCO code is present — no silent fallback per project decision Q4 2026.

Salary-band selection is driven by ROLE seniority (parsed title + years of
experience), NOT by the composite SeniorityScore.total. The composite score
is a blend of experience + skills + education + progression + domain; education
and skill breadth must not push a genuine senior into the lead salary band.
See infer_salary_seniority() for the decoupled logic.
"""

from __future__ import annotations

import logging
import re
from typing import Literal

from src.salary_ispv import (
    ISPVDataMissingError,
    ISPVLoader,
    ISPVLookupError,
    MissingISCOError,
    NonCZLocationError,
)
from src.schemas import Experience, Resume, SalaryBand, SalaryData, SalaryEstimate, SeniorityScore

logger = logging.getLogger(__name__)

# Seniority band type alias — mirrors Experience.seniority_level literal.
SeniorityBand = Literal["junior", "medior", "senior", "lead", "principal"]

# Location multipliers applied on top of ISPV national bands.
# ISPV M8r is country-wide aggregate; Praha pays ~24% above national median,
# regional CZ pays 10-25% below. Praha as primary uplift, Brno near national,
# regional significantly below. Keys are lowercase substrings matched against location.
#
# TODO(review): Praha multiplier calibration needed.
# Current value 1.15 is a round estimate.  The ISPV M8r dataset itself publishes
# KRAJ-level breakdowns — the Praha multiplier should be derived by dividing
# Praha median (ISPV KRAJ=CZ010) by the national median for the same ISCO group,
# then averaged across the top 5 IT ISCO codes (2511, 2512, 2519, 2522, 2523).
# Until that calculation is done against the real XLSX, 1.15 is a documented guess.
# Suggested owner: whoever owns ISPVLoader.load() — add a _load_kraj_multipliers()
# method that reads KRAJ rows and returns a dict[str, float] for use here.
_LOCATION_MULTIPLIERS: dict[str, tuple[float, str]] = {
    "praha": (1.15, "Praha multiplier"),
    "prague": (1.15, "Praha multiplier"),
    "brno": (1.00, "Brno multiplier"),  # Brno IT ~10% below Praha, near national
    "ostrava": (0.85, "Ostrava regional multiplier"),
    "olomouc": (0.85, "Olomouc regional multiplier"),
    "zlín": (0.85, "Zlín regional multiplier"),
    "zlin": (0.85, "Zlín regional multiplier"),
    "liberec": (0.85, "Liberec regional multiplier"),
    "plzeň": (0.88, "Plzeň regional multiplier"),
    "plzen": (0.88, "Plzeň regional multiplier"),
    "pardubice": (0.85, "Pardubice regional multiplier"),
    "hradec králové": (0.85, "Hradec Králové regional multiplier"),
    "hradec kralove": (0.85, "Hradec Králové regional multiplier"),
    "české budějovice": (0.85, "České Budějovice regional multiplier"),
    "ceske budejovice": (0.85, "České Budějovice regional multiplier"),
    "jihlava": (0.80, "Jihlava regional multiplier"),
    "karviná": (0.78, "Karviná regional multiplier"),
    "karvina": (0.78, "Karviná regional multiplier"),
    "most": (0.78, "Most regional multiplier"),
    "teplice": (0.80, "Teplice regional multiplier"),
}

# Non-CZ country/city / non-CZ-employer remote indicators. Word-boundary
# regex so substrings like "cz" don't false-match (e.g. "Szczecin"). If any
# token matches, location is treated as non-CZ even when "remote" or "czech"
# also appears (e.g. "Remote US", "Czech expat in Berlin").
_NON_CZ_COUNTRIES_RE = re.compile(
    r"\b("
    # Germany
    r"germany|german|deutschland|berlin|hamburg|münchen|munchen|munich|frankfurt|cologne|"
    # Poland
    r"poland|polish|polska|warsaw|warszawa|szczecin|kraków|krakow|wrocław|wroclaw|"
    # Slovakia
    r"slovakia|slovak|slovensko|bratislava|košice|kosice|"
    # Austria
    r"austria|österreich|vienna|wien|salzburg|graz|"
    # UK / Ireland
    r"uk|united kingdom|england|britain|london|manchester|edinburgh|ireland|dublin|"
    # USA
    r"usa|us|u\.s\.a|united states|america|new york|san francisco|los angeles|chicago|"
    # Other EU
    r"france|paris|spain|españa|madrid|barcelona|italy|italia|rome|roma|milano|"
    r"netherlands|nederland|amsterdam|rotterdam|"
    r"belgium|brussels|switzerland|zürich|zurich|"
    r"sweden|stockholm|denmark|copenhagen|norway|oslo|finland|helsinki|"
    r"hungary|budapest|romania|bucharest|"
    # Asia/elsewhere
    r"india|china|japan|tokyo|singapore|dubai|israel|tel aviv"
    r")\b",
    re.IGNORECASE,
)


def _score_to_seniority(total: float) -> str:
    """Map SeniorityScore.total (0-100) to a DISPLAY seniority label.

    This label is shown in the UI and stored in assumptions for transparency.
    It is NOT used for salary band selection — see infer_salary_seniority().

    # TODO(review): display-seniority thresholds need empirical calibration.
    # Current breakpoints (junior<30, medior<60, senior<80, lead<92, principal>=92)
    # were chosen by intuition.  To calibrate properly:
    # 1. Collect a ground-truth set of labelled CVs with known seniority levels
    #    (minimum 50 CVs per band, ideally from hiring data or salary surveys).
    # 2. Run the scorer on all CVs, plot the score distribution per known band.
    # 3. Pick breakpoints that minimise band misclassification on that labelled set.
    # Until then, treat these thresholds as provisional and do not rely on them for
    # fine-grained salary precision claims.
    """
    if total < 30:
        return "junior"
    elif total < 60:
        return "medior"
    elif total < 80:
        return "senior"
    elif total < 92:
        return "lead"
    else:
        return "principal"


def _primary_experience(resume: Resume) -> Experience | None:
    """Return the candidate's primary / most-recent substantive experience entry.

    Uses the same recency key as estimate() for ISCO selection:
    - ongoing roles (end_year=None) first,
    - then by end_year descending,
    - then longer-running tenure wins the tie-break (lower start_year).
    """
    if not resume.experiences:
        return None

    def _recency_key(exp: Experience) -> tuple[int, int, int]:
        end = exp.end_year
        start = exp.start_year or 0
        is_current = 1 if end is None else 0
        return (is_current, end or 0, -start)

    return sorted(resume.experiences, key=_recency_key, reverse=True)[0]


def _years_of_relevant_experience(resume: Resume) -> int:
    """Compute de-duplicated total years of experience across all entries.

    Overlapping intervals are merged so parallel roles don't double-count.
    """
    from datetime import UTC, datetime

    current = datetime.now(UTC).year
    intervals: list[tuple[int, int]] = sorted(
        (exp.start_year, exp.end_year or current)
        for exp in resume.experiences
        if (exp.end_year or current) > exp.start_year
    )
    merged: list[tuple[int, int]] = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return max(0, sum(end - start for start, end in merged))


def infer_salary_seniority(resume: Resume) -> tuple[SeniorityBand, bool]:
    """Derive the salary band from ROLE-grounded signals only.

    Education, skill breadth, and composite scores must NOT influence this.
    The goal is conservative accuracy: when in doubt, pick the LOWER band to
    avoid over-estimating salary (public-facing tool).

    Strategy (in priority order):
    1. Explicit seniority_level on the primary/most-recent role — the strongest
       signal because it comes directly from the candidate's job title.
    2. Any explicit lead/principal seniority_level anywhere in career history
       combined with management signals — they held the role, they get the band.
    3. Fallback: infer from total years of relevant experience + management flag.

    Heuristic year thresholds (conservative, bias toward lower band on tie):
      < 2y  → junior
      2–5y  → medior  (inclusive of 2, exclusive of 5)
      5–9y  → senior  (inclusive of 5, exclusive of 9)
      9y+   → senior  (default unless explicit lead/management signals)
      lead  → only with explicit lead/principal seniority_level OR is_management=True

    Returns:
        (band, inferred) — inferred=True means we fell back to heuristics;
        the caller should surface a warning to the UI so the user knows confidence
        is lower.

    # TODO(review): year thresholds (<2 junior, 2–5 medior, 5–9 senior, 9+ senior/lead)
    # are heuristic placeholders. Calibrate against the labeled eval set once N >= 50
    # per band. Derive thresholds from the precision-recall curve for each boundary.
    """
    primary = _primary_experience(resume)
    total_years = _years_of_relevant_experience(resume)
    has_management = any(exp.is_management for exp in resume.experiences)

    level_order: dict[str, int] = {
        "junior": 0,
        "medior": 1,
        "senior": 2,
        "lead": 3,
        "principal": 4,
    }

    # Years-based band — a concrete tenure signal. Capped at 'senior': long tenure
    # alone never implies lead/principal (those need an explicit title or a
    # management signal). Education and skills are deliberately excluded.
    if total_years < 2:
        years_band: SeniorityBand = "junior"
    elif total_years < 5:
        years_band = "medior"
    else:
        years_band = "lead" if (total_years >= 9 and has_management) else "senior"

    # --- Signal 1: explicit seniority_level on the primary role ---
    if primary is not None and primary.seniority_level is not None:
        level: SeniorityBand = primary.seniority_level  # type: ignore[assignment]
        # 'medior' is the parser's DEFAULT for titles without an explicit level
        # (e.g. a plain "Developer"), so it is a weak signal — let concrete years
        # of experience LIFT it (a 10y dev is not a medior). Explicit junior and
        # senior/lead/principal are real signals from the title, trusted as-is.
        if level == "medior" and level_order[years_band] > level_order["medior"]:
            return years_band, True
        return level, False

    # --- Signal 2: explicit lead/principal anywhere in career + management ---
    # Candidate's most recent role title didn't carry a level, but earlier roles did.
    best_level_anywhere: SeniorityBand | None = None
    for exp in resume.experiences:
        if exp.seniority_level is not None:
            current_rank = level_order.get(exp.seniority_level, -1)
            best_rank = level_order.get(best_level_anywhere or "junior", 0)
            if best_level_anywhere is None or current_rank > best_rank:
                best_level_anywhere = exp.seniority_level  # type: ignore[assignment]

    if best_level_anywhere in ("lead", "principal"):
        # They explicitly held a lead/principal role somewhere — honor it.
        return best_level_anywhere, False  # type: ignore[return-value]

    if best_level_anywhere in ("senior", "medior", "junior"):
        # We have a partial signal — combine with years for better confidence.
        # At this point we KNOW the primary role has no label, so use years as
        # corroboration: if best explicit label is "senior" and years >= 5, trust it.
        if best_level_anywhere == "senior" and total_years >= 5:
            return "senior", False
        if best_level_anywhere == "medior" and total_years >= 2:
            return "medior", False
        if best_level_anywhere == "junior" and total_years < 5:
            return "junior", False
        # Fall through to heuristic if mismatch (e.g. "junior" title but 10+ years).

    # --- Signal 3: heuristic from years + management flag ---
    # years_band (computed above) already encodes the conservative mapping:
    # senior for long careers, lead only with an explicit management signal.
    # Education/skills must NOT trigger promotion (root cause of the original bug).
    return years_band, True  # inferred — surface as low-confidence warning


class SalaryEstimator:
    """Estimates monthly gross salary (CZK) from Resume + SeniorityScore.

    Requires a loaded ISPVLoader instance. Raises ISPVLookupError if ISPV data
    is unavailable or the resume contains no ISCO code — hard fail, no silent fallback.
    """

    def __init__(self, ispv_loader: ISPVLoader | None = None) -> None:
        self._loader = ispv_loader
        self._last_salary_source: str = "ispv" if ispv_loader else "generic_fallback"
        self._last_salary_data: SalaryData | None = None

    @property
    def last_salary_source(self) -> str:
        """Label for the data source used in the most recent estimate() call."""
        return self._last_salary_source

    @property
    def last_salary_data(self) -> SalaryData | None:
        """SalaryData from the most recent estimate() call, or None."""
        return self._last_salary_data

    def _is_non_cz_location(self, resume: Resume) -> bool:
        """Return True only when an explicit non-CZ country indicator is present.

        Default behaviour is to assume CZ — the small CZ-city whitelist used to
        be required for "pass" classification and rejected legitimate cities
        like Tábor, Karlovy Vary, or Mladá Boleslav. We now block only when a
        recognised non-CZ country/city token is found (Berlin, London, Bratislava,
        Remote US, etc.). Empty location is treated as CZ.
        """
        location = (resume.location or "").strip()
        if not location:
            return False
        return bool(_NON_CZ_COUNTRIES_RE.search(location))

    def _detect_location_multiplier(self, resume: Resume) -> float:
        """Return a location multiplier based on resume.location string.

        Only CZ-city multipliers are honored. "Remote EU"/"Remote US" used
        to receive uplift here, but those locations now hard-fail at
        _is_non_cz_location() upstream — the README contract is CZ-only.
        Plain "remote" without a country indicator is treated as CZ employer
        and gets the default ×1.0.

        Matching uses word-boundary regex so short keys like "most" do not
        false-positive on "mostly remote" or "almost there".  Keys with
        spaces (e.g. "hradec králové") match the entire phrase.
        """
        location = (resume.location or "").lower()
        for keyword, (mult, _label) in _LOCATION_MULTIPLIERS.items():
            # Build a word-boundary pattern for each keyword.
            # For multi-word keys like "hradec králové" we anchor the whole phrase.
            pattern = r"(?<![^\W_])" + re.escape(keyword) + r"(?![^\W_])"
            if re.search(pattern, location):
                return mult
        return 1.0

    def _has_management(self, resume: Resume) -> bool:
        """Return True if any experience has is_management=True."""
        return any(exp.is_management for exp in resume.experiences)

    def estimate(self, resume: Resume, score: SeniorityScore) -> SalaryEstimate:
        """Produce a salary estimate backed by ISPV M8r 2025 data.

        Salary band is selected via infer_salary_seniority(resume), which reads
        role-level signals (parsed seniority_level, years of experience, management
        flag) — NOT from score.total. score.total is a composite that blends in
        education and skill breadth; those must not influence the salary band.

        Args:
            resume: Structured resume with ISCO codes on experience entries.
            score:  Seniority score — used for DISPLAY only (composite total shown
                    in assumptions); not used for band selection.

        Returns:
            SalaryEstimate with low/mid/high CZK values.

        Raises:
            ISPVLookupError: If ISPV data is not loaded, no ISCO code found,
                             or ISCO lookup fails at all fallback levels.
        """
        # Role-grounded seniority for salary band selection (decoupled from composite score).
        seniority, seniority_inferred = infer_salary_seniority(resume)
        # Display seniority from composite score — shown in UI / logs but not used for band.
        display_seniority = _score_to_seniority(score.total)
        if seniority_inferred:
            logger.warning(
                "Salary seniority inferred from experience years / management signal "
                "(no explicit seniority_level on primary role). "
                "Band=%s  composite_display=%s  score.total=%.1f",
                seniority,
                display_seniority,
                score.total,
            )

        if self._is_non_cz_location(resume):
            raise NonCZLocationError(
                f"Salary estimate je dostupný pouze pro CZ trh. "
                f"Detekovaná lokace: {resume.location!r}. "
                "ISPV data pokrývají pouze českou mzdovou sféru."
            )

        # Pick ISCO from the candidate's primary current role.
        # Sort key: ongoing role first (end_year=None), then by end_year desc,
        # then prefer the *longer*-running tenure on tie-break — a 6-year main
        # job should outrank a 1-year side gig that started later. Negating
        # start_year flips the tie-break inside the descending sort.
        isco_code: str | None = None

        def _recency_key(exp: object) -> tuple[int, int, int]:
            end = getattr(exp, "end_year", None)
            start = getattr(exp, "start_year", 0) or 0
            is_current = 1 if end is None else 0
            return (is_current, end or 0, -start)

        for exp in sorted(resume.experiences, key=_recency_key, reverse=True):
            if exp.isco_code:
                isco_code = exp.isco_code
                break

        if self._loader is None or not self._loader.is_loaded():
            raise ISPVDataMissingError(
                "ISPV data nejsou načtená. Stáhni dataset přes tlačítko v sidebaru."
            )
        if not isco_code:
            raise MissingISCOError(
                "Z CV se nepodařilo určit ISCO kód žádné role — parser nedetekoval profesi."
            )

        # Whitelist check — if isco_code is absent from ISPV index, first try
        # ISPVLoader.lookup()'s built-in 3-digit→2-digit→1-digit prefix fallback
        # chain (closest occupation), only then fall back to the broad family heuristic.
        known = self._loader.known_codes()
        if isco_code not in known:
            logger.warning(
                "ISCO %s not in ISPV whitelist (%d known codes) — "
                "attempting ISPVLoader prefix fallback chain first",
                isco_code,
                len(known),
            )
            # Phase 1: try ISPVLoader.lookup() which walks 3→2→1 prefix chain.
            try:
                salary_data = self._loader.lookup(isco_code)
                match_level_note = salary_data.isco_match_level
                logger.info(
                    "ISCO %s resolved via prefix fallback to %s (match_level=%s)",
                    isco_code,
                    salary_data.isco_code,
                    match_level_note,
                )
                if match_level_note not in ("4digit", "3digit"):
                    # Warn when precision is low — 2-digit or 1-digit matches are broad.
                    logger.warning(
                        "ISCO %s: coarse prefix match (%s) — salary estimate may be imprecise",
                        isco_code,
                        match_level_note,
                    )
            except ISPVLookupError:
                # Phase 2: prefix chain exhausted — try broad occupation_family fallback.
                logger.warning(
                    "ISCO %s: prefix fallback chain failed — falling back via occupation_family",
                    isco_code,
                )
                family: str | None = None
                for exp in sorted(resume.experiences, key=_recency_key, reverse=True):
                    if exp.isco_code == isco_code and exp.occupation_family:
                        family = exp.occupation_family
                        break
                if family is None:
                    raise ISPVLookupError(
                        f"ISCO {isco_code!r} not in ISPV dataset and no occupation_family available for fallback."
                    ) from None
                fallback_code = self._loader.best_code_for_family(family)
                if fallback_code is None:
                    raise ISPVLookupError(
                        f"No ISPV data for occupation_family {family!r}."
                    ) from None
                salary_data = self._loader.lookup(fallback_code)
        else:
            try:
                salary_data = self._loader.lookup(isco_code)
            except ISPVLookupError:
                logger.exception("ISPV lookup failed for ISCO %s", isco_code)
                raise  # hard error — no silent fallback

        self._last_salary_data = salary_data
        self._last_salary_source = "ispv"

        band: SalaryBand | None = getattr(salary_data, seniority, None)
        if band is None:
            # Paranoid fallback within the resolved salary_data object itself.
            band = salary_data.medior
        if band is None:
            # ISPVLoader always populates at least medior — if somehow still None, hard fail.
            raise ISPVLookupError(
                f"No salary band data available for ISCO {isco_code} at seniority {seniority!r}"
            )

        location_mult = self._detect_location_multiplier(resume)
        # 1.10 fits team lead roles (the bulk of is_management=True cases) without
        # overestimating full engineering managers / directors. Tier-based multiplier
        # (lead vs manager vs director) is a known backlog item.
        #
        # IMPORTANT: skip the multiplier when seniority is already lead/principal —
        # those bands extrapolate above ISPV D9 (× 1.10 / 1.20 / 1.35 and × 1.30 /
        # 1.50 / 1.75 respectively), which already absorbs the management upside.
        # Stacking ×1.10 on top double-counts and pushes a CEO CV above 600 k CZK
        # for the band's high — see eval golden-set principal overshoot.
        is_top_band = seniority in ("lead", "principal")
        management_mult = 1.10 if (self._has_management(resume) and not is_top_band) else 1.0

        low = round(band.low * location_mult * management_mult / 1000) * 1000
        mid = round(band.mid * location_mult * management_mult / 1000) * 1000
        high = round(band.high * location_mult * management_mult / 1000) * 1000

        management_note = (
            f"mgmt × {management_mult:.2f}"
            if not is_top_band
            else f"mgmt × 1.00 (already in {seniority} band extrapolation)"
        )
        seniority_source = "inferred (years+mgmt)" if seniority_inferred else "role title"
        reasoning = (
            f"Zdroj: ISPV M8r 2025 (ISCO {salary_data.isco_code}, "
            f"{salary_data.isco_match_level}) | {seniority} band [{seniority_source}] | "
            f"location × {location_mult:.2f} | {management_note}"
        )

        seniority_assumption = f"Salary band: {seniority} (from {seniority_source})" + (
            " — LOW CONFIDENCE: no explicit seniority_level on role title; "
            "verify the band against the actual job level"
            if seniority_inferred
            else ""
        )
        composite_assumption = (
            f"Composite score (display only, not used for band): {score.total:.1f} "
            f"→ would map to '{display_seniority}' — education/skills excluded from band selection"
        )

        return SalaryEstimate(
            currency="CZK",
            low=low,
            mid=mid,
            high=high,
            reasoning=reasoning,
            assumptions=[
                seniority_assumption,
                composite_assumption,
                f"ISCO code: {salary_data.isco_code} ({salary_data.isco_match_level})",
                f"Location multiplier: × {location_mult:.2f}",
                (
                    f"Management multiplier: × {management_mult:.2f}"
                    + (
                        f" (skipped — {seniority} band already extrapolates above ISPV D9)"
                        if is_top_band
                        else ""
                    )
                ),
                f"Dataset: {salary_data.source_dataset_version}",
            ],
        )
