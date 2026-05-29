"""Salary estimator — ISPV-backed implementation.

Produces SalaryEstimate (CZK range) from Resume + SeniorityScore using
ISPV M8r 2025 dataset via ISPVLoader. Hard error if ISPV data is not loaded
or no ISCO code is present — no silent fallback per project decision Q4 2026.
"""

from __future__ import annotations

import logging
import re

from src.salary_ispv import (
    ISPVDataMissingError,
    ISPVLoader,
    ISPVLookupError,
    MissingISCOError,
    NonCZLocationError,
)
from src.schemas import Resume, SalaryBand, SalaryData, SalaryEstimate, SeniorityScore

logger = logging.getLogger(__name__)

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
    """Map SeniorityScore.total (0-100) to salary band name.

    # TODO(review): salary-band-from-total-score thresholds need empirical calibration.
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

        Args:
            resume: Structured resume with ISCO codes on experience entries.
            score:  Seniority score used to select the salary band.

        Returns:
            SalaryEstimate with low/mid/high CZK values.

        Raises:
            ISPVLookupError: If ISPV data is not loaded, no ISCO code found,
                             or ISCO lookup fails at all fallback levels.
        """
        seniority = _score_to_seniority(score.total)

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
        reasoning = (
            f"Zdroj: ISPV M8r 2025 (ISCO {salary_data.isco_code}, "
            f"{salary_data.isco_match_level}) | {seniority} band | "
            f"location × {location_mult:.2f} | {management_note}"
        )

        return SalaryEstimate(
            currency="CZK",
            low=low,
            mid=mid,
            high=high,
            reasoning=reasoning,
            assumptions=[
                f"Seniority band: {seniority} (score {score.total:.1f})",
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
