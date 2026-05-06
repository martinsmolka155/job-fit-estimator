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
    # Cross-border remote indicators — "Remote EU", "Europe remote", etc.
    # all imply non-CZ employer; the README contract says hard-fail.
    r"eu|europe|european|"
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
    """Map SeniorityScore.total (0-100) to salary band name."""
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
        """
        location = (resume.location or "").lower()
        for keyword, (mult, _label) in _LOCATION_MULTIPLIERS.items():
            if keyword in location:
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
        # Sort key: ongoing role first (end_year is None), then by end_year desc,
        # then by start_year desc (longer current tenure outranks short side gigs).
        isco_code: str | None = None

        def _recency_key(exp: object) -> tuple[int, int, int]:
            end = getattr(exp, "end_year", None)
            start = getattr(exp, "start_year", 0) or 0
            is_current = 1 if end is None else 0
            return (is_current, end or 0, start)

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

        # Whitelist check — if isco_code is absent from ISPV index, attempt family fallback.
        known = self._loader.known_codes()
        if isco_code not in known:
            logger.warning(
                "ISCO %s not in ISPV whitelist (%d known codes) — falling back via occupation_family",
                isco_code,
                len(known),
            )
            family: str | None = None
            for exp in sorted(resume.experiences, key=_recency_key, reverse=True):
                if exp.isco_code == isco_code and exp.occupation_family:
                    family = exp.occupation_family
                    break
            if family is None:
                raise ISPVLookupError(
                    f"ISCO {isco_code!r} not in ISPV dataset and no occupation_family available for fallback."
                )
            fallback_code = self._loader.best_code_for_family(family)
            if fallback_code is None:
                raise ISPVLookupError(f"No ISPV data for occupation_family {family!r}.")
            isco_code = fallback_code

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
        management_mult = 1.10 if self._has_management(resume) else 1.0

        low = round(band.low * location_mult * management_mult / 1000) * 1000
        mid = round(band.mid * location_mult * management_mult / 1000) * 1000
        high = round(band.high * location_mult * management_mult / 1000) * 1000

        reasoning = (
            f"Zdroj: ISPV M8r 2025 (ISCO {salary_data.isco_code}, "
            f"{salary_data.isco_match_level}) | {seniority} band | "
            f"location × {location_mult:.2f} | mgmt × {management_mult:.2f}"
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
                f"Management multiplier: × {management_mult:.2f}",
                f"Dataset: {salary_data.source_dataset_version}",
            ],
        )
