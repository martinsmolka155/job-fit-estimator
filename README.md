# Job Fit & Salary Estimator

End-to-end Python pipeline. Vstup: CV (PDF/DOCX). Výstup: skóre seniority 0–100, odhad měsíční hrubé mzdy v CZK a 3 doporučení, jak ji posunout výš (každé s poctivým tier + range dopadem).

## Spuštění (1–2 příkazy)

```bash
make install                  # uv sync
cp .env.example .env          # doplň OPENAI_API_KEY
make run-ui                   # → http://localhost:8501
```

V sidebaru klikni na **🔄 Stáhnout ISPV** — stáhne se `data/ispv_2025.xlsx` (~150 kB) z ispv.cz. Pak nahraj CV a vidíš výsledky.

Alternativy:

```bash
uv run python main.py path/to/cv.pdf --pretty       # CLI
uv run uvicorn api:app --reload                     # HTTP API → /docs
docker compose up                                   # Streamlit v kontejneru
```

## Co dostaneš

```
{
  "score": { "total": 67.4, "components": [...], "confidence": 0.95 },
  "salary": { "low": 78000, "mid": 92000, "high": 110000, "currency": "CZK" },
  "explanation": {
    "summary": "...",
    "strengths": [...],
    "gaps": [...],
    "recommendations": [
      { "title": "...", "impact_tier": "medium", "impact_range_pct": { "low_pct": 8, "high_pct": 15 }, "timeframe_months": 6, "first_action": "...", ... },
      { ... }, { ... }   // právě 3 doporučení; dopad jako tier + range, žádný umělý součet
    ]
  },
  "validation_flags": [...],   // hallucination guard
  "meta": { "total_cost_usd": 0.0014, "total_duration_s": 4.2, ... }
}
```

## Jak funguje pipeline

```
CV (PDF/DOCX)
  ↓ [1] Extractor       PyMuPDF / python-docx → raw text (scanned PDF → clean error)
  ↓ [2] LLM Parser      OpenAI Structured Outputs (Pydantic, temp=0) → Resume + ISCO-08 kód
  ↓ [3] Validator       Substring guard (lowercase + bez diakritiky) → ValidationFlags
  ↓ [4] Scorer          Rule-based, 5 vážených komponentů, žádné ML black-box
  ↓ [5] Estimator       ISPV decile bands → region × management × inflation
  ↓ [6] LLM Explainer   3 doporučení (impact tier + range), CZ output, grounded v CV
PipelineResult
```

**Skóre = 0.40 × Experience + 0.25 × Skills + 0.15 × Progression + 0.10 × Education + 0.10 × DomainExpertise**

Každý komponent má human-readable `reasoning` string viditelný v UI.

## Jak jsem přistoupil k datům

**ISPV M8r 2025** — oficiální datasety MPSV. ISPV (Informační systém o průměrném výdělku) publikuje jednou ročně XLSX s decilovými rozdělením mezd pro každý 4-digit CZ-ISCO-08 kód. Stahovač v UI scrape-uje `ispv.cz` index page a stáhne aktuální `MZS_M8r-xlsx` soubor (~150 kB).

LLM parser přiřadí každé pracovní zkušenosti ISCO kód (např. `2512` = Software developers); estimator vyhledá decily, sestaví 5 seniority bandů (junior … principal) a aplikuje:

- **Region multiplier** — Praha ×1.15, Brno ×1.00, regionální CZ ×0.78–0.88. Non-CZ lokace **hard-fail** — ISPV pokrývá jen CZ trh, vracet odhad pro Berlín by lhalo.
- **Management multiplier** — ×1.10 když parser detekuje lead/manager titul nebo direct reports.
- **Inflační korekce** — compound per-month od publikace datasetu, per-occupation-family rate (IT ×1.08/yr, healthcare ×1.05/yr, …) v `data/inflation_factors.json`.

Pokud parser vyhodí ISCO mimo dataset (model halucinuje 4-digit kód), loader fallbackuje přes 3-digit → 2-digit → 1-digit prefix → highest-POCET v occupation_family. Match level se propaguje až do UI jako confidence badge.

## Validace + sanity checks

`ResumeValidator` cross-checkuje každé extrahované pole proti raw textu. Substring match s normalizací (lowercase + bez diakritiky + collapse whitespace), takže `"Ceska Sporitelna"` extrahované z `"Česká spořitelna"` neflagne false-positive. Hallucinace snižují `confidence` (0.20 za error, 0.05 za warning) a UI zobrazí varování.

Volitelný **Layer 2 — EmbeddingValidator** (off by default, opt-in v `Pipeline(enable_embedding_validator=True)`) přidá embedding-based fuzzy match přes OpenAI `text-embedding-3-small` pro paraphrases typu `"ČVUT"` ↔ `"Czech Technical University"`.

## Test plan

```bash
make test                              # unit testy (pytest), ~2 s  — NO LLM calls
uv run python scripts/eval.py --limit 2          # accuracy eval na 2 CVs (~$0.004)
uv run python scripts/eval.py --html report.html # full eval + HTML report
```

## Evaluating accuracy

Accuracy is measured against a labeled ground-truth file, not just plausibility
ranges. The harness lives in `scripts/eval.py`; labels in `data/eval_labels.json`.

### Label schema

```json
{
  "cv": "tests/fixtures/cv_it_developer.txt",
  "occupation": "Backend developer / Python architect",
  "isco_code": "2512",
  "seniority": "junior|medior|senior|lead|principal",
  "salary_czk_month": 95000,
  "salary_source": "ISPV-2025-annual ISCO 2512 senior band + Praha x1.15",
  "notes": "optional context"
}
```

`salary_czk_month` can be a scalar int (single known figure) or `{"low": X, "high": Y}`
for a range. `salary_source` documents where the truth came from:
- `"ISPV-2025-annual ..."` — derived from the ISPV dataset + documented multipliers (legitimate ground truth for synthetic profiles)
- `"real"` — actual known salary from an offer letter / negotiation
- `"TODO_real"` — placeholder; the CV owner must fill before this entry contributes to salary metrics

Entries with `isco_code == "TODO"` or `seniority == "TODO"` or `salary_source` containing
`"TODO_real"` are excluded from accuracy metrics but still run through the pipeline
(crash detection stays active).

### Metrics computed

| Metric | What it measures |
|---|---|
| ISCO exact match | 4-digit ISCO code exact equality rate |
| ISCO 3-digit prefix | 3-digit prefix agreement (occupation group correct, sub-group off) |
| Seniority exact | Band exact match rate |
| Seniority off-by-one | Adjacent-band agreement (e.g. medior predicted vs senior true) |
| Salary MAE | Mean absolute error of predicted midpoint vs true midpoint (CZK) |
| Salary MAPE | Mean absolute percentage error vs true midpoint |
| Within ±15% | Fraction of CVs where predicted mid is within ±15% of true mid |
| True in pred range | Fraction where true salary label's range overlaps predicted low–high |

### Running the eval

```bash
# All labeled CVs (~$0.02 on gpt-4o-mini for 5 entries)
uv run python scripts/eval.py

# Cost-controlled runs
uv run python scripts/eval.py --limit 2          # first 2 entries
uv run python scripts/eval.py --only nurse       # entries whose cv path contains "nurse"

# Exports
uv run python scripts/eval.py --json out.json
uv run python scripts/eval.py --html out.html
```

The script calls OpenAI — it is **not** run in `pytest` (no CI cost). Run it manually
or as a nightly job.

### Adding a labeled CV

1. Place the CV file in `tests/fixtures/` (PDF or TXT).
2. Add an entry to `data/eval_labels.json` with the correct `isco_code`, `seniority`,
   and `salary_czk_month`. Use `salary_source: "real"` for actual salaries, or
   `salary_source: "ISPV-2025-annual ..."` when the label was derived from ISPV data.
3. Commit `data/eval_labels.json` and the fixture file. The JSON is tracked — it is
   the ground truth; it does not grow from prompt iteration.

### Calibration target

The current labeled set has 4 synthetic profiles + 1 real-world TODO. Accuracy numbers
from 5 CVs are anecdotal. The target for trustworthy calibration is **~100 real labeled
CVs** covering a spread of roles, seniority levels, locations, and sectors. Until then,
the eval serves as a regression check and a way to surface concrete failure modes, not
as a publishable benchmark.

This harness is what unblocks the `TODO(review)` items in `src/estimator.py` and
`src/scorer.py` — Praha multiplier calibration, seniority-band thresholds, and ISPV
confidence scoring all require a labeled set to tune and validate against.

## Náklady & latency

- gpt-4o-mini: ~$0.001–0.002 / CV, end-to-end 3–6 s
- gpt-4o: ~$0.02–0.04 / CV
- Daily cap: `DAILY_API_BUDGET_USD` (default $5) — pre-flight reservation per model

## Limity

Krátký výčet, plný v [`docs/DESIGN.md`](docs/DESIGN.md#honest-limits): ISPV je roční snapshot (kompenzováno inflační korekcí), sektor (banking / FAANG / agentura) není modelovaný (±15 %), výstup je HPP gross měsíčně (IČO ekvivalent ~×1.4–1.6, badge to dokumentuje), API nemá auth ani rate limit (jen budget cap), OCR pro skenované PDF není podporováno.

Plný technický popis, design rationale, scoring methodology, ISPV decile mapping a production roadmap najdeš v **[`docs/DESIGN.md`](docs/DESIGN.md)**.

## Licence

MIT.
