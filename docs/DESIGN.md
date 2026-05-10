# Design — Job Fit & Salary Estimator

Technický deep-dive pro reviewery: architektura, scoring methodology, ISPV decile mapping, hallucination guard, design decisions a honest limits. Pro rychlý start viz hlavní [`README.md`](../README.md).

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Scoring methodology](#scoring-methodology)
4. [Salary methodology — ISPV M8r](#salary-methodology--ispv-m8r)
5. [Hallucination guard](#hallucination-guard)
6. [LLM provider layer](#llm-provider-layer)
7. [Cost tracking & daily budget](#cost-tracking--daily-budget)
8. [Output schemas](#output-schemas)
9. [Personality traits — design note](#personality-traits--design-note)
10. [Configuration](#configuration)
11. [Honest limits](#honest-limits)
12. [Project layout](#project-layout)
13. [Testing & eval](#testing--eval)
14. [Production roadmap pointer](#production-roadmap-pointer)

---

## Overview

Pipeline převede CV na strukturovaný JSON se třemi hlavními výstupy:

- **Seniority Score** 0–100 s reasoning per komponent
- **Salary Estimate** v CZK měsíčně hrubé (low / mid / high)
- **Tři actionable doporučení** s odhadovaným dopadem na mzdu summujícím ≥ 30 %

Důraz je kladen na **vysvětlitelnost** — každé skóre má human-readable reasoning, salary má jmenovaný ISPV ISCO kód a match level, doporučení odkazují konkrétní akci. Žádný black-box ML model.

---

## Architecture

```
CV (PDF/DOCX)
    ↓
[1] Extractor   → ExtractedDocument (raw text + scanned-flag)
    ↓
[2] LLM Parser  → Resume (Pydantic, OpenAI Structured Outputs, temperature=0)
    ↓
[3] Validator   → list[ValidationFlag]   (substring hallucination guard)
    ↓
[4] Scorer      → SeniorityScore         (rule-based, 5 weighted components)
    ↓
[5] Estimator   → SalaryEstimate         (ISPV decile bands + multipliers)
    ↓
[6] LLM Explainer → Explanation          (3 recommendations, grounded in CV)
    ↓
PipelineResult (JSON-serialisable)
```

### Per-stage error handling

Pipeline orchestrátor (`src/pipeline.py`) má per-step `try/except` se selektivní tolerancí:

| Stage | Failure mode |
|---|---|
| Extract | `UnsupportedFormatError` → propaguje ven (HTTP 415) |
| Extract (scanned PDF) | Clean `ValueError` — OCR není v MVP |
| Parse | LLM error → `PipelineInfrastructureError` (HTTP 502) — partial cost zachován |
| Validate | Selhání → empty flags, pipeline pokračuje |
| Score | Selhání → fallback ScoreComponent, pipeline pokračuje |
| Estimate | `ISPVLookupError` / `NonCZLocationError` / `MissingISCOError` → propaguje (HTTP 422 / 503) |
| Explain | LLM error → fallback recommendations s `warning="below_30pct_target"` nebo `error="llm_failed"` |

**Cost record** se persistuje **i v error path** — `_persist_cost_record` se volá z outer `except`, takže parser tokens spotřebované před tím než estimator selhal se započítají do daily budgetu. Bez toho by refused/halucinované runs propalovaly tokeny "neviditelně".

### Stage details

| # | Modul | Vstup | Výstup | Náklady |
|---|---|---|---|---|
| 1 | `extractor.py` | `Path` | `ExtractedDocument` | $0 (lokální) |
| 2 | `parser.py` | raw text | `Resume` (Pydantic) | ~$0.001 (gpt-4o-mini) |
| 3 | `validator.py` | `Resume + raw_text` | `list[ValidationFlag]` | $0 (Layer 1), ~$0.0001 (Layer 2 opt-in) |
| 4 | `scorer.py` | `Resume` | `SeniorityScore` | $0 (čistá Python logika) |
| 5 | `estimator.py` + `salary_ispv.py` | `Resume + Score` | `SalaryEstimate + SalaryData` | $0 (lokální XLSX lookup) |
| 6 | `explainer.py` | `Resume + Score + Salary` | `Explanation` | ~$0.001 (gpt-4o-mini) |

---

## Scoring methodology

`SeniorityScore.total` je vážený součet 5 komponentů (váhy ∑ = 1.0):

```
total = 0.40 × Experience
      + 0.25 × Skills
      + 0.15 × Progression
      + 0.10 × Education
      + 0.10 × DomainExpertise
```

### ExperienceComponent (40 %)

- **Total years of experience** s **merged overlapping intervals** — paralelní HPP + freelance se nezapočítávají dvakrát.
- Base score: `min(100, total_years × 8)` — tedy 12.5 let → 100.
- **+10 bonus** pokud je v CV alespoň jedna senior/lead/principal role.
- "Ongoing" role se počítá proti aktuálnímu roku (per-call, nikoli zachycený při importu).

### SkillComponent (25 %)

- **Anti-inflation cap**: max 5 skills per kategorie (language / framework / tool / soft / domain).
- Base score: `min(70, capped_count × 4)`.
- **Depth bonus**: +3 za každý skill s `depth ∈ {advanced, expert}`, max +20.
- **Inflation penalty**: pokud `>30 skills && <5 with depth` → ×0.7. Cílí na "spray and pray" CV s 50 buzzwordy ale žádnou hloubkou.

### ProgressionComponent (15 %)

- Počet **upward seniority moves** napříč rolemi (junior → medior → senior → lead → principal).
- Score: `min(100, progressions × 30)`.
- `max()` invariant na běžícím prev_level — dočasný downward step (např. senior s contractor stintem popsaným jako "medior") nereresetuje bar a pozdější návrat na senior se nezapočítá jako čerstvý progress.
- **Contractor exception**: pokud jakákoli role má `is_contractor=True`, neaplikuje se job-hopping penalty (5+ rolí v 3 letech → -20).

### EducationComponent (10 %)

- Highest degree score: PhD 100, Master 75, Bachelor 50, other 25, none 0.
- **+10 bonus** pokud `field` matchuje keyword z `_RELEVANT_FIELDS_BY_FAMILY[primary_family]`.
- Primary family = `occupation_family` z **primary current role** (ongoing s nejdelší tenure, viz `_recency_key`).
- Keywords pokrývají EN i CZ (`computer science` / `informatika`, `medicine` / `medicína`, …).

### DomainExpertiseComponent (10 %)

- Měří **focus**: méně různých `occupation_family` napříč rolemi → vyšší skóre.
- Score: `max(0, 100 - (num_domains - 1) × 15)`. Tedy 1 domain = 100, 2 domains = 85, atd.

### Confidence reduction

Po skóre se aplikuje hallucination penalty: `confidence × (1.0 - errors × 0.20 - warnings × 0.05)`, clamp 0. UI to vizualizuje barvou (zelená ≥80 %, žlutá ≥50 %, červená).

---

## Salary methodology — ISPV M8r

### Datový zdroj

[ISPV M8r](https://www.ispv.cz/cz/vysledky-setreni/aktualni.aspx) — oficiální dataset Ministerstva práce a sociálních věcí ČR. Mzdová sféra (`MZS-M8r`) pokrývá soukromý sektor, publikuje se jednou ročně (Q1 následujícího roku) v XLSX. Stahovač v Streamlit sidebaru scrape-uje `ispv.cz` index page a stáhne aktuální `MZS_M8r-xlsx` (~150 kB).

XLSX má pro každý 4-digit ISCO-08 kód:

| Sloupec | Význam |
|---|---|
| POCET | Počet respondentů (tis. osob) |
| MEDIAN | Mediánová mzda CZK/měs |
| D1 | 10. percentil |
| Q1 | 25. percentil |
| Q3 | 75. percentil |
| D9 | 90. percentil |
| PRUMER | Aritmetický průměr |

### ISCO klasifikace v parseru

LLM parser dostává seznam nejčastějších CZ-ISCO-08 kódů + disambiguation pravidla:

```
2512 Software developers (pure dev)
2511 Systems analysts (DevOps, SRE, Cloud, Platform)
2522 Systems administrators (legacy sysadmin, NOT DevOps)
2519 Software/applications developers NEC (Data, ML, QA, DBA)
2211 General practitioners
2221 Nursing professionals
3322 Commercial sales representatives
1120 Managing directors and CEOs
…
```

Disambiguation matters: ISPV median pro `2511` ≈ 92 k, `2512` ≈ 98 k, `2522` ≈ 74 k. Špatný kód = špatný odhad o desítky tisíc.

### Decile mapping → 5 seniority bands

```
junior     D1                 → midpoint(D1, Q1)        → Q1
medior     Q1                 → MEDIAN                  → midpoint(MEDIAN, Q3)
senior     midpoint(MEDIAN, Q3) → Q3                    → D9
lead       D9 × 1.10          → D9 × 1.20               → D9 × 1.35
principal  D9 × 1.30          → D9 × 1.50               → D9 × 1.75
```

Lead/principal jsou extrapolované nad ISPV 90. percentil — ISPV survey nepokrývá tyto úrovně dostatečně. Multipliery jsou kalibrovány proti veřejným reportům (No Fluff Jobs CZ Salary Report, Algoritma surveys).

### Multipliers

- **Region** — Praha 1.15, Brno 1.00, Plzeň 0.88, Ostrava/Olomouc/Zlín/Liberec/Pardubice/Hradec/ČB 0.85, Jihlava/Teplice 0.80, Karviná/Most 0.78. Lokace mimo whitelist = default 1.0 (CZ-default — vyhneme se false-rejectu pro Tábor, Karlovy Vary, Mladá Boleslav). Detekce non-CZ lokace přes `_NON_CZ_COUNTRIES_RE` (Berlin, Bratislava, London, …) → `NonCZLocationError`.
- **Management** — ×1.10 pokud `is_management=True` na jakékoli zkušenosti. Konzervativní, fits team-lead level. Director/VP je known backlog item.
- **Inflační korekce** — `annual_rate ** (age_months / 12)`, per-occupation-family rate v `data/inflation_factors.json` (IT 1.08, healthcare 1.05, services 1.04, ...). Compoundováno per-month od `_ISPV_PUBLICATION_DATE`.

### ISCO fallback chain

Když parser vyhodí kód, který není v ISPV indexu (model halucinuje):

```
4-digit exact → 3-digit prefix (highest POCET) →
2-digit prefix → 1-digit prefix → family_fallback (highest POCET in same occupation_family)
```

Match level je propagovaný do `SalaryData.isco_match_level` a `confidence`:

| Match level | Confidence default | UI badge |
|---|---|---|
| 4digit + POCET ≥ 30 | high | zelená |
| 4digit + POCET < 30 | medium | žlutá |
| 3digit / 2digit | medium | žlutá |
| 1digit / family_fallback | low | červená + warning |

### Hard-fail policy

Místo silent fallback (vrácení obecného odhadu nebo nuly) raisuje pipeline dedikované exception:

- `NonCZLocationError` (HTTP 422) — lokace jasně mimo CZ
- `MissingISCOError` (HTTP 422) — parser nedetekoval ISCO ani jedné role
- `ISPVDataMissingError` (HTTP 503) — operátor nestáhl XLSX
- `ISPVLookupError` (HTTP 422) — ISCO není v datech a fallback chain selhala

UI/CLI/API mají dedikované error handlery s konkrétními česky lokalizovanými hláškami.

---

## Hallucination guard

### Layer 1 — substring guard

`ResumeValidator` (`src/validator.py`) cross-checkuje každé extrahované pole proti raw textu:

- `_normalize()`: NFKD lowercase + strip diakritiky + collapse whitespace. Účel: `"Ceska Sporitelna"` extrahované parsem matchne `"Česká spořitelna"` v raw textu bez false-positive.
- `_is_in_text()`: pro tokeny **≤ 3 znaků** word-boundary regex (`(?<!\w)c\+\+(?!\w)` — `re.escape` pokrývá special chars), pro delší plain substring containment.
- Severity: `error` pro full_name a email mismatch, `warning` pro company / start_year / institution / skill, `info` pro education year.

### Layer 2 — embedding guard (opt-in)

`EmbeddingValidator` (`src/validator.py`) — když Layer 1 vyflagovala chybu nebo warning, zavolá batched embedding přes `text-embedding-3-small` a porovná flag issue text proti chunked raw text (200 char windows). Pokud max cosine similarity > 0.65, přidá info-level flag s notou "fuzzy match found" — typicky paraphrases jako `ČVUT` ↔ `Czech Technical University`.

Threshold 0.65 byl kalibrován proti reálným měřením:

- EN/CS paraphrase pairs: ~0.67
- Different institutions: ~0.41
- Same string self-similarity: 1.00

(Původní spec měla 0.85; bylo to nedosažitelné pro legitimní paraphrases — viz Phase 19 review.)

Off by default, aby běžný run neplatil ~$0.0001 navíc za embeddingy.

### Confidence propagation

```python
confidence_reduction = max(0, 1.0 - errors × 0.20 - warnings × 0.05)
score.confidence = round(score.confidence × confidence_reduction, 3)
```

Streamlit UI vizualizuje confidence barvou v "Skóre & Mzda" tabu.

---

## LLM provider layer

`src/llm_provider.py` má `LLMProvider` ABC + jednu konkrétní implementaci `OpenAIProvider`. AnthropicProvider byl vyhozen 2026-05-05 (jediný blocker byl tool calling shape pro structured outputs; ABC zůstává jako extensibility point pro Ollama / lokální LLM).

### Klíčové invariants

- **Temperature 0.0** (gpt-4o*) — deterministický strukturovaný extract. Pro gpt-5* family je temperature override odmítnut API, takže se vynechává (model default 1.0 — známý compromise).
- **Structured Outputs** přes `client.beta.chat.completions.parse` s `response_format=<PydanticModel>`. Žádný JSON parsing, žádný schema validation manuálně.
- **Refusal handling** je mandatory — `msg.refusal` se kontroluje před `model_validate`, jinak by Pydantic vrátil garbage.
- **Retry**: 3 attempts s exponenciálním delay (1 s / 2 s / 4 s), respektuje `Retry-After` header. Refusal a None-parsed jsou permanent (bez retry).
- **Partial cost preservation**: `LLMProviderError` nese `cost_usd` z usage objektu i když request selhal — daily budget reflectuje reálný spend, ne nulu.

### gpt-5* gotchas

Komentář v `_PRICING` dictionary + odkaz na `decisions/2026-05-05-dynamic-salary.md`:

1. Používá `max_completion_tokens` (gpt-4o akceptuje oba)
2. Odmítá custom temperature s HTTP 400
3. Reasoning tokens se počítají do `max_completion_tokens`; default 4096 sežere reasoning a zbude 0 output. Potřeba 16k+ headroom pro netriviální schema.

### Pre-flight cost reservation

`estimate_run_cost_usd(parser_model, explainer_model)` spočítá worst-case cost z `_PRICING` dictionary × typical token mix × 2.5× safety margin. Tedy pre-flight reservation pro `check_budget()` je realistická per-model:

| Model | Reservation |
|---|---|
| gpt-4o-mini × gpt-4o-mini | ~$0.005 |
| gpt-4o × gpt-4o | ~$0.083 |

Tím $5 daily cap admit-uje ~1000 mini runs nebo ~60 4o runs, místo dříve broken flat $0.005 který pro gpt-4o under-počítal 8×.

---

## Cost tracking & daily budget

`src/cost_tracker.py` persistuje per-run record do `data/.cost_log.jsonl` (gitignored, append-only):

```json
{"run_id":"a1b2c3d4e5f6","timestamp_utc":"2026-05-10T14:32:11+00:00",
 "fixture_or_file":"cv_senior.pdf","total_cost_usd":0.0014,
 "parse_cost_usd":0.0008,"explain_cost_usd":0.0006,"embed_cost_usd":0.0,
 "duration_s":4.2,"score_total":67.4,"error":null}
```

`check_budget(estimated_cost_usd)` zavolaný před LLM voláním:

- Sumuje today's spend z JSONL
- Pokud `spent + estimate > budget` → `BudgetExceededError` (HTTP 429)
- Pokud `spent + estimate > budget × 0.90` → `logger.warning` (no raise)

`DAILY_API_BUDGET_USD` je field na `Settings` (Pydantic), takže `.env` value se aplikuje lokálně — fixed bug kde to dříve fungovalo jen v Dockeru přes `env_file:`.

`scripts/show_metrics.py` agregátně reportuje today/week/total spend, fail rate, mean duration.

---

## Output schemas

Všechny Pydantic v2 modely jsou v `src/schemas.py` (single source of truth — jiné moduly nesmí definovat `BaseModel`). Klíčové:

```python
class Resume(BaseModel):
    full_name: str | None
    location: str | None
    email: str | None
    phone: str | None
    languages: list[Language]
    educations: list[Education]
    experiences: list[Experience]
    skills: list[Skill]
    raw_text_length: int

class Experience(BaseModel):
    role_title: str
    isco_code: str | None       # validated regex r"^[1-9][0-9]{3}$"
    occupation_family: Literal["it","healthcare","legal",
                                "education","trades","services","other"] | None
    company: str
    start_year: int
    end_year: int | None         # None = ongoing
    description: str | None
    seniority_level: Literal["junior","medior","senior","lead","principal"] | None
    is_management: bool
    is_contractor: bool

class Recommendation(BaseModel):
    title: str
    why_it_matters: str
    estimated_salary_impact_pct: float = Field(ge=0, le=100)
    timeframe_months: int = Field(ge=1, le=60)
    first_action: str

class Explanation(BaseModel):
    summary: str
    strengths: list[str]
    gaps: list[str]
    recommendations: list[Recommendation]   # validator vynutí len == 3
```

`PipelineResult.meta` obsahuje per-stage timings, costs, model names, run_id, salary_source label, salary_data (full SalaryData s confidence + warnings).

---

## Personality traits — design note

Zadání zmiňuje senioritu jako kompozit *„dovedností, zkušeností, **osobnostních rysů** a vzdělání"*. Scorer pokrývá čtyři z pěti explicitně:

| Brief faktor       | Scorer komponent       | Status |
|--------------------|------------------------|--------|
| Skills             | `SkillComponent` (25%) | ✓      |
| Experience         | `ExperienceComponent` (40%) | ✓ |
| Education          | `EducationComponent` (10%) | ✓ |
| Personality traits | — (ne přímo)           | ⚠ proxy |
| (added)            | `ProgressionComponent` (15%) | proxy pro drive / ambici |
| (added)            | `DomainExpertiseComponent` (10%) | proxy pro focus / specialization |

**Personality traits jsou záměrně neextrahované** z CV textu, protože signal je nespolehlivý:

- Sebe-popisy ("team player", "detail-oriented") jsou téměř univerzální a adversariálně gameable
- Extrakce traits z popisů projektů by znamenala LLM hádání bez grounding
- Validace by vyžadovala labeled training data linkující CV phrasing k validovaným trait inventories (Big Five, DISC) — out of scope pro MVP

Místo toho dva indirect proxies z objektivních CV dat:

- **Progression** — počet upward seniority moves. Junior → Medior → Senior za 8 let signaluje drive a schopnost růst v organizaci, bez ohledu na to, jak se kandidát popisuje.
- **DomainExpertise** — méně různých role types = focus, více = scattered. Proxy pro specialization depth.

Pro produkční řešení by skutečné personality assessment vyžadovalo separátní dotazník nebo labeled training data. Out of scope pro MVP.

---

## Configuration

| Variable | Default | Účel |
|---|---|---|
| `OPENAI_API_KEY` | required | OpenAI API key |
| `PARSER_MODEL` | `gpt-4o-mini` | Model pro CV parsing (schema-aware) |
| `EXPLAINER_MODEL` | `gpt-4o-mini` | Model pro doporučení |
| `DAILY_API_BUDGET_USD` | `5.00` | Hard cap; pipeline raisuje před LLM voláním když by se překročilo |

Per-CV cost je ~$0.001–$0.002 s `gpt-4o-mini`, ~$0.02–$0.04 s `gpt-4o`. End-to-end latency 3–6 s.

Setup:

```bash
cp .env.example .env
# Edit .env, doplň OPENAI_API_KEY
```

---

## Honest limits

Každý je tradeoff, ne bug. Vidi je každý kdo čte source.

| Area | Current | Production-grade |
|---|---|---|
| Salary data lag | ISPV publikuje Q1 následujícího roku (2025 data → březen 2026 release). Inflační korekce per-month od publikace | Continuous refresh + half-year revize import + sub-national XLSX import |
| ISCO accuracy | LLM-emitted 4-digit kód, validovaný proti ISPV whitelistu s family-level fallback | Whitelist enforcement at parse time + sekundární klasifikátor (taxonomy lookup) |
| Sector tier | Není modelováno — banking / FAANG / agentura / state mohou posunout o ±15 % | Curated company-tier dictionary + parser-detected `Experience.company_tier` |
| HPP vs IČO | Veškerý output je HPP gross měsíčně. UI badge dokumentuje rozdíl; IČO ekvivalent se nepočítá | Dual output (HPP + IČO ~×1.4–1.6) s toggle |
| Lead/principal bandy | Extrapolované nad ISPV 90. percentilem dokumentovanými multiplikátory | Custom survey senior compensation + equity disclosure |
| Tech bonus underreport | ISPV pokrývá payroll only; AI/ML/cloud premium a equity jsou částečně neviditelné. Mitigováno per-family inflation rates (IT ×1.08/yr) | Vendor-specific compensation feeds (AON, Mercer) |
| Sample suppression | Některé úzké ISCO kódy mají málo respondentů; loader fallbackuje na širší kódy s `confidence=medium` warning | Multi-source merge, region-stratified sampling |
| Output language | Pouze čeština — explainer prompt vynucuje CS bez ohledu na jazyk CV | Locale detection + per-request `OUTPUT_LANGUAGE` override |
| OCR | Není podporováno — skenované PDF vrací clean error | Tesseract + post-processing |
| Eval dataset | 4 syntetické PDF fixtures (junior, medior, principal, contractor) + 2 placeholder TXT pro extender | 100+ anonymizovaných reálných CV s ground-truth labels |
| API auth & rate limits | Žádné — `/analyze` je open; spoléhá pouze na `DAILY_API_BUDGET_USD` cap | OAuth + per-tenant rate limits + audit logging |
| Personality traits | Proxy přes Progression + DomainExpertise (viz design note) | Big Five questionnaire + labeled CV-phrasing → trait inventory mapping |

Sekvenční production roadmap (1–2 týdny vs. 1–2 měsíce další práce) viz [`docs/PRODUCTION_ROADMAP.md`](PRODUCTION_ROADMAP.md).

---

## Project layout

```
src/
├── schemas.py       # Pydantic v2 models — single source of truth
├── extractor.py     # PDF/DOCX → raw text
├── parser.py        # Raw text → Resume (LLM)
├── validator.py     # Hallucination guard (Layer 1 + Layer 2)
├── scorer.py        # Rule-based seniority score (5 components)
├── salary_ispv.py   # ISPV M8r loader + decile mapping + ISCO fallback chain
├── estimator.py     # Seniority + ISPV → SalaryEstimate
├── explainer.py     # LLM recommendations (3 recs, +30% impact enforced)
├── pipeline.py      # End-to-end orchestrator + per-stage error handling
├── llm_provider.py  # OpenAI provider via abstract base class + cost estimation
├── cost_tracker.py  # JSONL cost log + daily budget enforcement
├── config.py        # Pydantic Settings (.env-aware)
├── logging_config.py # structlog JSON + run_id correlation
└── paths.py         # Project-anchored filesystem paths

prompts/
├── cv_parser_system.txt    # ISCO classification + extraction rules
└── explainer_system.txt    # Strengths/gaps/recs grounding + +30% constraint

data/
├── ispv_2025.xlsx          # Downloaded by sidebar button (gitignored)
├── inflation_factors.json  # Per-occupation-family annual CPI
└── .cost_log.jsonl         # Append-only cost log (gitignored)

ui/
├── app.py          # Streamlit 3-tab UI (Profil / Skóre & Mzda / Doporučení)
└── styles.py       # Custom CSS (jfe-* classes)

tests/               # 195 pytest tests + 6 golden-set evals (live LLM)
scripts/
├── eval.py             # Eval harness s HTML reportem
├── generate_fixtures.py # Synthetic CV PDF generator (reportlab)
└── show_metrics.py     # Daily/weekly cost + fail rate aggregation
docs/
├── DESIGN.md           # Tento soubor
└── PRODUCTION_ROADMAP.md # Sequenced "what next" plan

main.py             # CLI entrypoint
api.py              # FastAPI endpoint
.github/workflows/test.yml # ruff + pyright + pytest on push/PR
```

---

## Testing & eval

### Unit tests

```bash
make test            # 195 testů, ~2 s
uv run pytest tests/test_scorer.py -v   # Single module
uv run pytest --cov=src --cov-report=html
```

Coverage napříč moduly:

| Modul | Test soubor | Tests |
|---|---|---|
| extractor | test_extractor.py | edge cases (gibberish, scanned, formats) |
| parser | test_parser.py + test_parser_isco_eval.py | mocked + ISCO accuracy |
| validator | test_validator.py | substring, diakritika, embedding fuzzy |
| scorer | test_scorer.py | per-component + integration |
| salary_ispv | test_salary_ispv.py | XLSX parse + decile bands + fallback chain |
| estimator | test_estimator.py | multipliers + non-CZ guard + recency |
| explainer | test_explainer.py | retry loop + +30% constraint + fallback |
| pipeline | test_pipeline.py | end-to-end mocked + error path coverage |
| api | test_api.py | HTTP status codes + error handlers |
| cost_tracker | test_cost_tracker.py | budget + JSONL + edge cases |
| llm_provider | test_llm_provider.py | mocked OpenAI + retry + cost computation |

### Eval harness (live LLM)

```bash
make eval                                          # 6 fixtures
uv run python scripts/eval.py --html report.html   # screen-share-ready report
uv run python scripts/eval.py --cv path/to/cv.pdf  # single CV
```

Fixtures: `cv_junior.pdf`, `cv_medior.pdf`, `cv_principal.pdf`, `cv_contractor.pdf` jsou aktuálně committed. `cv_senior.pdf` a `cv_real_world.pdf` jsou v EXPECTED mapě v `scripts/eval.py` — eval harness je SKIPne, pokud soubor chybí.  Generuj přes `scripts/generate_fixtures.py` nebo doplň vlastní.

PASS/FAIL kritéria: score in range, salary_low ≥ min, salary_high ≤ max, has 3 recs, has strengths, has gaps. Real-world CV má wide ranges intentionally — sledujeme robustnost, ne strict accuracy.

### CI

`.github/workflows/test.yml` na push + PR:

1. ruff check + format check
2. pyright (basic mode)
3. pytest

Žádné live LLM v CI — eval harness je on-demand kvůli nákladu a flakiness.

---

## Production roadmap pointer

Sequenced plán "co další 1–2 týdny vs. 1–2 měsíce další práce odemkne":

- **1–2 týdny** — OAuth + rate limits, OCR fallback (Tesseract), output language detection, ISPV semi-annual revize import, sektor tier dictionary (top 50 CZ company prefixes)
- **1–2 měsíce** — labeled real-CV eval set (100+ anonymized CV s ground-truth), separátní personality questionnaire, IČO equivalent toggle, multi-source salary merge (Glassdoor scrape + No Fluff Jobs CZ + ISPV reconciliation)

Plný plán: [`docs/PRODUCTION_ROADMAP.md`](PRODUCTION_ROADMAP.md).

---

## License

MIT.
