# Job Fit & Salary Estimator

End-to-end Python pipeline. Vstup: CV (PDF/DOCX). Výstup: skóre seniority 0–100, odhad měsíční hrubé mzdy v CZK a 3 doporučení, jak ji posunout o ≥30 %.

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
      { "title": "...", "estimated_salary_impact_pct": 12, "timeframe_months": 6, "first_action": "...", ... },
      { ... }, { ... }   // sum ≥ 30 % je vynucený
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
  ↓ [6] LLM Explainer   3 doporučení (sum impact ≥30 %), CZ output, grounded v CV
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
make test                              # 195 unit testů (pytest), ~2 s
make eval                              # eval harness (live LLM) na PDF fixtures v tests/fixtures/
uv run python scripts/eval.py --html report.html
```

## Náklady & latency

- gpt-4o-mini: ~$0.001–0.002 / CV, end-to-end 3–6 s
- gpt-4o: ~$0.02–0.04 / CV
- Daily cap: `DAILY_API_BUDGET_USD` (default $5) — pre-flight reservation per model

## Limity

Krátký výčet, plný v [`docs/DESIGN.md`](docs/DESIGN.md#honest-limits): ISPV je roční snapshot (kompenzováno inflační korekcí), sektor (banking / FAANG / agentura) není modelovaný (±15 %), výstup je HPP gross měsíčně (IČO ekvivalent ~×1.4–1.6, badge to dokumentuje), API nemá auth ani rate limit (jen budget cap), OCR pro skenované PDF není podporováno.

Plný technický popis, design rationale, scoring methodology, ISPV decile mapping a production roadmap najdeš v **[`docs/DESIGN.md`](docs/DESIGN.md)**.

## Licence

MIT.
