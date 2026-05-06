# Production Roadmap

What is shipped in this repo, what is deliberately MVP-scope, and what one to two weeks vs. one to two months of further work would unlock. Knowing the limits is the senior signal.

---

## What is shipped

- Six-stage pipeline: extract → parse → validate → score → estimate → explain. Per-stage `try / except`, per-run cost + duration in `PipelineResult.meta`.
- OpenAI Structured Outputs parser with `temperature=0.0`, refusal handling, rate-limit-aware retries.
- Hallucination guard via `ResumeValidator` — normalised substring matching against the raw CV text on every extracted entity (companies, emails, schools, skills).
- ISPV M8r 2025 salary backend: official Czech wage statistics with five-band decile mapping, regional + management multipliers, per-occupation-family inflation correction, ISCO whitelist with POCET-based fallback.
- Hard-fail surface for "we cannot honestly estimate this": non-CZ location, missing ISPV dataset, missing ISCO, lookup failure — each with a distinct exception subclass and surfaced as a targeted UI / HTTP message.
- Streamlit UI with sidebar ISPV downloader (scrapes `ispv.cz`), 3 result tabs, freshness-disclosure badge, HPP/IČO + sector caveats.
- Headless CLI (`main.py`) and FastAPI endpoint (`api.py`) for integration outside the UI.
- 182 pytest tests + 4 golden-set ISCO classification tests gated behind `OPENAI_API_KEY` (live LLM). Coverage 68 %.

---

## Known MVP gaps

These are documented, not hidden.

| Gap | Production-grade direction |
|---|---|
| **Sector tier not modeled** — banking / FAANG / agency / govt can shift estimates by ±15 % vs ISPV national bands | Curated company-tier dictionary (top 100 CZ employers) + parser-detected `Experience.company_tier` field, surfaced as a multiplier alongside region and management |
| **HPP-only output** — UI badge documents the IČO equivalent (~×1.4–1.6) but no toggle | Dual output (HPP + IČO equivalent) with explicit toggle. Requires schema field + UI rework |
| **Sample suppression** — narrow ISCO codes with low POCET fall back to broader prefixes; selection is highest-POCET in the prefix group | Multi-source merge (ISPV + Hays + LinkedIn surveys) to fill thin cells before ISPV publishes |
| **Lead / principal extrapolation** — bands above D9 are documented multipliers (×1.10–×1.75), not measured | Custom senior-comp survey across CZ tech (~50 respondents per family) calibrating the extrapolation curve |
| **Data lag** — ISPV releases the previous full year in March (≈ 2-month lag). Inflation correction compounds per-month from publication | Half-year revision import + sector-specific lag adjustments (tech moves faster than baseline ČNB inflation captures) |
| **No OCR** — scanned PDFs return a clean error | Tesseract + post-processing pipeline with quality scoring; OCR'd text routed through the same parser |
| **Eval coverage** — 4 synthetic CV fixtures across IT / healthcare / services + 1 real-world CV | 100+ anonymised real CVs across seniority levels and industries, with human-labelled ground truth scores and salary expectations |
| **`pipeline.py` integration coverage** — 0 % unit coverage; tested only via end-to-end runs | Mock-LLM + mock-ISPV integration tests covering each step's error path |
| **No multi-tenant / auth** — single-user demo | OAuth + per-tenant rate limits + audit logging of CV uploads (PII handling) |
| **No GDPR compliance audit** — temp-file ephemeral storage, no consent flow | Right-to-erasure, data residency, DPA with OpenAI (zero-data-retention path), ÚOOÚ documentation |

---

## What 1–2 weeks of work would unlock

- **Sector-tier multiplier.** Curate a CZ company-tier list (~100 firms covering banking, FAANG-CZ, scale-up agencies, government), add `Experience.company_tier` to the parser prompt, apply as an additional multiplier in the estimator. Closes the largest documented calibration gap.
- **Real-CV regression suite.** Solicit 20–50 anonymised CV donations across families. Run on every commit; alert on validation-flag pattern shifts.
- **Continuous eval.** Nightly run against the eval corpus. Track score drift and ISCO classification confusion matrix. Alert on > 5 % drift.
- **Multi-language output.** Currently CS-only output. Add `OUTPUT_LANGUAGE` env or per-request param; localise the UI chrome to match.
- **Pipeline integration tests.** Mock LLM + mock ISPVLoader; cover the exception-propagation paths that are currently exercised only by manual UI runs.
- **A/B prompt testing.** Versioned prompts, side-by-side eval, statistical significance check before promotion.

---

## What 1–2 months of work would unlock

- **Production deployment.** Containerised, OAuth-fronted, Redis rate limits, Stripe metered billing keyed to OpenAI cost, ≥ 2 replicas behind a health check, Grafana + alerting.
- **GDPR compliance.** Privacy policy / ToS, DPA with OpenAI, right-to-erasure, ÚOOÚ documentation.
- **Manual eval team.** Two to three Czech-speaking reviewers, eval corpus expansion to 500+ labelled CVs, reviewer agreement metrics (Cohen's κ).
- **Custom CZ-market model.** 1k+ labelled `(CV, score, salary)` tuples; fine-tune `gpt-4o-mini` or open-weight base; A/B against zero-shot.

---

## Cost at scale

Baseline measurement: ≈ $0.0013 per CV (`gpt-4o-mini`, parser + explainer). Latency 3–6 s end-to-end.

| Volume | Daily cost | Monthly cost | Notes |
|---|---|---|---|
| 10 CV/day | ~$0.013 | ~$0.40 | Within demo budget cap |
| 100 CV/day | ~$0.13 | ~$4 | Default `DAILY_API_BUDGET_USD=$5` fits |
| 1,000 CV/day | ~$1.30 | ~$40 | Bump cap to $10; add cost-trend monitoring |
| 10,000 CV/day | ~$13 | ~$400 | OpenAI tier-2 rate limits relevant; add worker queue (Celery/RQ); rotate cost log to SQLite |

---

## Deliberate non-goals

These are choices, not gaps.

- **Rule-based scoring over ML** — transparency over accuracy. ML would hide tradeoffs in opaque weights.
- **OpenAI-only provider** — `LLMProvider` ABC retained for future Ollama / local LLM. Multi-provider increases test surface for marginal MVP value.
- **Hard fail over silent fallback** — non-CZ location, missing ISPV data, missing ISCO all raise rather than return a number we cannot defend.
- **No automatic ISPV refresh** — manual sidebar download is intentional. ISPV publishes annually; live scraping adds ToS risk and brittleness.

---

Last updated: 2026-05-06.
