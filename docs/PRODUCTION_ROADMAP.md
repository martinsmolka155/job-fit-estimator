# Production Roadmap

Honest documentation of what Phases 18–20 shipped and what's still missing for full production deployment. Senior engineering signal: knowing the limits.

---

## What's done — Phases 18–20

### Phase 18 — Salary data pipeline
- `SalaryDataSource` ABC with `StaticJSONSource` (default) and `LocalCSVSource` (manual CSU.gov.cz CSV import)
- CLI refresh command: `uv run python scripts/update_salary_data.py [--static | --csv path]`
- Validation layer: sanity-checks band ranges (low ≤ mid ≤ high, monotonic seniority, no negatives) before write
- `_meta.fetched_at` + `_meta.source_url` enrichment
- Fallback wrapper: primary source → secondary source on validation/fetch failure

### Phase 19 — Multi-layer hallucination detection
- Layer 1 (existing): `ResumeValidator` normalized substring matching
- Layer 2 (new, opt-in): `EmbeddingValidator` cosine similarity via OpenAI `text-embedding-3-small`
- Threshold 0.65 calibrated against real measurements (ČVUT ↔ "Czech Technical University" = 0.677, ČVUT ↔ Cambridge = 0.41)
- Pure-Python cosine implementation (no numpy dependency)
- Pipeline integration via `enable_embedding_validator: bool = False` (default off — cost transparency)

### Phase 20 — Observability + cost guardrails
- Structured JSON logging via structlog with per-run `run_id` correlation
- Per-run cost tracking: parse + explain + embed costs aggregated in `PipelineResult.meta.total_cost_usd`
- Append-only cost log at `data/.cost_log.jsonl` (gitignored)
- Daily budget cap: `DAILY_API_BUDGET_USD` env (default $5.00); raises `BudgetExceededError` BEFORE LLM calls if would exceed; warns at 90% consumption
- CLI metrics dashboard: `uv run python scripts/show_metrics.py --last N | --today`

### Phase 22 — Dynamic salary research with Tavily + caching
- `SalaryResearcher` calls Tavily search + extracts via `gpt-5-mini` (fallback to `gpt-5`)
- `SalaryCache` per-role with 90-day TTL on disk (`data/salary_cache/`)
- Multi-tier fallback: cache → research → static IT defaults → generic
- Sources cited per band, confidence scored, warnings flagged
- UI badges show salary_source + confidence; sources expandable
- `explainer_model` upgraded gpt-4o-mini → gpt-5-mini for better Czech

---

## What's still MVP

These are knowable limits — documented, not hidden.

### Eval coverage
- **Current:** 6 fixtures (5 synthetic + 1 real-world)
- **Production:** 100+ real anonymized CVs with human-labeled ground truth scores and salary expectations
- **Why MVP:** statistical claims about precision/recall require larger sample. 6 fixtures verify "no crash + within plausibility" not "scoring accuracy."

### Salary data ground truth
- **Current:** dynamic Tavily search + `gpt-5-mini` extraction, 90-day disk cache, static IT fallback. Per-role on-demand; first run pays research cost (~$0.005-$0.02), subsequent runs free from cache.
- **Production:** multi-source consensus (multiple search providers), automated refresh of stale cache, NACE/ISCO code mapping for CSU.gov.cz integration, regional breakdowns (Praha/Brno/Olomouc), per-industry adjustments.
- **Why MVP:** Tavily free tier (1000 credits/month) sufficient for demo + early production. Multi-source consensus and regional/industry breakdowns add complexity that doesn't serve the demo narrative.

### Schema versioning
- **Current:** `src/schemas.py` invariant-frozen, no migration tests
- **Production:** schema version field + backward-compat migration helpers + serialized fixture corpus per version
- **Why MVP:** single-version deployment, no historical data to migrate

### Auth / multi-tenant / rate limits
- **Current:** single-user demo, no authentication
- **Production:** OAuth + API keys + rate limits per tenant + audit logging of CV uploads (PII)
- **Why MVP:** scope explicitly excluded; demo runs locally

### GDPR compliance
- **Current:** CV uploads ephemeral (temp file deleted after pipeline run); no persistence
- **Production:** explicit consent flow, right-to-erasure, data residency disclosure, DPA with OpenAI
- **Why MVP:** scope explicitly excluded; CV data not persisted

---

## What would need 1–2 weeks

### Continuous eval pipeline (drift detection)
- Nightly run against eval corpus
- Alert on score drift > 5% from baseline
- Confusion matrix tracking for role classification

### A/B prompt testing infrastructure
- Versioned prompts in `prompts/` with explicit `prompt_version` field
- Side-by-side eval comparing two versions
- Statistical significance test before promoting v2 → v1

### Real-CV regression suite
- Anonymized CV donations from collaborators (with consent)
- 20–50 real CVs across seniority levels + industries
- Run on every PR; alert on new validation flag patterns

### Multi-language output toggle
- Currently CS-only output (Phase 17)
- Add `OUTPUT_LANGUAGE=cs|en|de|sk` env or per-request param
- Translate UI chrome similarly

### Salary research multi-source consensus
- Pluggable `SearchProvider` ABC with implementations for Tavily, SerpAPI, Bing
- Consensus algorithm: agree across ≥2 providers within ±15% before flagging high confidence
- Detect outliers and exclude (e.g., one source citing $80k USD for backend in Olomouc)

### Cache management UI
- Streamlit panel for "Browse cached roles" with researched_at + confidence
- "Force refresh" button per role
- Bulk-expire stale cache (older than X days)

---

## What would need 1–2 months

### Production deployment infrastructure
- Containerization (already have Dockerfile/docker-compose for dev)
- Auth layer (OAuth + JWT)
- Rate limiting (Redis-backed)
- Billing integration (Stripe metered billing keyed to OpenAI cost)
- HA deployment (≥ 2 replicas, health checks, graceful degradation)
- Monitoring (Grafana dashboards, PagerDuty integration)

### GDPR compliance audit
- Privacy policy + ToS legal review
- Data Processing Agreement with OpenAI (zero data retention path)
- Right-to-erasure implementation (CV-data hashing + audit trail)
- Compliance documentation for Czech personal data office (ÚOOÚ)

### Quality team / manual eval reviewers
- Hire/contract 2–3 Czech-speaking reviewers
- Eval corpus expansion (100 → 500+ labeled CVs)
- Reviewer agreement metrics (Cohen's kappa)

### Custom fine-tuned model for CZ market
- Collect 1k+ labeled (CV, score, salary) tuples
- Fine-tune GPT-4o-mini or open-weight base on Czech market specifics
- A/B test fine-tuned vs zero-shot

---

## Cost projections at scale

Baseline measurement: $0.0013 per CV (Phase 17 real-world run with cv_real_world.pdf), $0.0078 per `make eval` run (6 fixtures). All numbers via `text-embedding-3-small` for embeddings + `gpt-4o-mini` for parser+explainer.

| Volume | Daily cost | Monthly cost | Notes |
|---|---|---|---|
| 10 CV/day | ~$0.013 | ~$0.40 | Trivial — well within demo budget cap |
| 100 CV/day | ~$0.13 | ~$4 | Comfortable; current `DAILY_API_BUDGET_USD=$5` fits |
| 1,000 CV/day | ~$1.30 | ~$40 | Bump `DAILY_API_BUDGET_USD` → $10. Add monitoring on cost trends |
| 10,000 CV/day | ~$13 | ~$400 | Rate limit considerations: OpenAI tier-2 rate limits. Add fallback model strategy (gpt-4o-mini → gpt-3.5 on rate limit) |

**Scaling concerns at 10k+ CV/day:**
- OpenAI rate limits per tier (tier-2 ~10k RPM)
- `data/.cost_log.jsonl` no rotation — switch to SQLite or external metrics store
- Single-process Pipeline → need worker queue (Celery / RQ)
- Embedding cost grows with chunk count, not just CV count — add embedding cache (Redis) keyed on chunk hash
- structlog stderr → centralized log aggregation (Loki / Cloudwatch)

---

## Trade-offs and explicit non-goals

These are deliberate choices, not gaps:

- **Rule-based scoring over ML** — transparency > accuracy for demo. ML would hide tradeoffs in opaque weights.
- **OpenAI-only provider** — Phase 13 retired Anthropic support. ABC retained for future Ollama / local LLM. Multi-provider increases test surface and audit burden.
- **Czech-language output default** — Phase 17 demo-driven. Toggle for other languages tracked in this roadmap (1–2 weeks).
- **No automatic salary data refresh** — manual CSV import via CLI is intentional. Live scraping adds vendor lock and ToS risk; CSU.gov.cz semi-annual cycle fits real refresh cadence anyway.
- **Embedding validation opt-in** — default off keeps baseline cost predictable. Operator opts in per-run via Pipeline arg.

---

---

Last updated: 2026-05-05.
