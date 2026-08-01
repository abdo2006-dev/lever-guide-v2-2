# LeverGuide v2 — Current Architecture

**Source audited:** `/Users/abdulrahmanahmad/Desktop/My Projects/lever-guide-v2 2/`
(git `2bd854f`, remote `github.com/abdo2006-dev/lever-guide-v2-2`).
Verified byte-identical in source to `~/Downloads/lever-guide-v2-2-main.zip` via `diff -rq`
(only `__pycache__`, `.next`, `out/`, `.qdrant`, `.env` and stray dirs differ).

**Total hand-written source: 5,835 lines** across 24 Python/TS/TSX files.

---

## 1. Runtime data flow, end to end

### Path A — user clicks "Try Demo Dataset"

```
 BROWSER                                                    │ SERVER
────────────────────────────────────────────────────────────┼──────────────────────────
 1. HomeClient.handleDemo()            HomeClient.tsx:47    │
 2. fetch("/demo/injection_molding_demo.csv")               │ StaticFiles mount
    → 1.13 MB, 5,000 rows                 csv.ts:155        │   main.py:100
 3. Papa.parse(header:true, dynamicTyping:false)            │
    → per-column kind inference, mean/σ/p25/p75 in JS       │
                                          csv.ts:107-143    │
 4. DEMO_ROLES applied (frontend copy)    csv.ts:6-40       │
 5. zustand store.setDataset(...)         store.ts:36       │
    → persisted to sessionStorage INCLUDING full csv_content│
                                          store.ts:63-78    │
 6. router.push("/setup")                                   │
 7. user may re-assign roles via <select> setup/page.tsx:336│
 8. handleAnalyze(): POST /api/analyze                      │
    body = { dataset_csv: <entire 1.13 MB CSV as a JSON     │
             string>, target, task, improve_direction,      │
             column_roles, dag_edges: [], random_seed:42 }  │
    90 s AbortController              setup/page.tsx:138-151│
                                                            │
                                                            │  9. pd.read_csv(StringIO)   analysis.py:244
                                                            │ 10. target coercion + 30-row / variance
                                                            │     guards                  analysis.py:98-117
                                                            │ 11. df.sample(2000, random_state=seed)
                                                            │     ── 60 % of the demo silently discarded
                                                            │                             analysis.py:258-259
                                                            │ 12. _assign_roles() — server-side DEMO_ROLES
                                                            │     is NOT used here; roles come from the
                                                            │     request. Unlisted cols → "confounder"
                                                            │                             analysis.py:80-95
                                                            │ 13. auto_dag() — dag_edges is always []
                                                            │     → 60 template edges     dag.py:132-154
                                                            │ 14. validate_dag(); 422 INVALID_DAG on fail
                                                            │                             dag.py:18-66
                                                            │ 15. build_feature_matrix(standardize=True)
                                                            │     ct.fit_transform on ALL rows
                                                            │                       preprocess.py:119
                                                            │ 16. run_predictive_pipeline() — split, fit
                                                            │     ≤5 models, sort by TEST r2, mark winner
                                                            │                          pipeline.py:46-174
                                                            │ 17. run_causal_analysis() — one OLS per
                                                            │     controllable          causal.py:31-131
                                                            │ 18. run_intervention_engine() — 2nd GBR fit
                                                            │     on ALL rows       intervention.py:79-84
                                                            │ 19. _compute_correlations / _compute_
                                                            │     distributions        analysis.py:120-173
                                                            │ 20. _build_executive() — templated strings
                                                            │                          analysis.py:176-233
                                                            │ 21. index_analysis_session() → Qdrant
                                                            │     (best-effort, warning on failure)
                                                            │                              rag.py:302-335
                                                            │ 22. track_analysis_run() → W&B if enabled
                                                            │                    wandb_tracking.py:35-175
 23. store.setAnalysis(bundle) → sessionStorage             │  ← AnalysisBundle JSON
 24. router.push("/analyze"); 6 tabs render from the bundle │
     analyze/page.tsx (705 lines, all tabs inline)          │
```

### Path B — user uploads their own CSV

Identical from step 3, except `inferDefaultRole()` (`csv.ts:92-98`) assigns every non-ID, non-text
column the role **`confounder`**. Since `/api/analyze` returns 422 when no column is `controllable`
(`analysis.py:272-276`), a fresh upload **always fails** until the user manually re-labels columns.
There is no auto-detection of levers and no guidance beyond a one-line hint.

### Path C — Copilot question

`analyze/page.tsx:569 askCopilot()` → `POST /api/copilot/ask` → `retrieve()` hashes the question
with `HashingVectorizer` and does a cosine query against Qdrant filtered by `analysis_id`
(`rag.py:338-366`) → `answer_with_groq()` posts the retrieved snippets to
`api.groq.com/openai/v1/chat/completions` (`rag.py:408-428`). Without `GROQ_API_KEY` it degrades to
a retrieval-only canned string (`rag.py:391-398`).

---

## 2. Responsibility split

| Concern | Where it runs | Evidence |
|---|---|---|
| CSV parsing (twice — once JS, once pandas) | Browser **and** server | `csv.ts:107`, `analysis.py:244` |
| Column-kind inference (two independent implementations that can disagree) | Browser `csv.ts:43-52` (80 % numeric heuristic, 30-cat cap) / Server `preprocess.py:18-26` (dtype-based, 30-cat cap) | — |
| Descriptive stats | Both — browser for the setup table, server for `distributions`/`correlations` | `csv.ts:70-88`, `analysis.py:140-173` |
| Row sampling to 2,000 | Server only, **after** transferring the full file | `analysis.py:258` |
| Imputation, encoding, scaling | Server, `ColumnTransformer` fit on the full frame | `preprocess.py:100-119` |
| Model fitting | Server, synchronous, inside the request | `pipeline.py`, `intervention.py` |
| Serialization | Pydantic v2 `AnalysisBundle` → JSON | `schemas.py` |
| State | `sessionStorage` only, key `leverguide-state` | `store.ts:62-79` |
| Raw data retention | **Browser sessionStorage holds the whole CSV.** Server keeps it only for the request lifetime; nothing is written to disk. | `store.ts:77` |
| Derived data retention | Qdrant on the server disk at `./.qdrant`, 6-hour TTL | `rag.py:47, 273-279, 452-467` |
| External calls | Groq (only when `GROQ_API_KEY` set), W&B (only when `WANDB_ENABLED=true`) | `rag.py:369`, `wandb_tracking.py:52` |

### Where raw data goes

* The **full CSV** is base-64-free JSON-escaped into a POST body — a 50 MB file would become a
  ~50 MB request body against a Render free tier with ~512 MB RAM.
* Column-level **summary statistics** (mean, σ, p25/p75, top-5 categorical values, per-column
  missing counts) are written into the Qdrant `dataset_profile` artifact (`rag.py:68-90`) and can be
  **sent to Groq** as retrieval context (`rag.py:374-382`). Raw rows are never sent. This is the
  correct design and is described accurately in the README, but it is still user data crossing a
  third-party boundary and is not surfaced in the UI.

---

## 3. API surface

| Route | Handler | Notes |
|---|---|---|
| `POST /api/analyze` | `analysis.py:236` | Synchronous. Full pipeline. Returns `AnalysisBundle`. |
| `POST /api/copilot/ask` | `analysis.py:407` | 404 `ANALYSIS_NOT_INDEXED`, 502 `COPILOT_GENERATION_FAILED`. |
| `GET /health` | `main.py:87` | `{"status":"ok","version":"2.0.0"}` |
| `GET /`, `/setup`, `/analyze` | `main.py:106-118` | Serve the static export when `STATIC_DIR` exists. |
| `GET /_next/*`, `/demo/*` | `main.py:96-100` | `StaticFiles` mounts. |
| `GET /api/docs` | FastAPI | OpenAPI UI. |

`AnalysisBundle` (`schemas.py`) carries: `predictive[]` (per model: metrics, ≤20 importances, ≤400
prediction points, coefficients for OLS), `causal[]`, `interventions[]`, ≤100 `correlations`,
`distributions` for the first 30 columns, `executive`, `dag_validation`, `warnings[]`. On the demo
this is a few hundred KB of JSON.

---

## 4. Frontend structure

```
apps/web/app/page.tsx          5 lines  → renders HomeClient
apps/web/components/HomeClient.tsx    154
apps/web/app/setup/page.tsx           483   4-step wizard
apps/web/app/analyze/page.tsx         705   ← ALL SIX TABS DEFINED INLINE
apps/web/components/analyze/          ← DEAD CODE, imported by nothing
   CausalTab.tsx        127
   PredictiveTab.tsx    145
   InterventionsTab.tsx 150
   ExecutiveTab.tsx     120
   EdaTab.tsx             1  ← empty stub
apps/web/lib/{store,csv,types,api-client}.ts
```

Confirmed by grep: no file imports anything from `components/analyze/`. `analyze/page.tsx` defines
its own `PredictiveTab`, `CausalTab`, `InterventionsTab`, `ExecutiveTab`, `CopilotTab`, plus shared
`Card`/`KpiCard`/`MetCard`/`Row`/`Badge`/`Empty` primitives. **There is no EDA tab in the UI at
all**, so the `correlations` and `distributions` the backend computes and serialises on every
request are never rendered.

### Dependency audit

`npm ls`-free static check — packages declared in `apps/web/package.json` with **zero** imports in
`app/`, `components/`, `lib/`:

* all **16 `@radix-ui/*` packages**
* **`reactflow`** ^11.11.4 — declared, never imported. **There is no DAG editor.**
* `zod`, `@tanstack/react-query`, `class-variance-authority`, `tailwind-merge`, `clsx`

Actually used: `next`, `react`, `react-dom`, `recharts` (3 files), `lucide-react` (3),
`sonner` (2), `zustand` (1), `papaparse` (1), `next-themes` (1), `tailwindcss-animate` (config).
**23 of 31 runtime dependencies are unused.**

Consequence for the DAG: `store.dagEdges` is initialised `[]` (`store.ts:28`), `setDagEdges` is
never called by any component, and `setup/page.tsx:149` therefore always sends `dag_edges: []`.
**Every analysis in the deployed app runs on the machine-generated `auto_dag` template.**

---

## 5. `auto_dag` — what the "causal graph" actually is

`dag.py:132-154`. Given roles it emits, unconditionally:

```
for cf in confounders:  cf → every controllable ;  cf → target
for c  in controllable: c  → target
for cx in context:      cx → target
```

On the demo this is **60 edges** and it is a complete bipartite template, not a domain model. It
encodes no lever-to-lever structure, so it cannot represent `mold_temperature_c → cooling_time_s`
(the reverse-causation mechanism that is the Datathon's entire headline finding), nor any mediator
chain.

`adjustment_set()` (`dag.py:93-129`) then largely ignores the graph: it takes
`confounders ∪ DAG-parents(cause) ∪ context`, minus mediators, minus descendants of the cause. Under
`auto_dag` the parents of a controllable are exactly the confounders, so the adjustment set reduces
to **"every confounder + every context variable, identically for every lever."** Verified on the
demo: the sets for `cooling_time_s`, `mold_temperature_c` and `injection_pressure_bar` are the same
12 columns, and `mold_temperature_c` is absent from all of them.

---

## 6. Deployment

| Artefact | Content | Problem |
|---|---|---|
| `render.yaml` (root) | one Python web service; `pip install -r apps/api/requirements.txt && cd apps/web && npm install && npm run build`; `uvicorn app.main:app`; `STATIC_DIR=../web/out`; `ALLOWED_ORIGINS=https://lever-guide.onrender.com` | plan unset → free tier |
| `apps/api/render.yaml` | **separate API-only service**, `plan: starter` (**$7/mo, not free**), `region: frankfurt`, `ALLOWED_ORIGINS=https://lever-guide.vercel.app` | Contradicts the root file: different topology, different origin, paid plan. Nothing states which is authoritative. |
| `Dockerfile` (root) | 2-stage node:20 → python:3.12-slim, installs `libgomp1`, `STATIC_DIR=/app/web/out` | Correct, but **unused** — neither render.yaml uses `runtime: docker`. |
| `apps/api/Dockerfile` | second Dockerfile | third deployment description |
| `next.config.ts` | `output: "export"`, `trailingSlash: true` | Static export; consistent with FastAPI serving it. |

`requirements.txt` pins everything except `wandb` (unpinned). Python 3.12 in Render vs **3.10.4**
in the committed local venv.

---

## 7. Error handling and optional dependencies

| Mechanism | Location | Assessment |
|---|---|---|
| CSV parse, missing target, <30 rows, non-numeric/constant target, no controllable, invalid DAG | `analysis.py:243-298` | Good — structured 422s with actionable messages. |
| Frontend 90 s / 45 s `AbortController` + typed `ApiError` | `setup/page.tsx:138`, `analyze/page.tsx:567`, `api-client.ts` | Good. |
| Qdrant indexing failure | `analysis.py:382-386` | Caught, surfaced as a user-visible warning. Good. |
| W&B failure | `analysis.py:388-402` | Returns a warning string rather than raising. Good. |
| **Per-model failure** | `pipeline.py:86, 105, 126, 145, 166` — five bare `except Exception: pass` | **Bad.** Silent. |
| `xgboost` / `lightgbm` unavailable | `pipeline.py:15-23` set to `None`, then swallowed by the same bare except | **Bad.** No warning reaches the bundle. |
| All models fail | `pipeline.py:169-170` raises → 500 | Only failure that is visible. |

**Verified live in this environment:** both optional boosters fail to import in the committed venv —
`xgboost` raises `XGBoostError: libxgboost.dylib could not be loaded … libomp.dylib`, `lightgbm`
raises `OSError: Library not loaded: @rpath/libomp.dylib`. The pipeline returned **3 of 5 models**
(`rf`, `ridge`, `ols`) and the bundle contained **no warning**, while `setup/page.tsx:411` continued
to display "Running: OLS · Ridge · Random Forest · XGBoost · LightGBM".

---

## 8. Tests

`apps/api/tests/` — 27 tests, **all passing** (`pytest -q` → `27 passed in 23.72s`).

Coverage is entirely structural: DAG cycle/malformed/unknown-node validation (6), adjustment-set
mediator exclusion on a synthetic 4-column frame (2), feature-matrix shape/NaN (3), pipeline returns
results and winner has best R² (2), causal analysis runs (2), endpoint happy path and 422s (7),
Copilot retrieval/persistence/404 (3), health (1), W&B enabled/disabled (2).

**Zero scientific tests.** Nothing asserts a sign, a magnitude, an adjustment-set membership on the
real demo data, support bounds on an intervention, or absence of leakage. The suite passes with
both boosted-tree models broken, which is itself evidence that it does not test what the product
claims to do. No frontend tests. No CI configuration anywhere in the repository.

---

## 9. Repository hygiene

* **Stray brace-expansion directories** committed to the working tree from a failed `mkdir -p`:
  `{apps/`, `{apps/web,apps/`, `{apps/web,apps/api}`, `apps/web/{app/...`, `apps/api/{app/...`.
  Untracked by git (`git ls-files` returns none) but present on disk and inside the zip's sibling.
* `apps/web/out/` (built static export) and `apps/api/.qdrant/` (SQLite index) are present on disk;
  both are correctly gitignored.
* `apps/api/.env` exists locally and **is correctly gitignored** (`git check-ignore` confirms). It
  contains a real `GROQ_API_KEY`. Not committed — no secret exposure in the repository. Its contents
  were not read or reproduced during this audit.
* `apps/api/wandb/offline-run-.../logs/debug.log` is committed in the zip — a stray run artifact.
* Two `.venv` directories on disk (root and `apps/api/`).

## 10. Field-name mismatch in W&B logging (confirmed bug)

`wandb_tracking.py:107-112` reads `metrics.get("adjusted_r2")`, `metrics.get("train_rows")`,
`metrics.get("test_rows")`. The `ModelMetrics` model (`schemas.py:97-106`) defines `adj_r2`,
`n_train`, `n_test`. All three W&B columns are therefore always `None`, silently. The two W&B tests
pass because they assert on a stub, not on real `ModelMetrics` field names.
