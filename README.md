# LeverGuide v2

Decision support for tabular regression problems. Upload a CSV, choose a numeric KPI, assign column roles, run predictive models, review assumption-based effect estimates, and ask a retrieval-grounded Copilot about the completed analysis.

## Links

- Main challenge repository: https://github.com/abdo2006-dev/datathon-CUB-2026
- This implementation repository: https://github.com/abdo2006-dev/lever-guide-v2-2

## What Works Today

| Capability | Current implementation |
|---|---|
| CSV upload | Frontend accepts `.csv` files up to about 5 MB. The file is held in `sessionStorage` and sent as a single JSON request body, which is the real ceiling. |
| Demo dataset | Injection-molding demo with 5,000 rows and 33 columns. Backend samples to 2,000 rows for analysis. |
| Task type | Regression only. Classification is not exposed in schemas, UI, or docs yet. |
| Column roles | `outcome`, `controllable`, `planning_lever`, `confounder`, `mediator`, `context`, `identifier`, `ignore`, `unassigned`. Uploaded columns default to `unassigned`: they are used as predictors and never as adjusters. |
| Analysis modes | `causal` (adjusted effect estimates plus screened what-if simulations) and `descriptive_predictive` (prediction and description only, no causal claims, no treatment required). |
| Demo ontology | Roles, per-lever adjustment sets, physical bounds and coupling constraints for the curated demo are declared in `apps/api/app/ontology/` and generated into the frontend. See [docs/implementation/PHASE_1A_TRUTH_IN_LABELLING.md](docs/implementation/PHASE_1A_TRUTH_IN_LABELLING.md). |
| Predictive models | OLS, Ridge and Random Forest always; XGBoost and LightGBM where their native libraries are importable. Every configured model reports a status — `succeeded`, `unavailable_dependency`, `training_failed`, `skipped_by_configuration` — so a model that did not run is never silently omitted. |
| Cross-validation | 3-fold CV R2 for OLS/Ridge/RF when enabled and enough rows are available. XGBoost and LightGBM report held-out metrics only. |
| Effect estimates | Adjusted OLS for numeric lever variables, under a declared or assumed causal graph. **Adjusted observational effect estimates**, not proof of causality: interpretation depends on the graph and on there being no important unmeasured confounding. |
| DAG handling | API accepts optional DAG edges, validates them, and rejects invalid DAGs. A dataset with a declared ontology uses that ontology's graph; otherwise a default graph is assumed from the column roles. Every response reports `dag_source`. There is no visual DAG editor yet. |
| What-if simulations | **Predictive what-if simulations** from a GradientBoostingRegressor: one input is changed and predictions are compared. Screened for physical feasibility, observed support, and agreement with the adjusted estimate — only candidates that pass all three are ranked. The rest are returned with the reason they were set aside. |
| Uncertainty | Effect estimates carry OLS confidence intervals under homoskedasticity. Simulations carry a row-resampling interval that holds the fitted model fixed. Where an interval cannot be computed, the bounds are `null` with a stated reason. |
| Executive summary | Generated from model, effect-estimate, intervention, and warning outputs. |
| Analysis Copilot | Optional lightweight RAG assistant at `POST /api/copilot/ask`, grounded in indexed analysis artifacts, stored in Qdrant, and powered by Groq when configured. |
| Experiment tracking | Optional Weights & Biases tracking for dataset profile, model metrics, and analysis artifacts from each `/api/analyze` run. |

## Architecture

```text
apps/web
  Next.js static frontend
  setup page: upload CSV, select target, assign roles, run analysis
  analyze page: result tabs and Analysis Copilot panel
  lib/api-client.ts: typed fetch wrapper with AbortSignal support

apps/api
  FastAPI backend
  app/routers/analysis.py: POST /api/analyze and POST /api/copilot/ask
  app/ontology/: declared dataset ontologies — roles, adjustment sets,
    bounds, coupling identities. Single source of truth; the frontend's
    copy is generated from it by scripts/export_ontology.py
  app/models/pipeline.py: regression model comparison and per-model status
  app/models/causal.py: adjusted OLS effect estimates
  app/models/intervention.py: predictive what-if simulation and screening
  app/models/feasibility.py: feasibility and observed-support checking
  app/rag.py: artifact corpus, Qdrant-backed retrieval, Groq generation
  app/utils/dag.py: DAG validation and adjustment-set helpers
```

The root `render.yaml` builds the Next static export and serves it from FastAPI as one Render service. `apps/api/render.yaml` is available for an API-only deployment.

## Backend Behavior

`POST /api/analyze` runs the full analysis pipeline:

1. Parse CSV with pandas.
2. Validate target and minimum row count. The target must coerce to a numeric regression target with at least 30 non-missing numeric values and more than one distinct value.
3. Sample datasets larger than 2,000 rows with `random_seed`.
4. Resolve a curated ontology if the dataset matches one closely enough; otherwise take the generic path with no domain claims.
5. Assign column roles from the request. A column the request does not mention is `unassigned` — a predictor, never an adjuster.
6. Validate the analysis configuration and stop with `422 INVALID_ANALYSIS_CONFIGURATION` if a causal question was requested without a treatment. The response names the problem, the remedy, and the columns involved.
7. Take the causal graph from the request, the ontology, or the role template, and record which.
8. Validate the DAG and stop with `422 INVALID_DAG` if it is cyclic, malformed, or references unknown columns.
9. Build the feature matrix.
10. Train regression models and record a status for each configured model.
11. In causal mode, run adjusted OLS effect estimation using declared or derived adjustment sets, then run and screen what-if simulations.
12. Generate EDA summaries, executive summary, provenance, and a Copilot retrieval index in Qdrant.
13. Optionally log run metrics and artifacts to Weights & Biases.

### Predictive Models

All models use the same train/test split. The test fraction is between 10% and 20%, depending on dataset size.

| Model | Current settings |
|---|---|
| OLS | `statsmodels.OLS` with intercept. |
| Ridge | `alpha=1.0`. |
| Random Forest | `n_estimators=100`, `max_depth=6`, `min_samples_leaf=15`, `n_jobs=-1`. |
| XGBoost | `n_estimators=150`, `learning_rate=0.08`, `max_depth=4`, `min_child_weight=15`, `subsample=0.8`, `colsample_bytree=0.8`, `n_jobs=1`. |
| LightGBM | `n_estimators=150`, `learning_rate=0.08`, `max_depth=4`, `num_leaves=20`, `min_child_samples=20`, `subsample=0.8`, `colsample_bytree=0.8`, `n_jobs=1`. |

Metrics are regression metrics: R2, adjusted R2 where applicable, RMSE, MAE, optional CV R2, train rows, and test rows.

The split is random — not grouped by any entity and not time-ordered — and the
winning model is selected on the same held-out set its score is reported on.
Both make the headline R2 optimistic; see Limitations.

XGBoost and LightGBM are optional. They need a system OpenMP runtime: the
Dockerfile installs `libgomp1`, and macOS local development needs `libomp`.
Where they cannot be imported the application still runs, and every response
carries a `model_statuses` entry saying which models ran and why the others did
not — so a three-model comparison is never displayed as a five-model one.

### Effect Estimates

For each numeric lever, the backend fits:

```text
target ~ lever + adjustment_set
```

The adjustment set comes from one of two places, and every estimate reports
which via `adjustment_set_source`:

- **`declared_domain_dag`** — a per-lever set declared by the dataset's ontology.
  Used for the curated demo. These are domain claims, checked at test time
  against the declared graph: no adjuster may be a descendant of its cause and no
  mediator may appear.
- **`derived_from_graph`** — observed confounders, graph parents of the cause,
  and context variables; minus mediators, descendants of the cause, the outcome
  and the cause itself. A practical heuristic, used when no ontology applies.

The two are never merged, because mixing a domain claim with a heuristic would
make the reported set untraceable. Mediators are stripped in either case, and
each estimate lists the ones it dropped.

Each estimate returns the coefficient, standard error, t-statistic, p-value,
confidence interval and its method, the estimand, the adjusted-for columns and
their source, the observation count, and an evidence-strength label that is
capped at "weak" whenever the interval includes zero.

These are **adjusted observational effect estimates**. They are useful for
transparent screening, but they are not do-calculus, not causal discovery, and
not guaranteed minimal valid adjustment sets. There are no fixed effects and no
cluster-robust standard errors, so for clustered panel data the intervals shown
are too narrow.

### DAG Validation

Invalid user DAGs never continue into adjustment-set logic. The API returns a structured 422 response:

```json
{
  "detail": {
    "code": "INVALID_DAG",
    "message": "The submitted DAG is invalid. Fix the graph and retry.",
    "errors": ["DAG contains cycles: A -> B -> A"],
    "warnings": []
  }
}
```

When no DAG is provided, the graph comes from one of two places, reported as
`dag_validation.dag_source`:

- **`declared_domain_ontology`** — the dataset matched a curated ontology and
  that ontology's declared graph is used. It contains lever-to-lever structure,
  which the role template cannot express.
- **`assumed_from_roles`** — the fallback template: confounders point to
  controllables and the target, controllables point to the target, context
  variables point to the target.

`dag_validation.valid` means *structurally* valid — acyclic, with every node a
real column. It has never meant scientifically defensible, and the response now
carries `graph_assumption` alongside it saying so. **No causal graph is
discovered from data anywhere in this application.** A visual DAG editor would be
a major product upgrade because the quality of the effect estimates depends on
making these assumptions visible and editable.

### What-If Simulation Engine

A `GradientBoostingRegressor` screens one-feature-at-a-time changes:

- train on the numeric predictive features
- move one lever by about one standard deviation, clipped to declared physical
  bounds and the observed range
- **each row keeps its own observed values for every other column** — only the
  lever changes, and the change is averaged across rows
- attach a row-resampling interval that holds the fitted model fixed
- screen the candidate for feasibility, support and evidence agreement
- rank only the candidates that pass

Every candidate carries a status: `eligible`, `exploratory`, `unsupported`,
`infeasible`, or `conflicting_evidence`. Only `eligible` results receive a rank
and appear in the primary list; the rest are returned as diagnostics with the
reason each was set aside — a rejected candidate is information, not noise.

The feasibility layer checks eligibility, declared physical bounds, observed
support, documented coupling identities between columns, and categorical
validity. On the demo this is what catches an independently changed
`shot_size_g`, which is mechanically determined by cavity count and part weight.

This is useful for prioritization, not operational control. The model is fitted
and evaluated on the same rows, so magnitudes are optimistic. Coupling
constraints are checked but not enforced by construction, and only documented
identities are declared — a variable can still be physically coupled to another
in a way this application does not know about.

## Optional Analysis Copilot

The Copilot is a lightweight grounded explanation layer around completed analysis artifacts. It does not replace the regression, effect-estimation, or intervention engines.

`POST /api/copilot/ask`

```json
{
  "analysis_id": "request_id from /api/analyze",
  "question": "Which levers should I focus on and why?",
  "max_citations": 5
}
```

Response:

```json
{
  "answer": "...",
  "citations": [
    {
      "artifact_id": "interventions",
      "title": "Predictive What-If Simulations",
      "kind": "intervention",
      "snippet": "...",
      "score": 0.42,
      "metadata": { "target": "scrap_rate_pct" }
    }
  ],
  "retrieved_artifact_ids": ["interventions"],
  "model": "llama-3.3-70b-versatile",
  "used_llm": true,
  "warnings": []
}
```

### RAG Design

- Corpus: dataset schema/profile summary, inferred column types and roles, model metrics and per-model run status, adjusted effect estimates, predictive what-if simulations with their status, EDA correlations, the causal graph and its source, and the executive summary. Each artifact leads with its result type and the caveat attached to it, so the Copilot repeats the caveat rather than the headline.
- Retrieval: chunked artifacts are vectorized with scikit-learn `HashingVectorizer` and stored in Qdrant.
- Storage: local Qdrant persistent mode by default through `QDRANT_PATH`; remote Qdrant is supported through env vars.
- Generation: Groq OpenAI-compatible chat completions when `GROQ_API_KEY` is configured.
- Citations: every response returns retrieved snippets and artifact ids.

This is not marketed as a state-of-the-art neural semantic assistant. It is a small, inspectable, retrieval-grounded explanation layer for fresh analysis results. If `GROQ_API_KEY` is not set, retrieval still works and the route returns citations with a retrieval-only message.

## What I Learned / Design Tradeoffs

### FastAPI Instead of Flask

FastAPI fits an API-first ML application because Pydantic request/response models give strong validation, clear typed contracts, and automatic OpenAPI docs. Flask would work, but more of the schema validation, docs, and error-shaping would need to be hand-built.

### Qdrant + Artifact Retrieval Instead of Sending Full Data to the LLM

The Copilot indexes compact analysis artifacts rather than sending a full dataframe to the model. This keeps prompts smaller, lowers latency and cost, avoids exposing raw uploaded data unnecessarily, and makes answers cite the exact model/intervention/summary artifacts they came from.

The tradeoff is that retrieval quality depends on the artifact summaries and hashed bag-of-words vectors. This is lightweight and deployable, but less semantically rich than neural embeddings.

### GradientBoostingRegressor for Counterfactual Screening

Gradient boosting handles nonlinear relationships better than a purely linear model while staying fast enough for a free-tier deployment. It is a good screening model for "what might change if this lever moves?" recommendations.

The tradeoff is realism. The current engine shifts one feature at a time and holds the rest fixed, so it does not fully model coupled process constraints or operational feasibility.

### Adjusted OLS for Transparent Effect Estimates

OLS gives interpretable coefficients, standard errors, p-values, and confidence intervals. That transparency is valuable for a decision-support app because users can see both the estimated direction and the uncertainty.

The tradeoff is that adjusted OLS is only as credible as the observed confounders and DAG assumptions. It is not causal discovery and does not prove that an intervention will work.

### Row Caps for Free-Tier Deployment

The backend samples datasets above 2,000 rows so analysis stays responsive on memory-limited free-tier or low-cost hosts. This keeps the demo practical and prevents large uploads from taking down the service.

The tradeoff is statistical fidelity. For larger production datasets, this should become a background job with larger compute, persisted runs, progress streaming, and configurable sampling.

## Environment Variables

### Backend (`apps/api/.env`)

```bash
ALLOWED_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
APP_ENV=development
STATIC_DIR=../web/out
LOG_LEVEL=INFO

# Optional Copilot generation
GROQ_API_KEY=
GROQ_MODEL=llama-3.3-70b-versatile
GROQ_API_BASE=https://api.groq.com/openai/v1
GROQ_TIMEOUT_SECONDS=30

# Optional RAG index tuning
RAG_INDEX_TTL_SECONDS=21600
RAG_VECTOR_SIZE=4096
RAG_MAX_CONTEXT_CHARS=7000

# Qdrant-backed Copilot storage
QDRANT_URL=
QDRANT_API_KEY=
QDRANT_PATH=./.qdrant
QDRANT_COLLECTION=analysis_copilot
QDRANT_TIMEOUT_SECONDS=10

# Optional Weights & Biases experiment tracking
WANDB_ENABLED=false
WANDB_PROJECT=leverguide-v2
WANDB_ENTITY=
WANDB_MODE=online
```

CORS is environment driven. Local development origins are allowed by default in non-production mode. Wildcard CORS is ignored when `APP_ENV=production`.

### Frontend (`apps/web/.env.local`)

```bash
# For local Next dev against local FastAPI:
NEXT_PUBLIC_API_URL=http://localhost:8000

# For same-origin static export served by FastAPI, leave blank:
# NEXT_PUBLIC_API_URL=
```

## Local Development

### Backend

```bash
cd apps/api
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

API docs are at `http://localhost:8000/api/docs`.

### Frontend

```bash
cd apps/web
npm install
cp .env.example .env.local
# set NEXT_PUBLIC_API_URL=http://localhost:8000 for local Next dev
npm run dev
```

Open `http://localhost:3000`.

### Single-Service Static Build

```bash
cd apps/web
npm install
npm run build

cd ../api
source .venv/bin/activate
STATIC_DIR=../web/out uvicorn app.main:app --reload --port 8000
```

Open `http://localhost:8000`.

## Tests

Backend:

```bash
cd apps/api
source .venv/bin/activate
pytest tests/ -v
```

Backend W&B tracking smoke test:

```bash
cd apps/api
source .venv/bin/activate
pytest tests/test_wandb_tracking.py -v
```

Frontend type check, lint and production build:

```bash
cd apps/web
npm install
npm run type-check
npm run lint
npm run build
```

Regenerate the frontend's copy of the demo ontology after changing
`apps/api/app/ontology/`. The backend test suite fails if the committed JSON has
drifted from the Python source:

```bash
cd apps/api
./.venv/bin/python scripts/export_ontology.py
```

## Scientific and Architectural Audit

A full audit of this repository and of the original analysis it derives from is
in [`docs/audit/`](docs/audit/README.md). It is imported verbatim and pinned to
commit `2bd854f`; it is a record of what was true then, not a live document.

Corrections made in response, with before/after numbers on the shipped demo, are
in
[`docs/implementation/PHASE_1A_TRUTH_IN_LABELLING.md`](docs/implementation/PHASE_1A_TRUTH_IN_LABELLING.md),
which also lists what was deliberately deferred.

## Limitations

- Regression only. Classification support would make the app useful for churn, pass/fail, conversion, default, approval, and defect/no-defect KPIs.
- No visual DAG editor yet. The graph is either declared by a dataset ontology or assumed from column roles. Nothing is discovered from data.
- Effect estimates are observational and may be biased by unobserved confounders. There are no fixed effects and no cluster-robust standard errors, so intervals are too narrow for clustered panel data.
- The train/test split is random — not grouped by any entity and not time-ordered — and the winning model is selected on the same held-out set its score is reported on. Both make the reported R2 optimistic.
- What-if simulations are fitted and evaluated on the same rows. Their intervals resample rows while holding the model fixed, so they exclude model-estimation uncertainty and the true interval is wider.
- Mediator propagation is not implemented: a lever whose effect runs through a mediator is marked exploratory rather than given a number.
- Only documented coupling identities are enforced. A variable can still be physically coupled to another in a way the application does not know about.
- Conditional, cap-only, additive-delta and combined-package interventions are not representable.
- Categorical controllable interventions are not implemented.
- No authentication or multi-user isolation.
- No persistent result history for full analysis runs; frontend state uses session storage, while Copilot retrieval chunks persist in Qdrant storage.
- No job queue or streaming progress; `/api/analyze` is synchronous.
- RAG retrieval uses hashed text vectors plus Qdrant storage, not a neural embedding API.

## Planned / Future Work

- Stable public deployment link.
- Persistent analysis storage and job history.
- Background jobs and streaming progress.
- Visual DAG editor.
- Classification analysis with classifiers, classification metrics, and matching UI.
- Categorical controllable interventions.
- Authentication and multi-user isolation.
- PDF export.
- Future fine-tuning support for preferred explanation style or repeated analyst workflows.
