# LeverGuide v2

Decision support for tabular regression problems. Upload a CSV, choose a numeric KPI, assign column roles, run predictive models, review assumption-based effect estimates, and ask a retrieval-grounded Copilot about the completed analysis.

## Links

- Main challenge repository: https://github.com/abdo2006-dev/datathon-CUB-2026
- This implementation repository: https://github.com/abdo2006-dev/lever-guide-v2-2
- Live demo: not listed as canonical until the deployment is stable. If the GitHub repository sidebar still points to `lever-guide-v2-2.vercel.app`, update the repository Website field to the main challenge repository above.

## What Works Today

| Capability | Current implementation |
|---|---|
| CSV upload | Frontend accepts `.csv` files up to 50 MB. |
| Demo dataset | Injection-molding demo with 5,000 rows and 32 columns. Backend samples to 2,000 rows for analysis. |
| Task type | Regression only. Classification is not exposed in schemas, UI, or docs yet. |
| Column roles | `outcome`, `controllable`, `confounder`, `mediator`, `context`, `identifier`, `ignore`. |
| Predictive models | OLS, Ridge, Random Forest, XGBoost, and LightGBM regressors. Optional native tree libraries are skipped gracefully if unavailable in local development. |
| Cross-validation | 3-fold CV R2 for OLS/Ridge/RF when enabled and enough rows are available. XGBoost and LightGBM report held-out metrics only. |
| Effect estimates | Back-door adjusted OLS for numeric controllable variables. This is observational adjustment, not proof of causality. |
| DAG handling | API accepts optional DAG edges, validates them, and rejects invalid DAGs. If no DAG is supplied, the backend builds a default assumed DAG from the selected column roles. There is no visual DAG editor yet. |
| Interventions | Numeric controllable recommendations from a GradientBoostingRegressor counterfactual screening model, annotated with adjusted OLS evidence when available. |
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
  app/models/pipeline.py: regression model comparison
  app/models/causal.py: adjusted OLS effect estimates
  app/models/intervention.py: counterfactual recommendation engine
  app/rag.py: artifact corpus, Qdrant-backed retrieval, Groq generation
  app/utils/dag.py: DAG validation and adjustment-set helpers
```

The root `render.yaml` builds the Next static export and serves it from FastAPI as one Render service. `apps/api/render.yaml` is available for an API-only deployment.

## Backend Behavior

`POST /api/analyze` runs the full analysis pipeline:

1. Parse CSV with pandas.
2. Validate target and minimum row count. The target must coerce to a numeric regression target with at least 30 non-missing numeric values and more than one distinct value.
3. Sample datasets larger than 2,000 rows with `random_seed`.
4. Assign column roles from the request.
5. Build a default assumed DAG or accept user-provided DAG edges.
6. Validate the DAG and stop with `422 INVALID_DAG` if it is cyclic, malformed, or references unknown columns.
7. Build the feature matrix.
8. Train regression models.
9. Run adjusted OLS effect estimation.
10. Generate intervention recommendations, EDA summaries, executive summary, and a Copilot retrieval index in Qdrant.
11. Optionally log run metrics and artifacts to Weights & Biases.

### Predictive Models

All models use the same train/test split. The test fraction is between 10% and 20%, depending on dataset size.

| Model | Current settings |
|---|---|
| OLS | `statsmodels.OLS` with intercept. |
| Ridge | `alpha=1.0`. |
| Random Forest | `n_estimators=100`, `max_depth=6`, `min_samples_leaf=15`, `n_jobs=-1`. |
| XGBoost | `n_estimators=150`, `learning_rate=0.08`, `max_depth=4`, `min_child_weight=15`, `subsample=0.8`, `colsample_bytree=0.8`, `n_jobs=1`. |
| LightGBM | `n_estimators=150`, `learning_rate=0.08`, `max_depth=4`, `num_leaves=20`, `min_child_samples=20`, `subsample=0.8`, `colsample_bytree=0.8`, `n_jobs=1`. |

Metrics are regression metrics: R2, adjusted R2 where applicable, RMSE, MAE, optional CV R2, train rows, and test rows. The Dockerfile installs `libgomp1` for boosted-tree native runtime support. macOS local development may require `libomp` for XGBoost/LightGBM.

### Effect Estimates

For each numeric controllable variable, the backend fits:

```text
target ~ controllable + confounders + DAG_parents(controllable) + context
```

The adjustment set is a practical heuristic:

- Include observed confounders, DAG parents of the cause, and context variables.
- Exclude mediators, descendants of the cause, the outcome, and the cause itself.
- Return coefficient, standard error, t-statistic, p-value, confidence interval, adjusted-for columns, and evidence-strength labels.

These are observational adjusted associations. They are useful for transparent screening, but they are not do-calculus, not causal discovery, and not guaranteed minimal valid adjustment sets.

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

When no DAG is provided, the backend creates a default assumed DAG:

- confounders point to controllables and the target
- controllables point to the target
- context variables point to the target

This default graph is a convenience assumption based on user roles. It is not learned from data. A visual DAG editor would be a major product upgrade because the quality of the effect estimates depends on making these assumptions visible and editable.

### Intervention Engine

The intervention engine uses a `GradientBoostingRegressor` to screen one-feature-at-a-time changes:

- train on numeric predictive features
- shift one numeric controllable by about one standard deviation
- hold the rest of the feature matrix fixed
- clip suggested values using distribution-based bounds
- rank recommendations by estimated KPI change
- attach adjusted OLS evidence when available

This is useful for prioritization, not operational control. It can propose unrealistic changes when variables are physically or operationally coupled because it does not enforce domain-specific constraints beyond simple clipping.

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
      "title": "Intervention Recommendations",
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

- Corpus: dataset schema/profile summary, inferred column types and roles, model metrics, effect estimates, intervention recommendations, EDA correlations, DAG validation, and executive summary.
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

Frontend type check:

```bash
cd apps/web
npm install
npm run type-check
```

## Repository Polish Notes

The source files in this local repository are normal multi-line files, not minified one-line blobs. Useful checks:

```bash
wc -l apps/api/app/models/pipeline.py \
  apps/api/app/models/causal.py \
  apps/api/app/models/intervention.py \
  apps/api/app/routers/analysis.py \
  apps/web/package.json
```

Expected current result is approximately:

```text
174 apps/api/app/models/pipeline.py
131 apps/api/app/models/causal.py
180 apps/api/app/models/intervention.py
453 apps/api/app/routers/analysis.py
 57 apps/web/package.json
```

If GitHub raw view shows these as extremely long one-line files, check that the pushed branch matches this local copy and that no generated or copied version replaced the formatted source.

## Limitations

- Regression only. Classification support would make the app useful for churn, pass/fail, conversion, default, approval, and defect/no-defect KPIs.
- No visual DAG editor yet. The default DAG is an assumption generated from column roles, not causal discovery.
- Effect estimates are observational and may be biased by unobserved confounders.
- The intervention engine shifts one numeric feature at a time and does not enforce domain-specific process constraints.
- Categorical controllable interventions are not implemented.
- No authentication or multi-user isolation.
- No persistent result history for full analysis runs; frontend state uses session storage, while Copilot retrieval chunks persist in Qdrant storage.
- No job queue or streaming progress; `/api/analyze` is synchronous.
- RAG retrieval uses hashed text vectors plus Qdrant storage, not a neural embedding API.

## Show-Off Demo Flow

1. Start backend and frontend.
2. Open the app, load the demo dataset, assign roles, and run analysis.
3. Capture the Predictive Models screen showing model comparison, metrics, feature importance, and actual-vs-predicted scatter.
4. Capture the Copilot screen showing a grounded answer with citations.
5. Optional: capture W&B run overview/config, model metrics table, and artifacts.
6. Optional: capture a terminal verification that Qdrant collection `analysis_copilot` exists and has indexed points.

For LinkedIn or portfolio screenshots, prioritize the polished UI screens. Terminal screenshots are useful proof, but they should come after the product screenshots.

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
