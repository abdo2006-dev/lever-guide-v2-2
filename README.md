# LeverGuide v2

Decision intelligence for tabular regression problems. Upload a CSV, choose a numeric KPI, assign column roles, run predictive models and DAG-aware causal analysis, then review intervention recommendations and an executive summary.

## What Works Today

| Capability | Current implementation |
|---|---|
| CSV upload | Frontend accepts `.csv` files up to 50 MB. |
| Demo dataset | Injection-moulding demo with 5,000 rows and 32 columns. Backend samples to 2,000 rows for analysis. |
| Task type | Regression only. Classification is not exposed in schemas, UI, or docs. |
| Column roles | `outcome`, `controllable`, `confounder`, `mediator`, `context`, `identifier`, `ignore`. |
| Predictive models | OLS, Ridge, Random Forest, XGBoost, and LightGBM regressors. If a native boosted-tree library is unavailable in local dev, that model is skipped without taking down the API. |
| Cross-validation | 3-fold CV R² for OLS/Ridge/RF when enabled and enough rows are available. XGBoost and LightGBM report held-out metrics only. |
| Causal analysis | Back-door adjusted OLS for numeric controllable variables. |
| DAG handling | API accepts optional DAG edges, validates them, and rejects invalid DAGs before causal analysis. There is no visual DAG editor in the frontend yet. |
| Interventions | Numeric controllable recommendations from a GradientBoostingRegressor counterfactual simulation, annotated with causal evidence when available. |
| Executive summary | Generated from model, causal, and intervention outputs. |
| Analysis Copilot | Optional RAG assistant at `POST /api/copilot/ask`, grounded in indexed analysis artifacts, stored in Qdrant, and powered by Groq when configured. |

## Architecture

```
apps/web
  Next.js static frontend
  setup page: upload CSV, select target, assign roles, run analysis
  analyze page: result tabs and Analysis Copilot panel
  lib/api-client.ts: typed fetch wrapper with AbortSignal support

apps/api
  FastAPI backend
  app/routers/analysis.py: POST /api/analyze and POST /api/copilot/ask
  app/models/pipeline.py: regression model comparison
  app/models/causal.py: adjusted OLS causal estimates
  app/models/intervention.py: counterfactual recommendation engine
  app/rag.py: artifact corpus, Qdrant-backed retrieval, Groq generation
  app/utils/dag.py: DAG validation and adjustment-set helpers
```

The root `render.yaml` builds the Next static export and serves it from FastAPI as one Render service. `apps/api/render.yaml` is available for an API-only deployment.

## Backend Behavior

### Analysis Flow

`POST /api/analyze` runs:

1. Parse CSV with pandas.
2. Validate target and minimum row count (`>= 30` rows). The target must coerce to a numeric regression target with at least 30 non-missing numeric values and more than one distinct value.
3. Sample datasets larger than 2,000 rows with `random_seed`.
4. Assign roles from the request.
5. Build or accept DAG edges.
6. Validate the DAG and stop with `422 INVALID_DAG` if it is cyclic, malformed, or references unknown columns.
7. Build the feature matrix.
8. Train regression models.
9. Run causal analysis.
10. Generate interventions, EDA summaries, executive summary, and a Copilot retrieval index in Qdrant.

### Predictive Models

All models use the same train/test split. The test fraction is between 10% and 20%, depending on dataset size.

| Model | Current settings |
|---|---|
| OLS | `statsmodels.OLS` with intercept. |
| Ridge | `alpha=1.0`. |
| Random Forest | `n_estimators=100`, `max_depth=6`, `min_samples_leaf=15`, `n_jobs=-1`. |
| XGBoost | `n_estimators=150`, `learning_rate=0.08`, `max_depth=4`, `min_child_weight=15`, `subsample=0.8`, `colsample_bytree=0.8`, `n_jobs=1`. |
| LightGBM | `n_estimators=150`, `learning_rate=0.08`, `max_depth=4`, `num_leaves=20`, `min_child_samples=20`, `subsample=0.8`, `colsample_bytree=0.8`, `n_jobs=1`. |

Metrics are regression metrics: R², adjusted R² where applicable, RMSE, MAE, optional CV R², train rows, and test rows. The Dockerfile installs `libgomp1` for boosted-tree native runtime support; macOS local development may require `libomp` for XGBoost/LightGBM to run.

### Causal Analysis

For each numeric controllable variable, the backend fits:

```text
target ~ controllable + confounders + DAG_parents(controllable) + context
```

Mediators and descendants of the controllable are excluded from adjustment sets. Results are observational adjusted associations with standard errors, confidence intervals, p-values, evidence strength labels, and caveats. They are not proof of causality.

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

The DAG utility helpers also return empty sets for missing graph nodes instead of surfacing raw NetworkX exceptions.

## Optional Analysis Copilot

The Copilot is a RAG explanation layer around completed analysis artifacts. It does not replace the regression, causal, or intervention engines.

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

- Corpus: dataset schema/profile summary, inferred column types and roles, model metrics, causal findings, intervention recommendations, EDA correlations, DAG validation, and executive summary.
- Retrieval: chunked artifacts are vectorized with scikit-learn `HashingVectorizer` and stored in Qdrant. By default the backend uses Qdrant local persistent mode on disk; you can point it at a remote Qdrant cluster with env vars.
- Generation: Groq OpenAI-compatible chat completions. Provider details are centralized in `app/rag.py` and configured by env.
- Citations: every response returns retrieved snippets and artifact ids.
- Storage: Qdrant-backed with TTL pruning. Local mode persists on the local filesystem path you configure; use a remote Qdrant cluster for durable shared deployment storage.

If `GROQ_API_KEY` is not set, retrieval still works and the route returns citations with a retrieval-only message.

## RAG Before Fine-Tuning

RAG grounds answers in project-specific artifacts from the current analysis session: metrics, recommendations, causal caveats, and dataset summaries. Fine-tuning would adapt behavior, style, or repeated task patterns, but it would not automatically know a user's newly uploaded dataset or generated results.

For this app, RAG is the correct first step because users need grounded explanations of fresh analysis outputs. Future fine-tuning can be added later because prompt assembly, retrieval, and provider calls are already separated.

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

Frontend type check:

```bash
cd apps/web
npm install
npm run type-check
```

## Limitations

- Regression only.
- No visual DAG editor in the frontend.
- No authentication.
- No persistent result storage for full analysis history; frontend state uses session storage, while Copilot retrieval chunks persist in Qdrant storage.
- No job queue or streaming progress; `/api/analyze` is synchronous.
- Categorical controllable interventions are not implemented.
- Causal estimates are observational and may be biased by unobserved confounders.
- RAG retrieval uses hashed text vectors plus Qdrant storage, not a neural embedding API.

## Planned / Future Work

- Persistent analysis storage and job history.
- Background jobs and streaming progress.
- Visual DAG editor.
- Classification analysis with real classifiers, classification metrics, and matching UI.
- Categorical controllable interventions.
- Authentication and multi-user isolation.
- PDF export.
- Future fine-tuning support for preferred explanation style or repeated analyst workflows.
