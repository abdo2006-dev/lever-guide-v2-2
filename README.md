# LeverGuide v2

**Decision intelligence for tabular data.** Upload a dataset, select a KPI, define your causal graph, and get ranked, explainable recommendations — with predictive *and* causal evidence shown side by side.

---

## Overview

Most analytics tools answer "what predicts my KPI?"  
LeverGuide answers "what should I *change*, and by how much?"

It does this by combining:

1. **Predictive modelling** — five models (OLS, Ridge, Random Forest, XGBoost, LightGBM) trained and compared on a held-out test set. The best model is selected by test-set R².
2. **DAG-aware causal analysis** — you assign column roles (controllable / confounder / mediator / context) and optionally draw a causal graph. The engine derives the back-door adjustment set and estimates adjusted OLS coefficients via statsmodels with proper standard errors and confidence intervals.
3. **Intervention recommendations** — for each controllable variable, a counterfactual simulation estimates the expected KPI change from a targeted shift, annotated with evidence type, strength, tradeoffs, assumptions, and an honest caveat.

---

## Problem Statement

Business analysts routinely face this situation: they have a dataset, a KPI they want to improve, and a set of variables they can actually change. Standard ML tells them what correlates — not what to do. Causal inference tools are often inaccessible without a PhD. LeverGuide bridges this gap for teams that need actionable, explainable decisions from tabular data.

---

## Product Capabilities

| Feature | Detail |
|---|---|
| Upload any CSV | Up to 50,000 rows, numeric + categorical columns |
| Demo dataset | Injection-moulding plant data — 5,000 rows, 32 columns |
| Column roles | outcome / controllable / confounder / mediator / context / identifier / ignore |
| Five-model comparison | OLS · Ridge · Random Forest · XGBoost · LightGBM |
| Cross-validation | 5-fold CV R² reported where n > 200 |
| Causal adjustment | Back-door adjusted OLS via statsmodels, per-variable adjustment sets |
| DAG editor | Drag-and-drop causal graph with cycle detection and validation |
| Interventions | Ranked recommendations with direction, magnitude, evidence type, tradeoffs |
| EDA | Histograms and categorical distributions for all model features |
| Executive summary | Plain-language summary with honest caveats for non-technical stakeholders |
| Dark / light mode | System-aware theme, switchable |

---

## Architecture

LeverGuide v2 uses a **split deployment architecture**:

```
┌─────────────────────────────────────────────────┐
│  Vercel (Next.js 15 App Router)                  │
│                                                   │
│  app/page.tsx        → Landing page               │
│  app/setup/page.tsx  → Upload + column roles      │
│  app/analyze/page.tsx→ Results dashboard          │
│                                                   │
│  lib/csv.ts          → Client-side CSV parsing    │
│  lib/store.ts        → Zustand state              │
│  lib/api-client.ts   → Typed fetch wrapper        │
└───────────────────────┬─────────────────────────┘
                        │ HTTPS POST /api/analyze
                        ▼
┌─────────────────────────────────────────────────┐
│  Render (Python 3.12 + FastAPI + Uvicorn)         │
│                                                   │
│  app/routers/analysis.py  → Main endpoint         │
│  app/models/pipeline.py   → 5-model ML pipeline   │
│  app/models/causal.py     → DAG-adjusted OLS      │
│  app/models/intervention.py → Counterfactual sim  │
│  app/utils/preprocess.py  → sklearn transformers  │
│  app/utils/dag.py         → NetworkX DAG ops      │
│  app/schemas.py           → Pydantic v2 schemas   │
└─────────────────────────────────────────────────┘
```

### Why split?

Heavy ML (Random Forest 200 trees, XGBoost, LightGBM, statsmodels OLS) does not belong in a serverless function. Vercel's serverless timeout (10–60 s) and 50 MB bundle limit make this impractical. A persistent Python microservice on Render handles all computation; Next.js handles the UI and is deployed to Vercel.

---

## Tech Stack

### Frontend (`apps/web`)
- **Next.js 15** — App Router, React Server Components
- **TypeScript 5** — strict typing throughout
- **Tailwind CSS 3** — utility-first styling
- **Zustand 5** — client state (dataset + analysis bundle)
- **Recharts 2** — scatter plots, bar charts
- **Sonner** — toast notifications
- **PapaParse** — client-side CSV parsing

### Backend (`apps/api`)
- **FastAPI 0.115** — async HTTP API
- **Pydantic v2** — schema validation, serialisation
- **scikit-learn 1.6** — preprocessing pipelines, Random Forest, Ridge, CV
- **XGBoost 2.1** — gradient boosted trees
- **LightGBM 4.5** — gradient boosted trees
- **statsmodels 0.14** — OLS regression with inference (SE, CI, p-values)
- **NetworkX 3.4** — DAG operations (cycle detection, ancestor/descendant sets)
- **Pandas 2.2 / NumPy 1.26** — data manipulation

---

## Modelling Approach

### Preprocessing

For every analysis run the backend builds a sklearn `ColumnTransformer`:

- **Numeric features**: `SimpleImputer(strategy="median")` → `StandardScaler()`
- **Categorical features** (≤30 unique values): `SimpleImputer(strategy="most_frequent")` → `OrdinalEncoder()`
- High-cardinality text columns are silently dropped
- Rows where the target is missing are removed

### Model training

All five models receive the same training set (80% of data, stratified by default). Evaluation is on the held-out 20% test set. 5-fold cross-validation is run on the full dataset when `n > 200`. The winner is selected by test-set R².

| Model | Key hyperparameters | Notes |
|---|---|---|
| OLS | λ=1e-6 ridge regularisation | Via statsmodels; provides coefficients + SEs |
| Ridge | α=1.0 | Via scikit-learn |
| Random Forest | 200 trees, max_depth=8, min_samples_leaf=10 | Prevents overfitting |
| XGBoost | 300 rounds, lr=0.05, max_depth=4, min_child_weight=10 | Early stop on test set |
| LightGBM | 300 rounds, lr=0.05, num_leaves=31, min_child_samples=20 | Fast, low memory |

---

## Causal Inference Approach

### What we do

For each `controllable` variable we estimate its causal effect on the target using **back-door adjustment**:

```
y ~ feature + confounders + DAG_parents(feature) + context_variables
```

The adjustment set is derived from the user-specified DAG:
- **Include**: declared confounders + DAG parents of the cause + context variables
- **Exclude**: mediators (blocking the causal path would remove the effect we want to estimate)
- **Exclude**: descendants of the cause (collider / post-treatment bias)

OLS is fit on standardised features via statsmodels, giving:
- Standardised β (effect per +1 SD of the cause on the target)
- Standard error, t-statistic, p-value, 95% CI
- Evidence strength classification (strong / moderate / weak / insufficient)

### What we don't do

- We do not claim to recover true causal effects from purely observational data
- We do not implement IV estimation, difference-in-differences, or RCT analysis
- We do not handle unobserved confounders

All causal estimates should be treated as "adjusted correlations with a defensible adjustment set", not as proof of causation. The frontend makes this explicit at every point.

### Role definitions

| Role | Meaning | Used in adjustment? |
|---|---|---|
| `outcome` | The KPI / target variable | No |
| `controllable` | Variables the operator can change | Yes — these are the "causes" we test |
| `confounder` | Causes both the controllable and the outcome | Yes — always included |
| `mediator` | On the causal path controllable→outcome | No — including would block the path |
| `context` | Fixed design/run context | Yes — included to reduce residual variance |
| `identifier` | Row ID / timestamp | No |
| `ignore` | Excluded from analysis | No |

---

## Intervention Logic

The intervention engine combines two evidence sources:

1. **Predictive (GBR counterfactual)**: A GradientBoostingRegressor is trained on all features. For each controllable variable, we simulate shifting it by ±1 SD while holding everything else at the mean, and compute the predicted KPI change. This is a "what-if" simulation — not causal.

2. **Causal direction**: If back-door adjusted OLS produced a significant coefficient (p < 0.05), the sign is used to confirm or override the counterfactual direction.

Each recommendation shows:

| Field | Source |
|---|---|
| Direction | Causal sign (if significant) else GBR |
| Suggested value | Current mean ± 1 SD, clipped to observed range |
| Estimated KPI change | GBR counterfactual simulation |
| Evidence type | `causal` (sig. adj. OLS) / `mixed` / `predictive` |
| Evidence strength | `strong` / `moderate` / `weak` |
| Tradeoff | Domain-aware note |
| Assumptions | Explicit list |
| Caveat | "Validate with experiment" |

---

## Reliability and Safeguards

- **DAG cycle detection** — NetworkX `simple_cycles`; invalid DAGs are rejected with clear error messages
- **Role validation** — mediators are never included in adjustment sets
- **Small-sample warnings** — causal estimates with n < 100 are flagged
- **Evidence typing** — every recommendation is labelled causal / mixed / predictive
- **Schema validation** — Pydantic v2 validates every request and response
- **File constraints** — CSV only, max 50 MB, min 30 rows
- **Row cap** — datasets > 50,000 rows are randomly sampled (seed-reproducible)
- **CORS** — only the declared frontend origin is allowed

---

## Limitations

1. **Observational data only** — all causal estimates may be biased by unobserved confounders.
2. **Linear causal model** — adjusted OLS assumes linear relationships; non-linear effects will be missed.
3. **No time-series structure** — temporal autocorrelation is ignored.
4. **Categorical controllables** — the intervention engine currently targets numeric controllable features only.
5. **No persistence** — analysis results are held in browser memory; refreshing loses them.
6. **No authentication** — the API has no auth layer; add one before exposing to the public internet.

---

## Local Development

### Prerequisites

- Node.js 20+
- Python 3.12+
- npm (or pnpm/bun)

### 1. Clone and install

```bash
git clone https://github.com/your-org/lever-guide.git
cd lever-guide

# Frontend
cd apps/web
npm install

# Backend
cd ../api
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
# API
cp apps/api/.env.example apps/api/.env
# Edit ALLOWED_ORIGINS if needed

# Web (no changes needed for local dev)
cp apps/web/.env.example apps/web/.env.local
```

### 3. Run the backend

```bash
cd apps/api
source .venv/bin/activate
uvicorn app.main:app --reload --port 8000
# API docs at http://localhost:8000/docs
```

### 4. Run the frontend

```bash
cd apps/web
npm run dev
# App at http://localhost:3000
```

The Next.js dev server proxies `/api/*` to `http://localhost:8000` via `next.config.ts` rewrites — no CORS setup needed locally.

### 5. Run tests

```bash
# Backend
cd apps/api
source .venv/bin/activate
pytest tests/ -v

# Frontend type check
cd apps/web
npm run type-check
```

---

## Deployment

### Frontend → Vercel

1. Push the repo to GitHub.
2. Import the repo in [vercel.com/new](https://vercel.com/new).
3. Set **Root Directory** to `apps/web`.
4. Add environment variable:
   ```
   NEXT_PUBLIC_API_URL = https://lever-guide-api.onrender.com
   ```
5. Deploy. Vercel auto-detects Next.js.

### Backend → Render

**Option A: render.yaml (recommended)**

```bash
# From repo root
render deploy --yaml apps/api/render.yaml
```

Or connect your GitHub repo in the Render dashboard and point it to `apps/api`.

Set the environment variable in Render dashboard:
```
ALLOWED_ORIGINS = https://lever-guide.vercel.app
```

**Option B: Docker**

```bash
cd apps/api
docker build -t lever-guide-api .
docker run -p 8000:8000 -e ALLOWED_ORIGINS=https://lever-guide.vercel.app lever-guide-api
```

**Render free tier note**: The free tier spins down after 15 minutes of inactivity. The first request after a cold start may take ~30 s. Upgrade to the Starter plan ($7/mo) to avoid this.

---

## Repository Structure

```
lever-guide/
├── apps/
│   ├── web/                        # Next.js 15 frontend
│   │   ├── app/
│   │   │   ├── layout.tsx          # Root layout + theme provider
│   │   │   ├── page.tsx            # Landing page
│   │   │   ├── setup/page.tsx      # Upload + column role config
│   │   │   └── analyze/page.tsx    # Results dashboard
│   │   ├── components/
│   │   │   └── analyze/
│   │   │       ├── PredictiveTab.tsx
│   │   │       ├── CausalTab.tsx
│   │   │       ├── InterventionsTab.tsx
│   │   │       └── ExecutiveTab.tsx
│   │   ├── lib/
│   │   │   ├── types.ts            # TypeScript types (mirrors Pydantic schemas)
│   │   │   ├── api-client.ts       # Typed fetch wrapper
│   │   │   ├── csv.ts              # Client-side CSV parsing + demo roles
│   │   │   └── store.ts            # Zustand global state
│   │   ├── public/demo/            # Injection-moulding demo CSV
│   │   ├── next.config.ts
│   │   ├── package.json
│   │   └── .env.example
│   │
│   └── api/                        # Python FastAPI backend
│       ├── app/
│       │   ├── main.py             # FastAPI app + CORS
│       │   ├── schemas.py          # Pydantic v2 schemas (source of truth)
│       │   ├── routers/
│       │   │   └── analysis.py     # POST /api/analyze — orchestrator
│       │   ├── models/
│       │   │   ├── pipeline.py     # OLS/Ridge/RF/XGB/LGBM training
│       │   │   ├── causal.py       # Back-door adjusted OLS
│       │   │   └── intervention.py # Counterfactual simulation + ranking
│       │   └── utils/
│       │       ├── preprocess.py   # sklearn ColumnTransformer builder
│       │       └── dag.py          # NetworkX DAG ops + adjustment sets
│       ├── tests/
│       │   └── test_pipeline.py    # Unit + integration tests
│       ├── requirements.txt
│       ├── Dockerfile
│       ├── render.yaml
│       └── .env.example
│
└── README.md
```

---

## Demo Dataset

The injection-moulding demo (`public/demo/injection_molding_demo.csv`) simulates a multi-plant plastics manufacturing operation with 5,000 rows and 32 columns.

**Target**: `scrap_rate_pct` (minimise)

**Controllable variables** (process knobs): `barrel_temperature_c`, `mold_temperature_c`, `injection_pressure_bar`, `hold_pressure_bar`, `screw_speed_rpm`, `cooling_time_s`, `clamp_force_kn`, `shot_size_g`

**Confounders** (environmental): `ambient_temperature_c`, `ambient_humidity_pct`, `resin_moisture_pct`, `resin_batch_quality_index`, `dryer_dewpoint_c`

**Mediators** (on causal path, do not adjust): `cycle_time_s`, `part_weight_g`

**Context** (fixed per run): `cavity_count`, `product_variant`, `operator_experience_level`, `tool_wear_index`, `calibration_drift_index`

Role assignments are applied automatically when loading the demo — they are documented in `apps/api/app/routers/analysis.py:DEMO_ROLES` and `apps/web/lib/csv.ts:DEMO_ROLES`.

---

## Future Roadmap

- [ ] Persistent result storage (PostgreSQL + job queue)
- [ ] Streaming analysis progress via Server-Sent Events
- [ ] Instrumental variable estimation for stronger causal claims
- [ ] Classification mode (logistic regression + AUROC)
- [ ] Categorical controllable interventions
- [ ] Interactive DAG editor with ReactFlow
- [ ] PDF export of the executive summary
- [ ] Multi-user authentication (Clerk / NextAuth)
- [ ] Dataset versioning and result history
- [ ] SHAP values for model explainability

---

## Honest Limitations and Known Gaps

- **No authentication** — add before public deployment
- **No persistent storage** — analysis results live in browser memory only
- **Observational data bias** — causal estimates are adjusted but not guaranteed
- **Linear causal assumption** — non-linear causal relationships are not captured by OLS
- **Cold start latency** — Render free tier has ~30 s cold start; budget plan eliminates this
- **No streaming** — the analysis endpoint is synchronous; large datasets may time out on slow connections

---

*LeverGuide is an analytical support tool. All recommendations should be validated with domain expertise and, for high-stakes decisions, with controlled experiments.*
