# Target Architecture

Covers audit requirements §8 (preserve / refactor / replace / remove) and §9 (target design).

---

## Part 0 — Is the proposed direction correct?

**Proposed:** *Causal Process Studio — From correlation to intervention in manufacturing*, a guided
recruiter-facing causal case study, replacing the generic "upload any CSV" product.

**Assessment: adopt it, with two amendments.** The evidence for it is not aesthetic:

1. **The generic path does not work today.** A fresh upload defaults every column to `confounder`
   (`csv.ts:92-98`) and the API rejects it (`analysis.py:272-276`). The advertised path fails on
   first use (`FRONTEND_REPO_QUALITY.md` F-8).
2. **Generic causal inference is not a solvable problem.** Roles, adjustment sets, physical bounds,
   coupling identities and mediator chains are all *domain knowledge*. `auto_dag` is what "generic"
   collapses to: 60 template edges that cannot express the one mechanism the case study is about
   (`CURRENT_ARCHITECTURE.md` §5). Every measured defect in `SCIENTIFIC_DISCREPANCIES.md` traces to
   trying to be domain-agnostic.
3. **The Datathon material is stronger than the app built on it**, and none of its distinctive
   assets — mediator chains, conditional/cap-only rules, ontology bounds, anti-recommendations —
   survive the generic abstraction.
4. **Recruiter economics.** 3–5 minutes of attention. A curated argument that lands one
   counter-intuitive finding beats a configurable tool whose demo path is the only one that works.

**Amendment 1 — do not delete generic upload; demote it.** Keep it behind a clearly-labelled
"experimental / bring your own data" route with honest limits. It is real engineering, it
demonstrates schema validation and error handling, and deleting it discards working code for a
positioning argument. It must stop being the homepage promise.

**Amendment 2 — trim the topic list.** Thirteen demonstrated topics is a syllabus, not a case study.
Colliders in particular have **no instance in this dataset** — nothing in the Datathon is labelled a
collider (`DATATHON_METHODOLOGY.md` §2). Teach a collider as a *counter-example* (why `defect_type`
must not be conditioned on) rather than pretending the data contains one. Same for "ML evaluation":
show one honest grouped-CV comparison, not a five-model leaderboard.

**One risk to state plainly.** The Datathon repository has its own confirmed defects — the
hard-coded trade-off narrative that contradicts its own computed output (D1/D2 in
`DATATHON_METHODOLOGY.md`). Rebuilding *from* it without fixing those inherits a demonstrably wrong
conclusion into the flagship artefact.

---

## Part 1 — Preserve / refactor / replace / remove

| Component | Classification | Justification |
|---|---|---|
| **Next.js** | **Preserve largely unchanged** | `output: "export"` static build is free on Vercel/Pages, needs no Node server, and App Router is the right fit for a guided narrative. No credible reason to change. |
| **TypeScript** | **Preserve largely unchanged** | Typed API contracts are a core quality signal. Fix the *duplication* (F-15), not the language. |
| **React Flow** | **Reimplement from the original Datathon source** | Currently a declared-but-unused dependency (F-4). The DAG is the centrepiece of the intended product, and the graph to render is `src/utils.py` `ADJUSTMENT_SETS` + `key_causal_pathways`, not `auto_dag`. Build it for real, sourced from the Datathon ontology. Alternative considered: static SVG/Mermaid — cheaper, but loses the click-a-node-see-its-adjustment-set interaction that is the pedagogical payload. |
| **FastAPI** | **Preserve but refactor** | Correct choice — Pydantic contracts, OpenAPI, async. Refactor from "the app" to "a thin optional layer over the analysis package": routes should call `causal_process/` functions, not contain pipeline logic. Note the free Render tier cold-starts in ~50 s, so the page must not *depend* on it. |
| **pandas** | **Preserve largely unchanged** | Right tool, both repos use it correctly. |
| **statsmodels** | **Preserve but refactor** | Keep for OLS inference. Refactor to add absorbed fixed effects (D-10) and `cov_type="cluster"` by `machine_id` (D-11). |
| **scikit-learn** | **Preserve but refactor** | Keep. Refactor every fit into a `Pipeline` inside `GroupKFold` folds (M-1, M-2, M-4). |
| **XGBoost** | **Remove from the core product** | Adds a native `libomp` dependency that **is currently broken in the committed venv** and fails silently (M-6). On this dataset it buys nothing a `HistGradientBoostingRegressor` cannot. Removing eliminates a whole class of deployment failure. |
| **LightGBM** | **Remove from the core product** | Same, and it is redundant with XGBoost. Keep **one** boosted model: sklearn's `HistGradientBoostingRegressor` — pure-wheel, no OpenMP, comparable accuracy. |
| **DoWhy** | **Defer to a later experimental feature** | Attractive because its four refutation tests (placebo treatment, random common cause, data subset, unobserved-confounder sensitivity) are exactly the "robustness and sensitivity" the product wants, and they are hard to hand-roll credibly. But it is a heavy dependency with a large transitive tree on a 512 MB tier. **Recommendation: use DoWhy offline in the precompute step only**, never at request time — refutation results become static artifacts. This is the one addition that adds genuine scientific content rather than technology count. |
| **DoubleML** | **Remove from the core product** | With 8 levers, a hand-specified DAG, ~20 covariates and 5,000 rows, DML solves a problem this analysis does not have (high-dimensional nuisance). It would add a dependency and a second estimator whose disagreement with OLS you would then have to explain. Revisit only if the adjustment sets grow large. |
| **Qdrant** | **Replace completely** | The corpus is ~7 short documents per analysis, embedded with `HashingVectorizer` — hashed bag-of-words, not semantic vectors (`rag.py:50-56`). A vector database is unnecessary infrastructure for cosine similarity over seven strings; the same result is a 30-line numpy dot product. It also writes SQLite to the server disk, which does not survive a free-tier restart. Replace with in-process retrieval over precomputed artifacts. |
| **Groq** | **Preserve but refactor** | Genuinely free tier, fast, and the grounding prompt (`rag.py:400-405`) is well written. Refactor: ground it in the *fixed* case-study artifacts rather than per-request output, fix the overstated source text first (`CAUSAL_CLAIMS_AUDIT.md` C-14), keep the graceful no-key degradation, and disclose in the UI that summary statistics leave the browser. |
| **Weights & Biases** | **Remove from the core product** | Currently logs `None` into three columns because of a field-name mismatch (`CURRENT_ARCHITECTURE.md` §10), is off by default, and tracks *user requests* rather than *experiments* — which is not what W&B is for. The reproducibility need is better met by versioned artifacts in git. Optionally re-add later to track the offline precompute sweep, which is a real experiment. |
| **Docker** | **Preserve but refactor** | The multi-stage Dockerfile is correct and is a good engineering signal. Refactor to make it the **actual** deployment mechanism (`runtime: docker`) instead of an unused file, and delete the second one. |
| **Render** | **Preserve but refactor** | Free tier suits an optional API. Refactor: delete `apps/api/render.yaml` (the `plan: starter` paid config) so one file is authoritative (F-12). Accept ~50 s cold starts — which is precisely why the case study must render from precomputed artifacts. |
| **Vercel** | **Preserve largely unchanged** | Free static hosting with instant loads and preview deploys. Should host the case study. |
| **Database / authentication** | **Remove from the core product** | No multi-user requirement, no private data in a public case study, and both add cost, ops surface and a login wall between a recruiter and the work. Replace persistence with **shareable URL state** (F-6) and precomputed JSON. |
| **Generic CSV upload** | **Defer to a later experimental feature** | See Amendment 1. Keep the code, move it behind `/experimental`, fix the default-role failure or state it plainly. |
| `apps/web/components/analyze/*` (dead) | **Replace completely** | 543 dead lines duplicating inline logic (F-2). Rebuild as the real components. |
| `dag.py::auto_dag` | **Replace completely** | Cannot express lever→lever edges; produces identical adjustment sets for every lever (§5 of `CURRENT_ARCHITECTURE.md`). Replace with the declared DAG + a real back-door adjustment search. |
| `dag.py::validate_dag` | **Preserve largely unchanged** | Cycle/malformed/unknown-node checks are correct and well tested. Only rename `valid` → `structurally_valid` (C-8). |
| `intervention.py` | **Reimplement from the original Datathon source** | Port `counterfactual_shift`, `simulate_combined_package`, `_apply_moisture_chain`, `_apply_drift_chain`, `LEVER_RANGES` — then add the bootstrap interval and coupling constraints that *neither* repository has. |
| `causal.py` | **Preserve but refactor** | Keep statsmodels OLS + the honest homoskedasticity disclosure. Add per-lever adjustment sets, fixed effects, clustered SE, bootstrap. Rename to `adjusted_effects.py` (C-6). |
| `preprocess.py` | **Preserve but refactor** | Keep the structure; move the fit inside folds, one-hot nominals (M-7), remove silent high-cardinality drops. |
| `pipeline.py` | **Preserve but refactor** | Keep 2–3 models; remove `eval_set` (M-5), CV on training rows only (M-3), surface failures (M-6), select on inner folds (M-2). |
| `src/utils.py` (Datathon) | **Preserve largely unchanged** | The most valuable single artefact across both repos. Port verbatim as the domain ontology. |
| `src/causal_helpers.py` (Datathon) | **Preserve but refactor** | Keep `estimate_adjusted_effect` and the beta_std/beta_unstd honesty note. Refactor the bootstrap to resample **rows and refit** rather than resampling residualised vectors (D7), and to cluster by machine. |
| `src/intervention_helpers.py` (Datathon) | **Preserve largely unchanged** | Delta-propagation is the best idea in the source material. Add intervals and coupling checks around it. |
| Notebooks 01–03 | **Preserve but refactor** | Keep as the narrative source and as executable documentation, but move the logic into the package so the notebooks *call* it. Fix D1–D3 hard-coded narratives first. |
| `wandb_tracking.py` | **Remove from the core product** | See W&B above. |
| `rag.py` | **Preserve but refactor** | Keep artifact-building and the Groq grounding; drop Qdrant. |

**Explicitly not added:** DoubleML, EconML, a database, an auth provider, a job queue, a second
vector store, a second boosted-tree library. The only new dependency recommended is **DoWhy, offline
only**, and only because refutation tests are scientific content the product currently lacks.

Net dependency change: **−2 native ML libraries, −1 vector DB, −1 tracking service, −23 npm
packages, +1 offline-only Python library.**

---

## Part 2 — Target design

```
┌──────────────────────────────────────────────────────────────────┐
│  causal-process-studio/                                          │
│                                                                  │
│  packages/causal_process/          ← the scientific core         │
│    ontology.py       roles, adjustment sets, LEVER_RANGES,       │
│                      coupling identities  (from src/utils.py)    │
│    data.py           load, validate, clip, schema assertions     │
│    eda.py            distributions, raw correlations, ρ vs β     │
│    dag.py            declared DAG, back-door search, d-separation│
│    effects.py        adjusted OLS + FE + clustered SE + bootstrap│
│    predict.py        leakage-safe GroupKFold pipelines           │
│    mediation.py      moisture & drift sub-models, delta method   │
│    simulate.py       constrained counterfactuals, packages, CI   │
│    refute.py         placebo, random common cause, subset,       │
│                      unobserved-confounder sensitivity (DoWhy)   │
│    artifacts.py      deterministic JSON emission + manifest      │
│                                                                  │
│  scripts/precompute.py  → artifacts/*.json  (committed)          │
│                                                                  │
│  apps/web/   Next.js static export → Vercel                      │
│    reads artifacts/*.json at build time                          │
│    optional: calls the API for live simulation                   │
│                                                                  │
│  apps/api/   FastAPI on Render free tier — OPTIONAL              │
│    thin wrapper over packages/causal_process                     │
│                                                                  │
│  tests/      pytest + Vitest      .github/workflows/ci.yml       │
└──────────────────────────────────────────────────────────────────┘
```

### Component specifications

#### 1. `packages/causal_process` — modular analysis library

* **Responsibility:** every scientific claim the product makes. Pure functions, no I/O beyond
  `data.py` and `artifacts.py`, no web framework imports.
* **Inputs:** the demo CSV + the ontology.
* **Outputs:** typed dataclasses → JSON artifacts.
* **Technology:** Python 3.11, pandas, numpy, scikit-learn, statsmodels; DoWhy in `refute.py` only.
* **Alternatives:** R + `dagitty`/`fixest` (better causal ecosystem, but splits the stack and blocks
  a single-language deployment); a notebooks-only project (no tests, no CI, no reuse); keeping logic
  in FastAPI (what exists today — untestable without HTTP, and unusable from a notebook).
* **Why this fits:** it is the one change that makes everything else testable. The same functions
  serve the precompute script, the API, the notebooks and the test suite.
* **What the reader learns:** that analysis code can be engineered — versioned, typed, tested, and
  callable from four contexts without duplication.

#### 2. `scripts/precompute.py` → committed artifacts

* **Responsibility:** run the full analysis deterministically and emit versioned JSON.
* **Inputs:** CSV + ontology + a pinned seed.
* **Outputs:** `artifacts/{eda,effects,models,mediation,interventions,refutations,manifest}.json`,
  where the manifest records input SHA-256, package version, library versions and seed.
* **Technology:** a plain Python script invoked by `make artifacts` and by CI.
* **Alternatives:** compute at request time (what exists — 20–60 s latency plus a ~50 s free-tier
  cold start, and results that change between visits); DVC/MLflow (real value at scale, pure
  overhead for one dataset).
* **Why this fits:** it is what makes the case study **free, instant, and reproducible at once**. A
  recruiter's page load costs zero compute. CI re-runs it and fails the build if any number moves.
* **What the reader learns:** reproducibility as an artifact you can diff, not a claim you make.

#### 3. Guided public case study (Next.js static export)

* **Responsibility:** carry the argument, in order:
  `Problem → EDA → Prediction vs Causation → the DAG → Identification → Adjusted effects →
   The sign reversal → Robustness → Constrained simulation → Trade-offs → Limits → Reproduce it`.
* **Inputs:** `artifacts/*.json` at build time.
* **Outputs:** static HTML/CSS/JS.
* **Technology:** Next.js App Router `output: "export"`, Tailwind, Recharts, React Flow.
* **Alternatives:** MDX/Quarto/Observable (excellent for a document; loses the React interactivity
  and the full-stack demonstration); a Jupyter Book (free, but reads as coursework).
* **Why this fits:** already in the repo, deploys free, loads instantly, and demonstrates
  front-end engineering alongside the science.
* **What the reader learns:** that the author can sequence a technical argument for a reader who has
  five minutes.

#### 4. Interactive DAG (React Flow)

* **Responsibility:** make identification visible. Click a lever → highlight its back-door paths,
  its adjustment set, the mediators deliberately left open, and the resulting estimate. Toggle
  "adjust for `mold_temperature_c`" and watch β move from −1.17 to −1.84 p.p./SD — the live
  demonstration of D-1.
* **Inputs:** declared DAG + per-lever adjustment sets + precomputed β under each adjustment choice.
* **Outputs:** rendered graph + a synchronised estimate panel.
* **Technology:** React Flow (already a dependency).
* **Alternatives:** static Graphviz/Mermaid image (cheap, zero interaction — and interaction *is* the
  lesson here); D3 by hand (more control, much more code); `dagitty` embed (authoritative
  d-separation, but an iframe to a third-party site).
* **Why this fits:** it converts the project's single most important finding from a sentence into
  something the reader operates. Nothing else in the plan has that ratio.
* **What the reader learns:** that an adjustment set is a *choice* with a measurable consequence.

#### 5. Leakage-safe model pipelines

* **Responsibility:** honest generalisation estimates.
* **Inputs:** feature frame + `machine_id` groups + timestamps.
* **Outputs:** per-fold metrics, per-machine held-out scores, forward-chaining time-split scores,
  permutation importances with CIs, residual diagnostics.
* **Technology:** sklearn `Pipeline` + `GroupKFold(5, groups=machine_id)` as primary; a
  forward-chaining time split as secondary. Two models only: Ridge (interpretable baseline) and
  `HistGradientBoostingRegressor` (nonlinear).
* **Alternatives:** see the measured comparison in `ML_EVALUATION_AUDIT.md` §2 — random split 0.491,
  grouped-by-machine 0.437, machine hold-out 0.413, time 0.461.
* **Why this fits:** the deployment unit is a machine. Reporting **0.437 instead of 0.491, and
  explaining the 0.054 gap**, is the strongest credibility signal the project can send.
* **What the reader learns:** that choosing the validation scheme *is* the modelling decision.

#### 6. Constrained intervention simulator

* **Responsibility:** turn identified effects into feasible policies with intervals.
* **Inputs:** fitted out-of-fold model, mediator sub-models, `LEVER_RANGES`, coupling identities,
  an intervention spec (`delta` | `target` | `cap_only`, optional condition, optional package).
* **Outputs:** PATE + bootstrap CI + rows-intervened + constraint-violation count + trade-off deltas
  (cycle time, energy) + a support warning.
* **Technology:** the ported Datathon logic plus the missing pieces.
* **Alternatives:** DoWhy `CausalModel.do()` (principled, but assumes a full SCM this data does not
  support); pure linear extrapolation from β (transparent, misses interactions the warpage cross-tab
  proves are real).
* **Why this fits:** it demonstrates the actual difference between a what-if and an intervention —
  and the `shot_size_g` failure (92.5 % of rows made physically impossible) is a memorable, concrete
  argument for domain constraints.
* **What the reader learns:** that unconstrained counterfactuals confidently recommend impossible
  things.

#### 7. Optional API-backed live simulation

* **Responsibility:** let a visitor move a slider and get a fresh PATE.
* **Inputs:** intervention spec + a pinned dataset id (no upload on this route).
* **Outputs:** PATE + CI.
* **Technology:** FastAPI on Render free, called only on interaction, with the precomputed value
  rendered immediately as the default so the page is complete before the API wakes.
* **Alternatives:** everything precomputed on a grid (fully free and instant, but a fixed grid);
  Pyodide in the browser (impressive, ~10 MB payload); serverless functions (cold starts too).
* **Why this fits:** progressive enhancement — the case study never *depends* on a service that
  cold-starts in 50 s, but the full-stack capability is still demonstrated.
* **What the reader learns:** how to degrade gracefully around an unreliable free tier.

#### 8. Tests and CI

* **Responsibility:** guarantee the scientific claims, not just the plumbing.
* **Scientific tests (the ones that do not exist today):**
  * `mold_temperature_c ∈ adjustment_set("cooling_time_s")`
  * `−1.9 < β(cooling) < −1.6` p.p./SD on the full demo
  * `sign(β(cooling)) != sign(ρ(cooling, scrap))` — the sign reversal, asserted
  * no mediator appears in any total-effect adjustment set
  * every simulated row satisfies `LEVER_RANGES` and every coupling identity
  * no predictor is a deterministic function of the target
  * every artifact matches its committed manifest hash
* **CI:** GitHub Actions — `ruff`, `mypy`, `pytest`, `tsc --noEmit`, `next build`,
  `make artifacts && git diff --exit-code artifacts/` (fails if any number silently moved).
* **Alternatives:** none worth considering; GitHub Actions is free for public repos.
* **What the reader learns:** that analytical results can be regression-tested like code.

---

## Cost

| Item | Cost |
|---|---|
| Vercel static hosting | $0 (hobby) |
| GitHub Actions (public repo) | $0 |
| Render free web service (optional API) | $0, ~50 s cold start |
| Groq (optional copilot) | $0 free tier |
| Removed: Qdrant, W&B, Render starter ($7/mo), database, auth | −$7/mo and −4 services |

**Total: $0/month**, and the case study renders fully with every external service down.
