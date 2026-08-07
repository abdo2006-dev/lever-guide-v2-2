# Phased Migration Plan

**Principle: no big-bang rewrite.** The existing app stays deployable and correct-as-described at
the end of every phase. Each phase is independently valuable and independently revertible.

**Sequencing decision, stated up front.** The suggested order puts repositioning (Phase 1) before
the analytical core (Phase 2). I recommend **swapping them**: Phase 2 first. The reason is
`SCIENTIFIC_DISCREPANCIES.md` D-1 — the shipped app understates its headline lever by 33 % and gets
`hold_pressure_bar`'s sign wrong. Repositioning the product around "from correlation to
intervention" while the intervention numbers are wrong would put a false claim on the front page and
require redoing the copy after Phase 2 anyway. Phase 1 as originally scoped is therefore split: a
tiny **Phase 1a (truth-in-labelling)** runs first because it is cheap and removes active
misstatements; the full repositioning becomes **Phase 1b** and runs after Phase 2.

Revised order: **0 → 1a → 2 → 1b → 3 → 4 → 5**.

---

## Phase 0 — Scientific and architectural audit ✅ complete

* **Scope:** reconstruct both systems; measure the discrepancies; classify every component.
* **Non-goals:** any code change, any commit, any dependency install.
* **Files affected:** none. Deliverables written outside both repositories.
* **Dependencies:** read access; the committed `apps/api/.venv`.
* **Risks realised:** `xgboost`/`lightgbm` unimportable in the local venv (reported, not
  worked around); the paper PDFs were not parsed, so all paper values are second-hand via the
  Datathon README.
* **Acceptance criteria — met:** 10 documents; every finding cited to `file:line`; five findings
  independently reproduced by running the app's own code.
* **Test strategy:** `pytest -q` → 27 passed; three read-only verification scripts.
* **Educational outcome:** a discrepancy matrix that distinguishes *what was measured* from *what
  was hypothesised* — including one hypothesis (cooling sign flip) that measurement **refuted**.
* **Rollback:** nothing to roll back.

---

## Phase 1a — Truth in labelling (1–2 days)

* **Scope:** correct every statement the audit proved false, without changing any computation.
  * "80/20 train/test split" → the actual ratio (`analyze/page.tsx:139`) — C-13.
  * "Max 50 MB" → the real limit, 3 places + README — C-12.
  * "Mediators are excluded" → state what is actually adjusted for — C-2.
  * "holding all others at their mean" → describe what the code does — I-5.
  * "Honest uncertainty — CIs always visible" → scope it to effect estimates — C-5.
  * Surface skipped/failed models in `bundle.warnings` — M-6.
  * Fix the W&B field names `adj_r2`/`n_train`/`n_test` — or delete the module.
  * Delete `apps/api/render.yaml`; remove stray `{apps` directories; delete the 543 dead lines in
    `components/analyze/`; drop the 23 unused npm dependencies.
* **Non-goals:** no role changes, no estimator changes, no new features. Numbers must not move.
* **Files:** `analyze/page.tsx`, `setup/page.tsx`, `HomeClient.tsx`, `intervention.py`,
  `pipeline.py`, `wandb_tracking.py`, `README.md`, `apps/api/render.yaml`, `package.json`.
* **Dependencies:** none.
* **Risks:** low. Dropping npm deps could break an unnoticed transitive import — mitigated by
  `next build` + `tsc --noEmit` before merge.
* **Acceptance:** every claim in `CAUSAL_CLAIMS_AUDIT.md` classified *incorrect* is either true or
  removed; `npm run build` and `tsc --noEmit` pass; `pytest` still 27/27; a diff of the analysis
  bundle before/after shows **no numeric change**.
* **Test strategy:** golden-file test — serialise the demo bundle before the change, assert
  byte-identical numerics after.
* **Educational outcome:** how to make a system honest before making it better — the cheapest
  credibility gain available.
* **Rollback:** single revertible commit; no behavioural surface touched.

---

## Phase 2 — Analytical core reconstruction (1–2 weeks)

* **Scope:** create `packages/causal_process/` and move the science into it.
  * `ontology.py` ported verbatim from `datathon-CUB-2026/src/utils.py` — roles, per-lever
    `ADJUSTMENT_SETS`, `FIXED_EFFECTS`, `LEVER_RANGES`, plus **new** coupling identities
    (`shot_size_g ≈ cavity_count × part_weight_g`, `cycle_time_s ⊃ cooling_time_s`).
  * Fix the roles: `dryer_dewpoint_c`→lever, `maintenance_days_since_last`→planning lever,
    `resin_moisture_pct`/`calibration_drift_index`/`tool_wear_index`→mediator,
    `clamp_force_kn`/`part_weight_g`→context, `cycle_time_s`→outcome/leakage (D-3…D-9).
  * `effects.py`: per-lever adjustment sets, absorbed FE, cluster-robust SE by `machine_id`,
    row-resampling clustered bootstrap (D-1, D-2, D-10, D-11).
  * `predict.py`: `Pipeline` fitted inside `GroupKFold(5, machine_id)`; one-hot nominals; inner-fold
    model selection; permutation importance with CIs; residual diagnostics; per-machine and
    forward-chaining time scores (M-1…M-4, M-7, M-8, M-10, M-11).
  * Drop XGBoost and LightGBM; keep Ridge + `HistGradientBoostingRegressor`.
  * `scripts/precompute.py` → `artifacts/*.json` + manifest with input SHA-256 and library versions.
  * **Fix the Datathon's own D1/D2** — recompute the cycle-time and scrap-to-cycle trade-off from
    data instead of hard-coding, and report whatever it says.
* **Non-goals:** no UI change; no new mediation or refutation; no removal of the existing API — the
  old path keeps running until Phase 3 switches routers over.
* **Files:** new `packages/causal_process/*`, new `scripts/`, new `artifacts/`; `apps/api/app/models/*`
  becomes a thin adapter; `analysis.py::DEMO_ROLES` and `csv.ts::DEMO_ROLES` corrected together.
* **Dependencies:** Phase 1a (so the numeric change is attributable and not tangled with copy edits).
* **Risks:**
  * *Every published number changes.* Cooling β moves −1.17 → ≈−1.74 p.p./SD; `hold_pressure_bar`
    flips sign. **Mitigation:** commit the old artifacts first, then the new ones, so the diff is
    reviewable and explainable — and put that diff in the case study as a finding.
  * Reported R² **drops** 0.491 → 0.437 when grouped CV replaces random. This is correct and must be
    framed as such, not hidden.
  * `DEMO_ROLES` duplicated in two languages; changing one and not the other silently breaks the
    demo. **Mitigation:** generate the TS copy from the Python ontology in CI.
* **Acceptance:**
  * `β(cooling_time_s)` within 2 % of the Datathon's −1.743 p.p./SD on the full 5,000 rows;
  * `mold_temperature_c ∈ adjustment_set("cooling_time_s")`;
  * no mediator in any total-effect adjustment set;
  * grouped-CV R² reported as primary with the random-split number shown as a labelled contrast;
  * `make artifacts` twice → byte-identical output.
* **Test strategy:** the seven scientific assertions in `TARGET_ARCHITECTURE.md` §8; a
  characterisation test pinning every artifact number to its manifest hash.
* **Educational outcome:** identification assumptions belong in a declared, tested ontology — and a
  validation scheme is a modelling decision with a measurable price.
* **Rollback:** the package is additive; the existing pipeline stays until Phase 3. Revert = stop
  calling the new module.

---

## Phase 1b — Product repositioning and frontend structure (1–2 weeks)

* **Scope:** *Causal Process Studio*. Rebuild the route structure as a guided narrative reading
  `artifacts/*.json` at build time: Problem → EDA → Prediction vs Causation → DAG → Identification →
  Effects → **The sign reversal** → Robustness → Simulation → Trade-offs → Limits → Reproduce.
  Split `analyze/page.tsx` (705 lines) into per-section components. Build the real EDA section from
  the `correlations`/`distributions` that are already computed and discarded (F-3). Rewrite the
  README around the finding, not the inventory (F-13). Move generic upload behind `/experimental`
  with honest limits. Add shareable URL state. Deploy one public URL.
* **Non-goals:** no DAG editor yet (Phase 3); no live simulation (Phase 4); no new estimators.
* **Files:** `apps/web/app/**`, new `apps/web/components/**`, `README.md`, `vercel.json`,
  delete `apps/web/components/analyze/*`.
* **Dependencies:** Phase 2 artifacts must exist and be correct.
* **Risks:** scope creep into visual redesign — **mitigation:** every UI change must cite a
  comprehension/trust/recruiter justification from `FRONTEND_REPO_QUALITY.md` in its PR description.
  Second risk: the narrative outruns the evidence — mitigation: every claim in the copy links to the
  artifact field that produces it.
* **Acceptance:** page loads with all external services down; time-to-first-finding under 30 s of
  reading; `next build` static export clean; one live URL; Lighthouse accessibility ≥ 90.
* **Test strategy:** Vitest for artifact-rendering components; Playwright smoke test walking the
  narrative; axe accessibility check in CI.
* **Educational outcome:** how to sequence a technical argument for a reader with five minutes.
* **Rollback:** the old `/setup` → `/analyze` flow stays live at `/experimental` throughout.

---

## Phase 3 — Identification, estimation, refutation (1–2 weeks)

* **Scope:** interactive React Flow DAG sourced from the ontology, with click-a-lever →
  back-door paths → adjustment set → estimate. The toggle that shows β moving −1.17 → −1.84 when
  `mold_temperature_c` is added is the centrepiece. Real back-door adjustment-set search replacing
  `auto_dag` (which is deleted). `refute.py`: placebo treatment, random common cause, data-subset
  refuter, unobserved-confounder sensitivity (E-value / Rosenbaum-style bound) — run **offline** in
  precompute, DoWhy used here only. Rename `causal.py` → `adjusted_effects.py` and
  `evidence_type: "causal"` → `adjustment_support` (C-3, C-6).
* **Non-goals:** no causal discovery from data — the DAG stays declared, and the UI says so.
* **Files:** `packages/causal_process/{dag,refute}.py`, `apps/web/components/dag/*`,
  `apps/api/app/utils/dag.py` (auto_dag removed, validate_dag kept), `schemas.py`.
* **Dependencies:** Phase 2 (adjustment sets), Phase 1b (narrative slots to render into).
* **Risks:** a refutation test may **fail** — e.g. the placebo test on a weak lever. That is a
  result, not a bug; the plan is to publish it. DoWhy's transitive dependency tree — mitigated by
  keeping it out of the API requirements entirely (offline only).
* **Acceptance:** every displayed effect carries its adjustment set, its identification assumption,
  a clustered bootstrap CI, and four refutation outcomes; the DAG is navigable by keyboard; the
  sign-reversal toggle is live.
* **Test strategy:** d-separation unit tests against hand-worked examples from the DAG notes;
  snapshot tests on refutation outputs; assert `auto_dag` no longer exists.
* **Educational outcome:** identification is a graph-theoretic argument, and refutation is how you
  probe an assumption you cannot verify.
* **Rollback:** refutation panels are additive; the DAG view can ship read-only first.

---

## Phase 4 — Constrained intervention simulator (1–2 weeks)

* **Scope:** port `counterfactual_shift`, `simulate_combined_package`, `_apply_moisture_chain`,
  `_apply_drift_chain` and `LEVER_RANGES` from the Datathon. Add what neither repo has: bootstrap
  CIs on every PATE (I-2), coupling-constraint enforcement with a violation count (I-4), support
  checking (I-3), out-of-fold simulation models (I-1), direction taken from the identified estimate
  with conflicts surfaced rather than badged (C-3, I-10), and the five Datathon action shapes —
  delta, cap-only, conditional, package, per-row target (I-7). Optional Render-backed live slider
  with the precomputed value as the instant default. Computed trade-offs (cycle time, energy)
  replacing the hard-coded lookup (I-8).
* **Non-goals:** no DoubleML, no full SCM, no optimisation over lever combinations.
* **Files:** `packages/causal_process/{mediation,simulate}.py`, `apps/api/app/models/intervention.py`
  (becomes an adapter), `apps/web/components/simulator/*`.
* **Dependencies:** Phases 2 and 3.
* **Risks:** the constraint layer may reject a lot of the counterfactual space — that is the point,
  and the rejection count should be displayed. Bootstrap cost: ~200 replicates × a boosted model is
  minutes, so it belongs in precompute, not in a request. Reproducing the Datathon's PATE magnitudes
  exactly is *not* expected — the Datathon itself reproduces its own paper to only ±35 % on
  individual actions.
* **Acceptance:** all five Datathon actions reproduce with the same **sign and rank order**;
  combined package lands within 10 % of −0.56 p.p.; **zero** simulated rows violate `LEVER_RANGES`
  or a coupling identity; every PATE carries an interval; the `shot_size_g` recommendation either
  disappears or ships with an explicit infeasibility flag.
* **Test strategy:** property test — no generated counterfactual violates a constraint, over
  randomised specs; regression test on the five action PATEs; a test asserting the pre-fix
  `shot_size_g` counterfactual is now rejected.
* **Educational outcome:** the difference between a what-if, an identified effect, and a *feasible*
  policy — demonstrated by a concrete impossible recommendation that the constraint layer catches.
* **Rollback:** simulator is a new route; the Phase 1b static intervention section stands alone.

---

## Phase 5 — Documentation, CI, deployment, polish (3–5 days)

* **Scope:** GitHub Actions (`ruff`, `mypy`, `pytest`, `tsc --noEmit`, `next build`, axe, and
  `make artifacts && git diff --exit-code artifacts/`). Method appendix per section. A reproduction
  guide that works from a clean clone in one command. Docker as the actual deploy mechanism.
  Deployment consolidated to one Vercel URL + one optional Render service. Delete W&B and Qdrant.
  Final accessibility pass: CI whiskers on the forest plot, non-colour significance encoding, chart
  `aria-label`s, full names in tooltips (F-9).
* **Non-goals:** no new analysis.
* **Files:** `.github/workflows/ci.yml`, `Makefile`, `docs/**`, `Dockerfile`, `render.yaml`,
  `vercel.json`; delete `rag.py`'s Qdrant path and `wandb_tracking.py`.
* **Dependencies:** all prior phases.
* **Risks:** the artifact-diff CI gate will fail on any library upgrade — that is intended, but pin
  versions and document the regeneration step so it is not experienced as flakiness.
* **Acceptance:** green CI on a clean clone; `make reproduce` regenerates every artifact
  byte-identically; one live URL; zero unused dependencies; Lighthouse accessibility ≥ 95; README
  opens with the finding.
* **Test strategy:** the CI matrix is the test; plus a scheduled weekly run to catch dependency drift.
* **Educational outcome:** reproducibility enforced by machinery rather than asserted in prose.
* **Rollback:** CI gates can be set to warn-only for one cycle before enforcing.

---

## Preservation strategy across all phases

1. **Tag `pre-migration` at HEAD before Phase 1a.** Everything below is a convenience on top of that.
2. **Commit artifacts before and after Phase 2** so the numeric change is a reviewable diff, not a
   silent overwrite.
3. **Keep the current app reachable at `/experimental`** through Phase 4. It is working software; it
   is being *repositioned*, not condemned.
4. **Never delete a component in the same commit that adds its replacement** — two commits, so a
   revert restores a working state.
5. **The Datathon repository is the reference, not a dependency.** Port code with attribution in
   comments; do not git-submodule it. It has its own unfixed defects (D1–D8).
6. **Golden-file tests are the safety net.** Once the Phase 2 numbers are accepted, any later
   unintended change fails CI.

## Effort summary

| Phase | Duration | Risk | Value |
|---|---|---|---|
| 1a Truth in labelling | 1–2 d | Low | High — removes active misstatements |
| 2 Analytical core | 1–2 w | **High** | **Highest** — fixes D-1/D-2 |
| 1b Repositioning | 1–2 w | Medium | Highest for recruiter evaluation |
| 3 Identification & refutation | 1–2 w | Medium | High — the DAG is the centrepiece |
| 4 Simulator | 1–2 w | Medium | High — differentiator |
| 5 CI & polish | 3–5 d | Low | Medium–High — reproducibility |

**Total ≈ 6–9 weeks part-time.** Phases 1a and 2 alone remove every finding classified Critical or
High in this audit.
