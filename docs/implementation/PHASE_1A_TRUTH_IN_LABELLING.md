# Phase 1A — Truth in Labelling and Scientific Safety

Branch `rework/phase-1a-truth-in-labelling`, from `2bd854f` on `main`.
Audit that motivated it: [`docs/audit/`](../audit/README.md).

---

## 1. Purpose

Make the application scientifically honest and operationally safe **before** any
rebranding or visual redesign. The Phase 0 audit found that the product's
long-form prose was careful while its badges, headings and bold numbers were
not — and that underneath the wording, several numbers were simply wrong.

This phase does not redesign the product, rename anything, add a library, or
attempt the analytical-core reconstruction. It corrects what was false and
refuses to present what cannot be supported.

Scope boundary in one line: **every claim the application makes is now either
true, or labelled as an assumption, or not made.**

---

## 2. Scientific problems corrected

Audit IDs in brackets. Every figure below was measured on the shipped 5,000-row
demo, sub-sampled to 2,000 rows with seed 42 — the same path a user takes when
they click "Try Demo Dataset". β is reported in percentage points of scrap per
standard deviation of the lever, converted from the app's dimensionless SD/SD
using σ(scrap) = 1.4976 p.p. on that sample.

### 2.1 `mold_temperature_c` could never be adjusted for — Critical [D-1]

The demo ran on a graph generated from role labels: every confounder points at
every lever and at the outcome. Such a graph has no lever→lever edges, so one
lever can never be a parent of another, so `mold_temperature_c` could never
enter any adjustment set. It is precisely the back-door path the dataset's
headline finding is about — operators extend cooling *in response to* observed
mould temperature.

| | Before | After | Source study |
|---|---|---|---|
| β(`cooling_time_s`) | **−1.171** p.p./SD | **−1.771** p.p./SD | −1.743 |
| Adjusters | 12, identical for every lever | 6, declared per lever | 6 |
| `mold_temperature_c` in the set | no | **yes** | yes |

The sign was never wrong; the magnitude was understated by 33 %. It is now
within 1.6 % of the published value, and matches the audit's reconciliation
ladder row "Datathon adjusters, no fixed effects" (−1.771) exactly — fixed
effects remain a Phase 2 item.

### 2.2 Mediators adjusted for when estimating total effects — High [D-3, C-2]

All three of the dataset's genuine mediators (`resin_moisture_pct`,
`calibration_drift_index`, `tool_wear_index`) were labelled `confounder` or
`context`, so all three entered every lever's adjustment set — while the UI
told the user "Mediators are excluded". The guard in the code was real; the
demo configuration simply never triggered it.

The three are now mediators. They are removed from every total-effect
adjustment set, twice: the declared sets do not contain them, and the estimator
strips them regardless of source. Each estimate lists the mediators it dropped.
They remain **predictors** of the simulation model, following the source study's
reasoning that a world-approximator should not have a real pathway blocked —
which is why levers acting through a mediator are reported as unsupported rather
than as intervention estimates (§2.4).

### 2.3 `shot_size_g` produced physically impossible recommendations — Critical [I-4]

Shot size is mechanically determined by the tooling: it must cover
`cavity_count × part_weight_g` plus runner and cushion. Measured across the
demo, the ratio is 0.972–1.214 with correlation 0.9989.

The engine's **rank-1 recommendation** set `shot_size_g = 66.344 g` for every
row while cavity count and part weight stayed fixed — a guaranteed short shot,
itself one of the dataset's defect classes, for 1,951 of the 2,000 analysed rows
(97.5 %; 92.5 % of the full 5,000).

`shot_size_g` is now marked `derived` in the ontology with the coupling identity
declared as data, and the feasibility layer evaluates the implied ratio per row.
The candidate is returned with `status: "infeasible"`, `rank: 0`, and the row
count in its reason. It is **not** discarded — a rejected candidate is
information about the process.

### 2.4 `hold_pressure_bar` presented as strong evidence — High [D-2]

| | Before | After | Source study |
|---|---|---|---|
| β | **+0.151** p.p./SD | **−0.035** p.p./SD | −0.087 |
| 95 % interval | excludes zero | **includes zero** | includes zero (+0.007) |
| Badge | "strong", ranked #5, *decrease* | `conflicting_evidence`, unranked | excluded from the action package |

Correcting the adjustment set to the source's own (`injection_pressure_bar` plus
the two ambient confounders, minus `tool_wear_index` — see §5) moved the
estimate to the same inconclusive place the source study reached. No sign was
hard-coded to make that happen: the ontology records the evidence as
specification-dependent, and the generic rules — interval crossing zero, or the
adjusted estimate disagreeing with the simulated direction — do the rest.

### 2.5 Two levers were unreachable — High [D-4, D-5]

`dryer_dewpoint_c` was labelled `confounder` and `maintenance_days_since_last`
was labelled `context`, so neither could ever be estimated or proposed. Both are
levers in the source taxonomy, and between them they carry two of the six causal
pathways in the challenge ontology.

| Lever | Before | After | Source study |
|---|---|---|---|
| `dryer_dewpoint_c` | not estimated | **+0.097** p.p./SD | +0.095 |
| `maintenance_days_since_last` | not estimated | **+0.065** p.p./SD | +0.103 |

Their adjustment sets are deliberately minimal — the two ambient confounders
only — so the mediated path stays open and a *total* effect is recovered.

Restoring them must not mean fabricating a causal result. Both act through
mediators that this phase's simulation holds fixed, so their what-ifs carry
`status: "exploratory"` and say why. Mediator propagation is Phase 4; the source
study's own notebook measures the understatement at roughly 3× for dew point,
which is exactly why the simulation is not presented as an intervention
estimate.

### 2.6 Other role corrections [D-6…D-9]

* `clamp_force_kn`: `controllable` → `context`. The source states it is *not* a
  lever — clamp force follows from tonnage requirement, which follows from mould
  and part geometry. It was previously ranked #6 with a suggested value.
* `part_weight_g`: `mediator` → `context`. A fixed design property cannot
  mediate a process setpoint; classifying it as one dropped a genuine adjuster
  from every set.
* `cycle_time_s`: `mediator` → post-treatment outcome (`ignore`). It
  mechanically subsumes `cooling_time_s`. It was previously excluded from the
  feature matrix only *incidentally*, and one dropdown change away from textbook
  leakage.
* `operator_shift`: `context` → `confounder`, matching the source taxonomy.

### 2.7 Evidence strength was a p-value threshold — Medium [D-12]

At n = 2,000 a `p < 0.01` rule labelled six of eight levers "strong", including
one with the wrong sign. Strength is now capped at "weak" whenever the 95 %
interval includes zero, regardless of p. The overview's "significant levers"
count is now estimates whose interval excludes zero.

### 2.8 Interventions had no uncertainty at all — Critical [I-2, C-5]

The homepage advertised "Confidence intervals … always visible". True of effect
estimates; false of the intervention tab, which is the product's headline output.

Each simulation now carries a row-resampling percentile interval over the
simulated change (`interval_method: "row_bootstrap_fixed_model"`). It resamples
rows and holds the fitted model fixed, so it captures variation across
production intervals but **not** model-estimation uncertainty — stated wherever
it is shown. It costs no model refit. Where an interval cannot be computed the
bounds serialise as `null` with a reason; a confidence-interval badge is never
rendered without one.

### 2.9 Two of five models silently did not run — High [M-6]

`xgboost` and `lightgbm` cannot be imported in this environment (`libomp`
missing). Five bare `except Exception: pass` blocks swallowed it, the bundle
carried no warning, and the setup page kept displaying "Running: OLS · Ridge ·
Random Forest · XGBoost · LightGBM".

Every configured model now reports a status — `succeeded`,
`unavailable_dependency`, `training_failed`, `skipped_by_configuration` — with
the underlying error. A model that did not run is never omitted. Both remain
optional; nothing was installed and the application starts without them.

### 2.10 A fresh upload could not succeed — High [F-8]

Every non-identifier column defaulted to `confounder` purely because it was
numeric, and the API then answered 422 because nothing was controllable. Fixed
in §4.6.

### 2.11 Factual corrections [C-12, C-13, I-5, §10]

* "Upload CSV files up to 50 MB" → about 5 MB, which is the real sessionStorage
  and request-body ceiling. Enforced, not just described.
* "All models on the same 80/20 train/test split" → it resolves to 90/10 at
  n = 2,000. Now reports the actual split sizes.
* "holding all others at their mean" → the code keeps each row's own observed
  values, which is the better method. The documentation described a worse method
  than the one implemented.
* W&B logged `adjusted_r2`, `train_rows` and `test_rows`; `ModelMetrics` defines
  `adj_r2`, `n_train` and `n_test`. All three columns were silently null.

---

## 3. Architecture changes

```
apps/api/app/ontology/            NEW — declared dataset ontologies
    schema.py                     VariableSpec, DerivedRelationship, DatasetOntology
    injection_molding.py          the curated demo: roles, adjustment sets, graph,
                                  bounds, coupling identities, source deviations
    registry.py                   resolve_ontology(), validate_ontology()
    export.py                     canonical JSON serialisation
apps/api/app/models/feasibility.py  NEW — feasibility and support checking
apps/api/scripts/export_ontology.py NEW — regenerates the frontend's copy

apps/web/lib/ontology.ts          NEW — typed access to the generated JSON
apps/web/lib/generated/
    demo-ontology.json            GENERATED — do not edit by hand

apps/web/components/analyze/      DELETED — 543 lines nothing imported
```

`analysis.py` no longer holds a hand-maintained `DEMO_ROLES` table; the name
survives as an alias onto the ontology so the import path keeps working.

### Ontology design

One `DatasetOntology` per curated dataset. Per variable:

| Field | Meaning |
|---|---|
| `name`, `label`, `unit`, `description` | identity and presentation |
| `causal_role` | the fine-grained role (10 values, §5) |
| `controllable` | can an operator or planner set this directly |
| `derived` | is this determined by other columns |
| `mediator` | does this sit on a path between a lever and the outcome |
| `valid_range` | declared physical / process bounds |
| `observed_range` | range actually present in the shipped data |
| `categories` | declared levels, for categoricals |
| `intervention_eligibility` | whether, and how honestly, it can carry an intervention |
| `evidence_status` | set when the evidence is known to be specification-dependent |
| `notes` | constraints, caveats, and disagreements between sources |

Plus, at dataset level: `adjustment_sets`, `edges`, `derived_relationships`,
`provenance`, `graph_assumption`, `version`.

### Synchronisation strategy

**Python is authoritative in one direction.**

```
apps/api/app/ontology/*.py
        │  scripts/export_ontology.py
        ▼
apps/web/lib/generated/demo-ontology.json   ──▶  lib/ontology.ts  ──▶  lib/csv.ts
```

`tests/test_ontology.py::test_generated_json_is_in_sync` re-serialises the Python
ontology and asserts byte equality with the committed JSON. Editing a role in one
language and not the other fails the suite. There is no second place to edit a
role, which is what the audit's F-15 asked for.

The frontend's `DEMO_ROLES` is now derived from that JSON rather than
hand-written.

---

## 4. Behaviour by area

### 4.1 Adjustment-set behaviour

1. If the request carries `dag_edges`, that graph is used (`dag_source:
   "user_supplied"`).
2. Else, if a curated ontology matches the dataset (≥ 90 % of its declared
   variables present *and* the target matches), the ontology's graph is used
   (`"declared_domain_ontology"`) and its per-lever adjustment sets are applied
   verbatim.
3. Else, the role-template graph is generated as before
   (`"assumed_from_roles"`), and adjustment sets are derived from it.

A declared set is **never merged** with a derived one — mixing a domain claim
with a heuristic would make the reported set untraceable. Each estimate reports
`adjustment_set_source`. Mediators are stripped in either case.

The declared sets are checked against the declared graph at test time: no
adjuster may be a descendant of its cause, no mediator may appear, every
adjuster must exist, and the graph must be acyclic with the outcome terminal.

`dag_validation.valid` still means *structurally* valid. It now travels with
`dag_source` and `graph_assumption`, so "Valid DAG: true" can no longer be read
as "scientifically defensible".

### 4.2 Intervention statuses

| Status | Meaning | Ranked |
|---|---|---|
| `eligible` | feasible, in support, evidence agrees | **yes** |
| `exploratory` | representable, but the mechanism is only partly modelled | no |
| `unsupported` | outside support, not a lever, or the model did not run | no |
| `infeasible` | violates a documented physical constraint | no |
| `conflicting_evidence` | the adjusted estimate disagrees, or is specification-dependent | no |

Only `eligible` results receive a rank and appear in the primary list. The rest
keep their numbers and appear in a clearly separate "Assessed and set aside"
section with the reason on each. Ranks are contiguous and ordered by simulated
magnitude.

### 4.3 Feasibility and support checks

`app/models/feasibility.py`, in order: eligibility → finiteness → declared range
→ observed support → derived-relationship consistency (per row) → categorical
validity. Every check is recorded with its outcome and detail, passed or failed,
so the UI shows the reasoning rather than a verdict.

Support is reported as `within_observed` / `outside_observed_within_declared` /
`outside_declared` / `unknown`. Extrapolation is never presented as equally
reliable to interpolation.

The layer works without an ontology: the generic path still gets observed-range
checking.

The previous bounds were `p90 × 1.5` and `p10 × 0.5`. Both are arbitrary, and
the sign-dependent lower branch is incoherent for negative-valued columns —
for `dryer_dewpoint_c` it produced −61.5 °C against an ontology floor of −50 °C.
That column was previously spared only because it was mislabelled and never
reached the engine; correcting the role would have made the bug live.

### 4.4 UI terminology

| Concept | Term used everywhere |
|---|---|
| Marginal correlation | **Association** |
| OLS under an assumed graph | **Adjusted observational effect estimate** |
| Model inputs changed, predictions compared | **Predictive what-if simulation** |

Carried as `result_type` on `CorrelationPair`, `CausalEffect` and `Intervention`,
and used for tab labels, section headings, badges and copy.

Adjusted estimates carry: *"Interpretation depends on the selected causal graph
and assumptions, including no important unmeasured confounding."*
What-if simulations carry: *"This modifies model inputs and compares
predictions. It is not automatically a causal intervention estimate."*

Long-form method text sits behind a "Show method details" disclosure rather than
in every card.

`evidence_type: "causal" | "mixed" | "predictive"` — awarded on a p-value alone,
never checking whether the sign agreed — is replaced by
`adjustment_support: "aligned" | "conflicting" | "inconclusive" | "none"`,
rendered as whether the adjusted estimate agrees with the direction.

Phrases removed or prevented: "proven cause", "true causal impact", "guaranteed
improvement", "AI-discovered causal graph", and "recommended intervention"
applied to a bare what-if. A test asserts none of them appears anywhere in the
response payload.

### 4.5 Traceability

`AnalysisProvenance` on every bundle: analysis mode, ontology id and version,
graph source and assumption, adjustment-set source, estimator, effect interval
method, simulation model and evaluation strategy, simulation interval method,
rows supplied versus analysed, sampling note, train/evaluation strategy, seed,
and the resolved role of every column.

Per result: `estimand`, `adjusted_for`, `adjustment_set_source`,
`interval_method`, `n_observations`, `causal_role`, `notes` on effects;
`status`, `status_reason`, `support_status`, `feasibility_checks`,
`adjustment_support`, `simulation_model`, `simulation_evaluation`,
`uncertainty_status` on simulations.

The UI shows a compact provenance panel; the API contract keeps all of it.

### 4.6 Generic upload

* Uploaded columns default to **`unassigned`**, not `confounder`. Unassigned
  columns are predictors and never adjusters.
* `analysis_mode` is now explicit. `descriptive_predictive` needs no treatment
  and returns predictive results and EDA with **no** effect estimates and **no**
  candidate actions — so a user can explore without a causal question being
  invented for them.
* `causal` mode requires a deliberately selected outcome and at least one
  `controllable` or `planning_lever` column, and warns when nothing is labelled
  as an adjuster.
* The client blocks submission and names what is missing and how to fix it. The
  API validates independently and returns structured `problems` with `code`,
  `message`, `remedy` and the columns involved. The client-side check is a
  courtesy, not the gate.
* Generic upload is presented as an advanced path, not automatic causal
  inference.

---

## 5. Role vocabulary and mapping

The ontology's `causal_role` is finer-grained than the app's `ColumnRole`. One
documented mapping bridges them:

| `causal_role` | → `ColumnRole` |
|---|---|
| `process_lever` | `controllable` |
| `planning_lever` | `planning_lever` *(new)* |
| `confounder` | `confounder` |
| `mediator` | `mediator` |
| `context`, `operator_covariate` | `context` |
| `batch_covariate` | `confounder` |
| `identifier` | `identifier` |
| `outcome` | `outcome` |
| `post_treatment_outcome` | `ignore` |

`ColumnRole` gains `planning_lever` and `unassigned`. A sessionStorage payload
written by an older build can hold a role this build does not know; unknown roles
are read back as `unassigned`.

### Resolved roles for the thirteen variables the brief named

| Variable | Causal role | Controllable | Derived | Mediator | Intervention eligibility |
|---|---|---|---|---|---|
| `dryer_dewpoint_c` | process lever | yes | no | no | `mediated_unsupported` |
| `resin_moisture_pct` | mediator | no | no | **yes** | `not_eligible` |
| `maintenance_days_since_last` | planning lever | yes | no | no | `preliminary` |
| `calibration_drift_index` | mediator | no | no | **yes** | `not_eligible` |
| `tool_wear_index` | mediator | no | no | **yes** | `not_eligible` |
| `cooling_time_s` | process lever | yes | no | no | `eligible` |
| `mold_temperature_c` | process lever | yes | no | no | `eligible` |
| `injection_pressure_bar` | process lever | yes | no | no | `eligible` |
| `hold_pressure_bar` | process lever | yes | no | no | `eligible`, evidence `conflicting` |
| `shot_size_g` | process lever | yes | **yes** | no | `derived_constrained` |
| `cavity_count` | context | no | no | no | `not_eligible` |
| `part_weight_g` | context | no | no | no | `not_eligible` |
| `scrap_rate_pct` | outcome | no | no | no | `not_eligible` |

### Where sources disagree

The challenge ontology JSON's coarse `role` field calls `dryer_dewpoint_c` a
mediator and `maintenance_days_since_last` a confounder. The paper taxonomy in
`src/utils.py` calls both levers. **The paper taxonomy wins**, and the
disagreement is recorded in each variable's `notes` rather than resolved
silently.

### Deviations from the source's adjustment sets

Three sets could not be ported verbatim. Each is recorded in
`SOURCE_DEVIATIONS` and surfaced on the affected estimate.

* **`mold_temperature_c`** — `cooling_time_s` dropped. The accepted graph
  encodes `mold_temperature_c → cooling_time_s`, so cooling time is a descendant
  and adjusting for it is post-treatment conditioning. The source declares both
  directions across its two sets, which no single acyclic graph can satisfy.
  Consequence: β moves from the source's +0.879 to +0.392 p.p./SD.
* **`injection_pressure_bar`** and **`hold_pressure_bar`** — `tool_wear_index`
  dropped. It is a mediator in the same source's taxonomy, and this application
  enforces that no mediator enters a total-effect set. The source treats it here
  as a prior wear state instead; deciding between the two readings needs
  temporal ordering that 30-minute intervals cannot supply.

---

## 6. Before / after on the shipped demo

Reproduced with the ontology's own role assignments, seed 42, 2,000 rows.

### Adjusted effect estimates (p.p. of scrap per SD of lever)

| Lever | Before | After | Source | Note |
|---|---|---|---|---|
| `cooling_time_s` | −1.171 *strong* | **−1.771** *strong* | −1.743 | mould temperature now adjusted for |
| `mold_temperature_c` | +0.438 *strong* | +0.392 *strong* | +0.879 | cooling dropped as a descendant |
| `injection_pressure_bar` | +0.215 *strong* | +0.293 *strong* | +0.324 | |
| `barrel_temperature_c` | −0.167 *strong* | −0.143 *strong* | −0.136 | |
| `screw_speed_rpm` | +0.017 *insufficient* | −0.126 *strong* | +0.002 | **known discrepancy — §8** |
| `dryer_dewpoint_c` | *not estimated* | **+0.097** *strong* | +0.095 | |
| `maintenance_days_since_last` | *not estimated* | **+0.065** *moderate* | +0.103 | |
| `hold_pressure_bar` | +0.151 *strong* | **−0.035** *insufficient* | −0.087 | interval now includes zero |
| `shot_size_g` | −0.066 *insufficient* | −0.110 *insufficient* | — | no declared set; graph-derived |
| `clamp_force_kn` | −0.085 *strong* | *not estimated* | — | not a lever |

Adjusters per lever: **12, identical for all** → **3–7, declared per lever**.

### Candidate changes

Before — seven ranked "recommendations", no intervals, no feasibility:

| Rank | Feature | Direction | Badge | Simulated Δ |
|---|---|---|---|---|
| **1** | `shot_size_g` | decrease | mixed / weak | −0.415 (−9.4 %) |
| 2 | `mold_temperature_c` | decrease | causal / strong | −0.365 (−8.3 %) |
| 3 | `cooling_time_s` | increase | causal / strong | −0.251 (−5.7 %) |
| 4 | `injection_pressure_bar` | decrease | causal / strong | −0.198 (−4.5 %) |
| 5 | `hold_pressure_bar` | decrease | causal / moderate | −0.040 (−0.9 %) |
| 6 | `clamp_force_kn` | increase | causal / moderate | −0.040 (−0.9 %) |
| 7 | `screw_speed_rpm` | increase | mixed / weak | −0.024 (−0.5 %) |

Two of the seven, including rank 1, pointed the opposite way from their own
adjusted estimate. Rank 1 was physically impossible for 92.5 % of production.

After — four ranked, four set aside, every one with an interval:

| Rank | Feature | Status | Simulated Δ | 95 % interval |
|---|---|---|---|---|
| 1 | `cooling_time_s` | eligible | −0.674 | [−0.715, −0.633] |
| 2 | `mold_temperature_c` | eligible | −0.345 | [−0.374, −0.317] |
| 3 | `injection_pressure_bar` | eligible | −0.174 | [−0.187, −0.160] |
| 4 | `barrel_temperature_c` | eligible | −0.015 | [−0.021, −0.009] |
| — | `shot_size_g` | **infeasible** | −0.264 | short shot for 1,951 / 2,000 rows |
| — | `hold_pressure_bar` | **conflicting evidence** | — | adjustment set determines the sign |
| — | `screw_speed_rpm` | **conflicting evidence** | — | adjusted estimate is specification-dependent (§8.1) |
| — | `dryer_dewpoint_c` | **exploratory** | −0.031 | acts through a mediator held fixed |
| — | `maintenance_days_since_last` | **exploratory** | −0.020 | acts through a mediator held fixed |

`screw_speed_rpm` previously disappeared from this table entirely rather than
appearing set aside — a defect independent of the discrepancy in §8.1,
described in §12.

`cooling_time_s` — the one lever with strong, well-identified, mechanism-backed
evidence — moves from rank 3 to rank 1. No ranked candidate conflicts with its
own adjusted estimate.

### Model status

| | Before | After |
|---|---|---|
| Models returned | 3 | 3 |
| Models reported | 3 (silently) | **5, with status and reason** |
| Warnings in bundle | 0 | 5 |
| UI claim | "Running: OLS · Ridge · RF · XGBoost · LightGBM" | actual per-model status |

Best-model R² is unchanged at 0.491 (Random Forest). The predictor set changed —
mediators in, `cycle_time_s` out — and the net effect on fit is nil.

---

## 7. Validation

Every command below was run on this branch. Exact outputs.

| Command | Result |
|---|---|
| `cd apps/api && ./.venv/bin/python -m pytest -q` | **104 passed in 11.85s** |
| `cd apps/api && ./.venv/bin/python scripts/export_ontology.py` | `wrote apps/web/lib/generated/demo-ontology.json` |
| `cd apps/web && npx tsc --noEmit` | clean, no output |
| `cd apps/web && npx next lint` | `✔ No ESLint warnings or errors` |
| `cd apps/web && npm run build` | `✓ Compiled successfully`, `✓ Exporting (3/3)`, 4 static routes |
| `./.venv/bin/python -c "import xgboost"` | `XGBoostError: … libomp.dylib` — **pre-existing, not fixed** |
| `./.venv/bin/python -c "import lightgbm"` | `OSError: … libomp.dylib` — **pre-existing, not fixed** |

Baseline at `2bd854f` for comparison: **27 passed in 6.44s**, with both boosted
models unimportable and no warning about it — which is itself evidence the old
suite did not test what the product claimed to do.

Every commit on this branch was checked out into a scratch worktree and tested
in isolation: `4fd8454`, `65a97d4`, `bf47001` each pass 56/56 at their own tree
state.

**No pre-existing failure was hidden and no test was weakened.** The two
`libomp` import failures are reported as found; `libomp` was not installed, per
the instruction not to add system dependencies. There were no pre-existing test
failures to distinguish from regressions — the baseline suite was fully green.

### Test coverage added

77 tests added (27 → 104), each tied to a measured Phase 0 finding:

| Requirement | Test |
|---|---|
| Demo ontology roles | `test_demo_ontology_roles` (13 parametrised) |
| Cooling-time adjustment set | `test_cooling_time_adjusts_for_mold_temperature`, `test_cooling_time_effect_matches_the_source_analysis` |
| Mediators not blindly added | `test_no_mediator_is_adjusted_for_in_any_estimate`, `test_no_mediator_appears_in_any_total_effect_adjustment_set` |
| `shot_size_g` cannot be ranked | `test_shot_size_cannot_be_ranked`, `test_shot_size_is_rejected_by_the_feasibility_layer_directly` |
| `hold_pressure_bar` not strong | `test_hold_pressure_is_never_a_strong_eligible_recommendation` |
| Dew point and maintenance | `test_previously_unreachable_levers_are_estimated`, `test_mediated_levers_are_not_presented_as_intervention_estimates` |
| Result-type labels | `test_the_three_result_types_are_distinct_and_labelled`, `test_no_unsupported_phrase_appears_anywhere_in_the_payload` |
| Missing intervals | `test_missing_intervals_are_null_and_explained_never_invented`, `test_every_intervention_either_has_both_bounds_or_neither` |
| Optional model failure | `test_unimportable_boosters_are_reported_not_swallowed`, `test_training_failure_is_distinguished_from_a_missing_dependency` |
| Fresh generic upload | `test_fresh_upload_in_causal_mode_is_rejected_with_a_remedy`, `test_unlabelled_columns_are_never_treated_as_confounders` |
| Out-of-support warnings | `test_out_of_support_value_is_flagged`, `test_support_checks_work_without_an_ontology` |
| Existing API validation | the 7 pre-existing 422 tests, unchanged |
| Existing behaviour preserved | the full pre-existing suite, unchanged except for the pipeline tuple return |

---

## 8. Known limitations and uncertainty

**Stated plainly, because this phase is about not overstating things.**

1. **`screw_speed_rpm` disagrees with the source, and it is not the sample
   size.** An independent review after this phase reproduced the earlier
   version of this note — which attributed the gap primarily to the 2,000-row
   sub-sample — and found that explanation unsupported. Re-run here, on this
   branch's own estimator (`app/models/causal.py`):

   | Specification | Rows | β (p.p./SD) | 95% interval | Excludes zero |
   |---|---|---|---|---|
   | Declared set, seed 42 (shipped demo) | 2,000 | −0.126 | [−0.190, −0.061] | yes |
   | Declared set, seed 1 | 2,000 | −0.124 | [−0.189, −0.059] | yes |
   | Declared set, seed 7 | 2,000 | −0.136 | [−0.201, −0.072] | yes |
   | Declared set, **full dataset** | **5,000** | **−0.132** | [−0.173, −0.091] | yes |
   | Declared set + `product_variant`, seed 42 | 2,000 | +0.015 | [−0.050, +0.081] | no |
   | Declared set + `product_variant`, **full dataset** | **5,000** | **+0.005** | [−0.037, +0.047] | no |
   | Source study, published (full FE) | 5,000 | +0.002 | [−0.037, +0.045] | no |

   The full 5,000-row dataset gives −0.132, statistically indistinguishable
   from every 2,000-row seed tried (−0.124 to −0.137). **Sub-sampling is not
   the driver of this discrepancy and the earlier claim that it was is
   withdrawn.**

   The source study's own code (`datathon-CUB-2026/src/causal_helpers.py`,
   run directly against the identical shipped dataset) was used to isolate the
   cause. It absorbs `machine_id`, `mold_id`, `product_variant` and
   `operator_shift` as fixed effects on **every** regression, on top of the
   declared backdoor set — a Phase 2 item this application does not implement
   (limitation 3 below). Toggling those fixed effects individually:

   | Fixed effects included | β (p.p./SD), full data | 95% interval |
   |---|---|---|
   | None | −0.132 | [−0.174, −0.084] |
   | `machine_id` + `mold_id` + `operator_shift` (no `product_variant`) | −0.090 | [−0.132, −0.045] |
   | `product_variant` only | +0.005 | [−0.036, +0.049] |
   | All four (published) | +0.002 | [−0.037, +0.045] |

   `product_variant` alone accounts for essentially the entire gap;
   `machine_id`/`mold_id`/`operator_shift` together move the estimate only
   from −0.132 to −0.090. An ANOVA on the shipped data confirms why:
   `screw_speed_rpm` varies sharply by `product_variant` (one-way ANOVA,
   F = 274, p ≈ 0; variant means range 61.7–76.4 rpm), and so does
   `scrap_rate_pct` (F = 126, p ≈ 1.4×10⁻¹⁷¹) — `product_variant` is a real
   common cause of both.

   This is not treated as a porting mistake, and `product_variant` has **not**
   been added to this lever's adjustment set. The source's own declared
   backdoor set for `screw_speed_rpm` (`src/utils.py`) is byte-identical to
   this application's — `product_variant` enters the source's number only
   through its blanket fixed-effects layer, not through a DAG-justified
   backdoor argument specific to this lever. Adding it here would mean
   approximating fixed effects one variable at a time to reproduce a specific
   published number, which this phase's instructions rule out. The two
   implementations therefore currently answer **slightly different adjusted
   questions** for this lever: this application's estimate is the total
   effect under the declared graph with no fixed effects; the source's is the
   same total effect additionally net of machine/mould/variant/shift
   group-level differences.

   Because the sign and significance of this specific estimate are not robust
   to that difference — unlike every other lever on this branch, which
   reconciles with the source to within the stated tolerances — `screw_speed_rpm`
   is now declared `evidence_status: "conflicting"` in the ontology, the same
   mechanism already used for `hold_pressure_bar` (§2.4). Its intervention
   status is `conflicting_evidence`: the adjusted estimate is still computed
   and shown, but it is not ranked and no direction is asserted. Resolving
   this for real needs the fixed-effects work in limitation 3, which is
   Phase 2 scope.
2. **`mold_temperature_c` is now further from the source** (+0.392 against
   +0.879) because `cooling_time_s` was dropped as a descendant. That is the
   right call under an acyclic graph, but it means the two codebases are
   answering slightly different questions for this lever.
3. **No fixed effects, no cluster-robust standard errors.** The source absorbs
   machine, mould, variant and shift in every regression. Mean scrap ranges
   3.82–5.53 % between machines. The i.i.d. OLS intervals shown are materially
   too narrow for repeated intervals on the same machines.
4. **Random, non-grouped, non-temporal validation split.** Grouped CV by machine
   scores about 0.05–0.08 R² lower and is the honest number for "will this work
   on your line".
5. **Simulation is in-sample.** The model is fitted and evaluated on the same
   rows; the audit measured optimism of +0.26 R² for this model. Stated on every
   simulation, not corrected.
6. **The simulation interval is partial.** It resamples rows and holds the model
   fixed, so it excludes model-estimation uncertainty. The true interval is
   wider. Labelled as such wherever it appears.
7. **Mediator propagation is not implemented.** Levers that need it are marked
   `exploratory` rather than being given a fabricated number.
8. **No conditional, cap-only, delta or package interventions.** The source's
   five actions cannot all be expressed; `injection_pressure_bar` in particular
   is estimated unconditionally where the source conditions on tool wear.
9. **Coupling constraints are checked, not enforced by construction.** Only the
   documented `shot_size_g` identity is declared. `cooling_time_s → cycle_time_s`
   and the energy couplings are known and unmodelled.
10. **The graph is an assumption.** Within-interval simultaneity between mould
    temperature and cooling time is not resolvable at 30-minute granularity; the
    direction encoded here is the mechanism the source study argues for, not
    something the data can show.
11. **The paper PDFs were never read**, by the audit or by this phase. Every
    "source study" value here is quoted second-hand from that repository's
    README and notebooks.
12. **No frontend test runner exists.** Frontend validation is `tsc`, `next
    lint` and `next build` only.
13. **The 2,000-row sub-sample is a simple random sample** of a time-ordered
    panel, so no figure here is directly comparable to a published figure
    computed on all 5,000 rows.
14. **`resin_batch_quality_index` is mapped to `confounder`** rather than a
    distinct batch-covariate role, because the app's coarse vocabulary has no
    better slot. It is adjusted for either way.

---

## 9. Deferred work

Explicitly **not** attempted here, in the order the audit recommends:

* **Phase 2 — analytical core.** Absorbed fixed effects, cluster-robust standard
  errors by machine, clustered bootstrap intervals on β, `GroupKFold` by machine
  as the primary validation scheme, preprocessing fitted inside folds, model
  selection separated from reporting, one-hot encoding for nominal categoricals,
  permutation importance, residual diagnostics, subgroup analysis, precomputed
  artifacts. This is what would resolve limitations 1–4.
* **Phase 1b — repositioning.** Product narrative, EDA section (the correlations
  and distributions are computed and still never rendered), README rewrite,
  splitting the 705-line analyze page, shareable permalinks, removing the 23
  unused npm dependencies, resolving the two contradictory `render.yaml` files.
* **Phase 3 — identification and refutation.** Interactive DAG, real back-door
  search, placebo and sensitivity refuters, renaming `causal.py` to
  `adjusted_effects.py`.
* **Phase 4 — constrained simulator.** Mediator propagation (the delta method),
  conditional / cap-only / delta / package interventions, out-of-fold simulation
  models, computed trade-offs.
* **Phase 5 — CI and deployment.**

---

## 10. Manual QA checklist

1. Home page — no claim that confidence intervals are "always visible"; upload
   limit reads ~5 MB; feature grid describes optional boosters honestly.
2. `Try Demo Dataset` → Setup — roles are pre-filled from the ontology;
   `dryer_dewpoint_c` shows as controllable, `maintenance_days_since_last` as
   planning lever, the three mediators as mediator, `clamp_force_kn` as context.
3. Run the demo → Overview — the model-status panel lists five models with
   XGBoost and LightGBM marked unavailable, with their reason; warnings mention
   both by name.
4. Adjusted Effect Estimates tab — `cooling_time_s` shows β ≈ −1.18 SD/SD; its
   adjustment-set card lists `mold_temperature_c` and says "declared by the
   dataset ontology"; `hold_pressure_bar` shows an interval spanning zero and is
   *not* badged strong; "Show method details" opens without claiming mediators
   are excluded from an analysis that adjusts for them.
5. What-If Simulations tab — four eligible candidates ranked 1–4 with intervals;
   a separate "Assessed and set aside" section containing `shot_size_g`
   (infeasible, with the row count), `hold_pressure_bar` and `screw_speed_rpm`
   (both conflicting evidence), and the two mediated levers (exploratory).
   Expanding a card shows the feasibility checks with pass/fail marks. No
   configured lever is missing from either section (§12).
6. Executive Summary — heading counts eligible and set-aside candidates; "Top
   Levers to Pull" is gone; the provenance panel shows the ontology version and
   the sampling note.
7. Upload any CSV → Setup — every non-identifier column reads `unassigned`; the
   Run button is disabled; step 4 explains what is missing and how to fix it.
8. Switch to "Descriptive & predictive only" and run — succeeds with no
   treatment; the Adjusted Effect Estimates and What-If tabs are empty with an
   explanation.
9. Label one column controllable and one confounder, switch back to causal mode,
   run — succeeds; unassigned columns appear in the warning about columns used
   as predictors only.
10. Reload `/analyze` directly — still redirects to `/setup` (pre-existing
    behaviour, unchanged; see audit F-10).

---

## 11. Reproduce

```bash
cd apps/api && ./.venv/bin/python -m pytest -q
cd apps/api && ./.venv/bin/python scripts/export_ontology.py
cd apps/web && npx tsc --noEmit && npx next lint && npm run build
```

---

## 12. Post-review corrections

An independent adversarial review after the rest of this phase found two High
findings and a documentation error, addressed here without reopening the
broader analytical-core work deferred to Phase 2.

1. **A constant-valued predictor, confounder or treatment could crash the
   API.** A design matrix with too little identifying information relative to
   its adjustment set — most directly triggered by a zero-variance column, but
   sharing the same root cause as ordinary multicollinearity — could make
   `statsmodels` return a non-finite coefficient, standard error or interval.
   Serialised through the real response path, this raised
   `ValueError: Out of range float values are not JSON compliant`
   (`starlette.responses.JSONResponse` sets `allow_nan=False`), an unhandled
   500. Fixed in `app/models/causal.py`, `app/models/pipeline.py` and
   `app/models/intervention.py`:
   * A constant **outcome** is rejected with a structured `CONSTANT_OUTCOME`
     validation error (`app/routers/analysis.py`).
   * A constant **treatment** is rejected for causal-effect estimation with a
     structured reason; if every configured treatment is constant, the whole
     request is rejected (`ALL_TREATMENTS_CONSTANT`); descriptive and
     predictive results are unaffected either way.
   * A constant **adjuster** (confounder, context or generated dummy) is
     dropped from the affected lever's design matrix, not from the user's
     configuration — `AnalysisProvenance.excluded_columns` records the
     column, the lever, and the reason, and a bundle warning names the count.
   * Every numeric field is validated finite immediately before it would be
     returned; a result that is still non-finite is withheld with a reason
     rather than serialised, never replaced with a fabricated number.
   * See `apps/api/tests/test_review_corrections.py` for behavioural coverage,
     including the exact crash reproduction above turned into a regression
     test.

2. **A configured lever could disappear from the Interventions tab with no
   trace.** `run_intervention_engine` tried both the "increase" and "decrease"
   direction for each lever and silently `continue`d — dropping the
   candidate entirely — whenever neither direction was estimated to improve
   the outcome. This is what made `screw_speed_rpm` vanish: it has a valid,
   strong adjusted estimate, but the predictive simulation found no improving
   direction for it, and the lever was dropped rather than reported. The same
   code path could drop any lever, and separately dropped a lever with no
   observed variation for a different reason (no headroom to simulate a
   change at all). Both cases now produce an explicit `Intervention` record
   with `status: "unsupported"` and a specific reason instead of being
   skipped. `screw_speed_rpm` specifically now also carries
   `evidence_status: "conflicting"` (item 3), so it resolves to
   `conflicting_evidence` rather than `unsupported` — its adjusted estimate is
   preserved and shown, no direction is asserted, and it appears in "Assessed
   and set aside", never in the ranked list. See
   `apps/api/tests/test_review_corrections.py::test_api_reports_unsupported_lever_when_neither_direction_improves`
   for the generic regression, and
   `apps/api/tests/test_review_corrections.py::test_screw_speed_rpm_appears_with_an_explicit_non_ranked_status`
   for the demo lever.

3. **The screw-speed documentation blamed the wrong thing.** §8.1 previously
   attributed the `screw_speed_rpm` discrepancy primarily to the 2,000-row
   sub-sample. Re-investigated and rewritten in place (§8.1): the full
   5,000-row dataset and three different seeds all reproduce the same
   negative, significant estimate this application reports, so sub-sampling
   does not explain it. The reproducible driver, isolated using the source
   study's own code against the identical dataset, is `product_variant` —
   which the source absorbs as a fixed effect on every regression rather than
   through a declared backdoor argument specific to this lever. `screw_speed_rpm`'s
   adjustment set was **not** changed to include it (that would mean adding a
   variable to match a published number, which these instructions rule out,
   and fixed effects remain Phase 2 scope); instead the lever is now declared
   `evidence_status: "conflicting"`, and the two implementations' differing
   estimates are documented as answering slightly different adjusted
   questions rather than one being an unexplained bug.

None of this reopens Phase 2 (absorbed fixed effects, `GroupKFold`, mediator
propagation) or any of the other Medium/Low findings from the same review.
`mold_temperature_c`'s adjustment set, the mediator-exclusion guarantee,
`shot_size_g`'s infeasible status, `hold_pressure_bar`'s non-primary status,
optional-model visibility, generic-upload role defaults, and the
   association/adjusted/predictive result-type distinction were all confirmed
   unchanged by the existing `apps/api/tests/test_scientific_safety.py` suite.

To regenerate the before/after numbers in §6, check out `2bd854f` into a
worktree and post the demo CSV to `/api/analyze` with that commit's
`DEMO_ROLES`.
