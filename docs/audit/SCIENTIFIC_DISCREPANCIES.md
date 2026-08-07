# Scientific Discrepancy Matrix — Datathon vs LeverGuide

All numbers below were produced during this audit by importing LeverGuide's own modules and running
them on the demo CSV both applications ship. Reproduction commands are in §5.

**Unit warning that governs every comparison in this file.** The two codebases report β in
different units and comparing them naively is itself a methodological error:

* Datathon `beta_std` (`src/causal_helpers.py:62-67`) — z-scored lever, **unscaled outcome** →
  **p.p. of scrap per SD of lever**.
* LeverGuide `effect_per_std` (`causal.py:52-60`) — z-scores **both** → **SD of scrap per SD of
  lever**, dimensionless.

σ(scrap) = 1.4976 p.p. in the 2,000-row sample LeverGuide analyses. Every LeverGuide β below has
been converted to p.p./SD by multiplying by σ(scrap) before comparison.

---

## 1. Role assignment — the source of nearly every downstream discrepancy

`DEMO_ROLES` is duplicated in two files that must agree: `apps/api/app/routers/analysis.py:37-77`
and `apps/web/lib/csv.ts:6-40`. They currently do agree. Compared against
`datathon-CUB-2026-main/src/utils.py:23-73`:

| Variable | Datathon role | LeverGuide role | Match |
|---|---|---|---|
| `dryer_dewpoint_c` | **process lever** | **confounder** | ✗ |
| `resin_moisture_pct` | **mediator** | **confounder** | ✗ |
| `maintenance_days_since_last` | **planning lever** | **context** | ✗ |
| `calibration_drift_index` | **mediator** | **context** | ✗ |
| `tool_wear_index` | **mediator** | **context** | ✗ |
| `cooling_time_s` | process lever | controllable | ✓ |
| `mold_temperature_c` | process lever | controllable | ✓ |
| `injection_pressure_bar` | process lever | controllable | ✓ |
| `scrap_rate_pct` | outcome | outcome | ✓ |
| `clamp_force_kn` | **context ("NOT a lever")** | **controllable** | ✗ |
| `cycle_time_s` | **outcome, leakage, excluded** | **mediator** | ✗ (partially mitigated) |
| `part_weight_g` | **context** | **mediator** | ✗ |
| `resin_batch_quality_index` | batch covariate | confounder | ≈ |
| `operator_experience_level` | adjusted covariate | context | ≈ (both adjusted) |
| `shot_size_g` | process lever | controllable | ✓ |

**9 of 15 audited variables carry a different causal role.** Three of the five actions the Datathon
recommends are unreachable as a direct consequence.

---

## 2. Discrepancy matrix

### D-1 — `mold_temperature_c` omitted from `cooling_time_s`'s adjustment set — **CRITICAL**

| | |
|---|---|
| **Original role** | Process lever, and the **required** confounder for cooling: it heads `ADJUSTMENT_SETS["cooling_time_s"]` (`src/utils.py:82`) because operators extend cooling *in response to* observed mould temperature. |
| **LeverGuide role** | `controllable` (`analysis.py:41`). `adjustment_set()` (`dag.py:116-119`) admits only `confounders ∪ DAG-parents ∪ context`. Under `auto_dag`, one controllable is never a parent of another, so **`mold_temperature_c` can never enter any lever's adjustment set**. |
| **Why it matters** | This is the exact back-door path the Datathon's headline finding is about. Leaving it open leaves the reactive-compensation confounding partially unblocked. |
| **Estimand changed** | Total causal effect of cooling → a partially-confounded association. |
| **Measured impact** | LeverGuide `effect_per_std = −0.7816` → **−1.171 p.p./SD**. Adding `mold_temperature_c` alone: **−1.840 p.p./SD**. Datathon: **−1.743 p.p./SD**. The shipped app **understates the most important lever in the dataset by 33 %**. |
| **Nuance — do not overstate** | The **sign does not flip.** I hypothesised a reversal before measuring and it did not occur: enough of the confounding is absorbed by the 12 other adjusters that the protective direction survives. The defect is a large magnitude bias, not a wrong recommendation. |
| **Severity** | **Critical** — it is the one number the whole case study exists to get right. |
| **Correction** | Replace `adjustment_set()`'s role-based heuristic with per-lever declared sets, or implement a real back-door search over a DAG that contains lever→lever edges. Add a regression test asserting `mold_temperature_c ∈ adjustment_set("cooling_time_s")` and `−1.9 < β < −1.6` p.p./SD on the demo. |

### D-2 — `hold_pressure_bar` sign reversal — **HIGH**

| | |
|---|---|
| **Original** | β = −0.087 p.p./SD, CI [−0.180, **+0.007**] crossing zero; adjustment set `{injection_pressure_bar, tool_wear_index, ambient×2}`; explicitly **excluded from the action package on significance grounds** (`notebooks/02` cell 25). |
| **LeverGuide** | β = **+0.1008** SD/SD = **+0.151 p.p./SD**, p = 7.6e-05, badged **"strong"**, ranked #5 with direction *decrease*. |
| **Why it matters** | Opposite sign, and the "CI crosses zero → do not act" judgement is inverted into a statistically strong actionable recommendation. |
| **Estimand** | Adjusting for `injection_pressure_bar` (a sibling lever, in the Datathon's set) is dropped, while 12 unrelated adjusters including three mediators are added. The coefficient is picking up the pressure pathway. |
| **Measured** | LG `+0.151` vs Datathon `−0.0398` p.p./SD — **ratio −3.80×**. |
| **Severity** | **High** — a confirmed sign error presented as strong evidence. |
| **Correction** | Per-lever adjustment sets; and gate "actionable" on a CI excluding zero, not on a p-value alone. |

### D-3 — Mediators adjusted for when estimating total effects — **HIGH (structural), LOW (measured on this dataset)**

| | |
|---|---|
| **Original** | `resin_moisture_pct`, `calibration_drift_index`, `tool_wear_index` are mediators; `data/README.md:97` — "Conditioning on [them] blocks causal pathways and produces direct (not total) effect estimates." Only `tool_wear_index` appears in one set, for `injection_pressure_bar`, where the Datathon treats it as a wear-state confounder. |
| **LeverGuide** | All three are labelled `confounder`/`context` and therefore enter **every** lever's adjustment set. Verified: the set for `cooling_time_s` contains `resin_moisture_pct`, `calibration_drift_index`, `tool_wear_index`. |
| **Why it matters** | This is the textbook total-vs-direct error, and the app's own UI text (`analyze/page.tsx:344`) claims the opposite: "Mediators are excluded (blocking the causal path would absorb the effect)." The code does exclude anything the *user* labels `mediator` — but the shipped demo labels the real mediators as something else, so the guard never fires. |
| **Estimand** | Total effect → controlled direct effect, for every lever whose pathway runs through moisture/drift/wear. |
| **Measured** | For `cooling_time_s` specifically, removing the three mediators moves β from −1.171 to −1.160 p.p./SD — **under 1 %**. Cooling does not act through these mediators, so it is barely affected. The damage is concentrated on the levers LeverGuide cannot estimate at all (D-4, D-5). |
| **Severity** | **High** as a design defect and a documentation contradiction; **Low** as a measured bias on the one lever where it can be measured. Reported this way deliberately — the fix is warranted on principle, not because of a large observed number. |
| **Correction** | Fix `DEMO_ROLES` to match `src/utils.py`; make the role vocabulary carry `planning_lever`; add a test that fails if any variable in the mediator list appears in any adjustment set. |

### D-4 — `dryer_dewpoint_c` demoted from lever to confounder — **HIGH**

| | |
|---|---|
| **Original** | Process lever. β = +0.095 p.p./SD. Recommended action #3: −5 °C **when ambient humidity ≥ 65 %**, PATE −0.09 p.p. Effect propagates through `resin_moisture_pct` via the moisture sub-model (R² = 0.09). |
| **LeverGuide** | `confounder` (`analysis.py:53`). Verified: it never enters `controllable`, so `run_causal_analysis` never estimates it and `run_intervention_engine` never proposes it. It appears only as an *adjuster* for other levers. |
| **Why it matters** | The moisture pathway — one of the six pathways in the challenge ontology, and the mechanism behind the highest-mean-scrap defect (splay, 4.74 %) — is invisible in the product. |
| **Estimand** | Not merely biased: **not estimated**. |
| **Severity** | **High.** |
| **Correction** | `dryer_dewpoint_c: "controllable"`, `resin_moisture_pct: "mediator"`, and implement mediated propagation (see `INTERVENTION_AUDIT.md` I-6). |

### D-5 — `maintenance_days_since_last` demoted to context — **HIGH**

| | |
|---|---|
| **Original** | Planning lever (`src/utils.py:30`) — a scheduling decision, not a per-interval setpoint. β = +0.103 p.p./SD. Action #5: cap at 14 days, propagated through the calibration-drift sub-model (R² = 0.78). |
| **LeverGuide** | `context` (`analysis.py:62`). Verified: never estimated, never recommended. |
| **Why it matters** | The drift chain is the *best-identified* chain in the dataset (R² = 0.78, vs 0.09 for moisture) and the app discards it. It is also the clearest example of a lever that operates on a different time-scale from a process setpoint — pedagogically the most interesting one. |
| **Severity** | **High.** |
| **Correction** | Introduce a distinct `planning_lever` role that is estimable and interventionable but excluded from per-interval "setpoint" framing. |

### D-6 — `calibration_drift_index` and `tool_wear_index` as context — **MEDIUM**

Both are mediators in the Datathon (`src/utils.py:43`). As `context` in LeverGuide they are
(a) adjusted for in every regression — D-3 — and (b) available as GBR predictors during
counterfactual simulation, which is *correct* for a world-approximator but is never accompanied by
the propagation logic that makes it meaningful. `tool_wear_index` is additionally the *conditioning
variable* for the Datathon's pressure rule ("−30 bar when wear ≥ 0.45"), and LeverGuide's API has no
way to express a conditional intervention at all. **Severity: Medium.** Correction: role →
`mediator`; add `condition_col`/`condition_threshold` to the intervention schema.

### D-7 — `clamp_force_kn` promoted to a controllable lever — **MEDIUM**

`src/utils.py:50` says in a comment: "`clamp_force_kn` is **NOT** listed as a controllable lever in
the paper." LeverGuide makes it `controllable` (`analysis.py:45`) and it duly appears as **rank 6**,
direction *increase*, suggested 3694 kN, badged `causal/moderate`. Clamp force on an injection
machine is set by tonnage requirement — it is a consequence of mould and part geometry, not a free
knob. Also note the Datathon clips it at 4400 kN as sensor noise (`src/utils.py:120`); LeverGuide
does not, and the observed max is exactly 4400.0 in the shipped demo CSV.
**Correction:** role → `context`.

### D-8 — `cycle_time_s` as mediator rather than leakage — **MEDIUM, currently harmless**

The Datathon is emphatic: `cycle_time_s` mechanically subsumes `cooling_time_s`, is an outcome, and
"will cause data leakage in any model estimating cooling-time effects"
(`data/README.md:96`, `src/utils.py:68`, enforced at `src/causal_helpers.py:137`).

LeverGuide labels it `mediator`. **In practice this is currently safe**: `pred_features`
(`analysis.py:302-305`) selects only `controllable|confounder|context`, so mediators are excluded
from the predictive matrix and from the intervention GBR. Verified — `cycle_time_s` is not among the
20 features in the built matrix. But the protection is incidental: **a user who relabels
`cycle_time_s` as `context` or `confounder` in the setup UI immediately introduces textbook target
leakage**, and nothing in the app warns them. **Correction:** add an explicit `leakage`/`outcome`
role that cannot be reassigned to a predictor role, plus a mechanical-subsumption warning.

### D-9 — `part_weight_g` as mediator rather than context — **LOW–MEDIUM**

Datathon: context, and an active member of the adjustment sets for `cooling_time_s`,
`mold_temperature_c` and `injection_pressure_bar` (`src/utils.py:83, 87, 96`). LeverGuide: mediator,
therefore **excluded from every adjustment set and from the predictive matrix**. It is a fixed
design property of the part — it cannot be a mediator of a process setpoint. Removing a genuine
adjuster contributes to D-1/D-2's attenuation. **Correction:** role → `context`.

### D-10 — No fixed effects anywhere — **HIGH**

The Datathon absorbs `machine_id`, `mold_id`, `product_variant`, `operator_shift` in **every**
regression (`src/utils.py:72`, applied at `src/causal_helpers.py:49`). LeverGuide labels
`machine_id`, `mold_id`, `plant_id`, `resin_lot_id` as `identifier` → routed straight into
`identifiers` (`analysis.py:270`) and dropped. Only `product_variant` and `operator_shift` survive,
as `context`, and are ordinal-encoded rather than dummy-encoded.

Measured relevance: mean scrap by machine ranges **3.82 % (DEN_IM_03) → 5.53 % (NAM_IM_02)** — a
1.7 p.p. spread the model cannot absorb. Re-running the Datathon estimator with and without FE:
cooling goes −1.771 → −1.826 p.p./SD, so FE is not the dominant term for cooling, but it is
unambiguously the right specification for panel data and it matters more for the plant-correlated
levers (dewpoint, humidity-conditional rules). **Correction:** dummy-encode the grouping identifiers
as absorbed fixed effects rather than discarding them.

### D-11 — Uncertainty procedure downgraded — **MEDIUM**

| | Datathon | LeverGuide |
|---|---|---|
| Effect CI | 300-replicate bootstrap percentile CI (`causal_helpers.py:70-77`) | Textbook OLS CI under homoskedasticity (`causal.py:100`) |
| Clustering | Acknowledged as needed, not implemented | Not acknowledged, not implemented |
| Intervention CI | none | none |

LeverGuide's `analyze/page.tsx:345` states the CIs are "from OLS inference under homoskedasticity
assumptions" — accurate and creditable. But with 12 machine clusters and serially correlated
half-hourly intervals, i.i.d. OLS standard errors are materially too narrow, which is what produces
the p-values ≤ 1e-40 that drive the "strong" badges. **Correction:** cluster-robust SE by
`machine_id`, or a machine-clustered block bootstrap.

### D-12 — Evidence strength = p-value threshold — **MEDIUM**

`causal.py:19-28`: `p<0.01 → "strong"`. With n = 2,000 this labels almost everything strong. Six of
eight levers came back "strong", including `clamp_force_kn` (β = −0.057 SD, a negligible effect) and
`hold_pressure_bar` (β with the wrong sign). The Datathon's confidence column is a *judgement*
combining effect size, CI width, mechanism plausibility and sub-model quality — High for cooling,
Medium for dewpoint *because R² = 0.09*, and explicit exclusion for hold pressure *because the CI
crosses zero*. **Correction:** derive strength from standardised effect size × CI exclusion of zero
× identification quality, not from p alone.

---

## 3. Summary table

| ID | Variable / issue | Original | LeverGuide | Estimand affected | Severity |
|---|---|---|---|---|---|
| D-1 | `mold_temperature_c` not adjustable-for | required confounder for cooling | never in any adjustment set | total effect of cooling, −33 % | **Critical** |
| D-2 | `hold_pressure_bar` | β −0.087, CI crosses 0, not actioned | β +0.151 p.p./SD, "strong", rank 5 | sign reversal | **High** |
| D-3 | mediators in adjustment sets | excluded by rule | all three included | total → direct effect | **High** (struct.) |
| D-4 | `dryer_dewpoint_c` | lever, action #3 | confounder — not estimable | not estimated | **High** |
| D-5 | `maintenance_days_since_last` | planning lever, action #5 | context — not estimable | not estimated | **High** |
| D-10 | fixed effects | machine/mould/variant/shift | none | panel structure ignored | **High** |
| D-6 | `calibration_drift_index`, `tool_wear_index` | mediators | context | over-adjustment + no conditioning | Medium |
| D-7 | `clamp_force_kn` | context, "NOT a lever" | controllable, rank 6 | recommends a non-lever | Medium |
| D-8 | `cycle_time_s` | outcome / leakage | mediator | latent leakage on relabel | Medium |
| D-11 | uncertainty | 300× bootstrap | i.i.d. OLS CI | CIs too narrow | Medium |
| D-12 | evidence strength | multi-factor judgement | p-value threshold | 6/8 badged "strong" | Medium |
| D-9 | `part_weight_g` | context, in 3 adj. sets | mediator, dropped | under-adjustment | Low–Med |

---

## 4. On the specific question "are mediators incorrectly adjusted for when estimating total effects?"

**Yes — confirmed, and by a different mechanism than the code suggests.**

The *mechanism* in `dag.py:122-123` is correct: anything in the `mediators` list is discarded from
the adjustment set. The failure is at the **role-assignment layer**: the demo configuration labels
all three of the dataset's genuine mediators as `confounder`/`context`, so the guard never sees
them. `analyze/page.tsx:344` then tells the user "Mediators are excluded", which is true of the code
and false of the analysis being displayed.

The *measured* consequence on `cooling_time_s` is small (<1 %) because cooling does not act through
moisture, drift, or wear. The consequence on the dewpoint and maintenance pathways is total, but
only because those levers were removed from the lever set entirely (D-4, D-5) — so there is no
biased estimate to measure, only a missing one. Both facts should be stated together; reporting only
the first understates the design defect, reporting only the second overstates the numeric harm.

---

## 5. Reproduction

Scripts used (read-only; they import LeverGuide's modules and change nothing):

```bash
cd "/Users/abdulrahmanahmad/Desktop/My Projects/lever-guide-v2 2/apps/api"
./.venv/bin/python /private/tmp/claude-502/-Users-abdulrahmanahmad-Desktop-My-Projects/5b4981c1-c79c-4d81-bdd1-8e1cec784f93/scratchpad/verify_leverguide.py
./.venv/bin/python /private/tmp/claude-502/-Users-abdulrahmanahmad-Desktop-My-Projects/5b4981c1-c79c-4d81-bdd1-8e1cec784f93/scratchpad/verify_decomp.py
```

Reconciliation ladder for `cooling_time_s`, 2,000-row sample, outcome in p.p.:

| Specification | β (p.p./SD) |
|---|---|
| LeverGuide as shipped | **−1.171** |
| LeverGuide adjusters **+ `mold_temperature_c`** | −1.840 |
| LeverGuide adjusters **− the three mediators** | −1.160 |
| Datathon adjusters, no FE | −1.771 |
| Datathon adjusters **+ FE** (= paper method) | −1.826 |
| Datathon method on all 5,000 rows | **−1.743** (published: −1.743) |

The dominant term is the omitted `mold_temperature_c`, not the mediator over-adjustment.
