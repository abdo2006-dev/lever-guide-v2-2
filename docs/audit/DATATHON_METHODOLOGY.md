# Datathon Methodology — Reconstruction

**Source:** `datathon-CUB-2026-main/` (extracted from `~/Downloads/datathon-CUB-2026-main.zip`)
**Audit date:** 2026-08-01
**Status of every claim below:** confirmed by reading the file cited, unless explicitly marked *hypothesis*.

The PDFs (`report/Injection_Molding_Paper_v3_Final.pdf`, `data/Injection_Molding_DAG_Notes.pdf`,
`data/Datathon_Student_Guide.pdf`) were **not** parsed in this audit. Everything below is
reconstructed from the code, notebooks (including their stored outputs), the ontology JSON, and
the markdown READMEs. Where the repository quotes paper values, they are labelled as such.

---

## 1. Business and analytical question

| Item | Value | Evidence |
|---|---|---|
| Domain | Injection moulding, 4 plants, 12 machines, 18 moulds, 8 product variants | `data/README.md:5-11` |
| Unit of analysis | One 30-minute production interval per (machine × mould × variant × resin lot) | `data/README.md:7` |
| N | 5,000 rows × 33 columns, 2026-01-05 → 2026-03-20, zero missing values | `notebooks/01_eda.ipynb` cell 4 output |
| Business question | "What operational, environmental, material, and tooling factors are *causing* scrap, and which controllable levers reduce it most effectively?" | `data/injection_molding_ontology.json` → `primary_business_question` |
| Framing | Explicitly an **intervention** question, not a prediction question | `README.md:14` |
| Baseline | Mean scrap 4.44 %, median 4.24 %, σ 1.52 p.p.; 78.2 % of intervals fail the 3.2 % threshold | `notebooks/01_eda.ipynb` cell 8 output |

The problem is framed as *chronic*, not *anomalous*: "most intervals are already failing … the task
is to lower the entire scrap distribution" (`README.md:12`). This framing is what justifies causal
inference over outlier detection.

## 2. Variable roles

All roles are declared in one place: **`src/utils.py:23-73`**. This is the single most reusable
artefact in the Datathon repository.

| Role | Variables | Line |
|---|---|---|
| Controllable process levers | `dryer_dewpoint_c`, `barrel_temperature_c`, `mold_temperature_c`, `injection_pressure_bar`, `hold_pressure_bar`, `cooling_time_s`, `screw_speed_rpm`, `shot_size_g` | `src/utils.py:23-27` |
| Planning lever (not a per-interval setpoint) | `maintenance_days_since_last` | `src/utils.py:30` |
| Confounders | `ambient_humidity_pct`, `ambient_temperature_c`, `operator_shift` | `src/utils.py:36-38` |
| **Mediators — must never enter an adjustment set for a total effect** | `resin_moisture_pct`, `calibration_drift_index`, `tool_wear_index` | `src/utils.py:42-44` |
| Operator covariate (adjusted, never a target) | `operator_experience_level` | `src/utils.py:47` |
| Context (condition on, never intervene) | `product_variant`, `cavity_count`, `part_weight_g`, **`clamp_force_kn`** | `src/utils.py:51-53` |
| Batch quality covariate | `resin_batch_quality_index` | `src/utils.py:58` |
| Identifiers | `timestamp`, `plant_id`, `machine_id`, `mold_id`, `product_variant`, `resin_lot_id` | `src/utils.py:60-63` |
| Outcomes / post-treatment | `scrap_rate_pct`, `scrap_count`, `defect_type`, `pass_fail_flag`, `parts_produced`, `energy_kwh_interval`, **`cycle_time_s`** | `src/utils.py:65-69` |
| Fixed effects absorbed in every regression | `machine_id`, `mold_id`, `product_variant`, `operator_shift` | `src/utils.py:72` |

Two role decisions carry the most methodological weight and are called out explicitly in the source:

* **`cycle_time_s` is an outcome, not a predictor.** `src/utils.py:68` — "mechanically subsumes
  `cooling_time_s` → excluded from predictors". Reinforced in `data/README.md:96` and enforced in
  `src/causal_helpers.py:137` (`exclude = set(IDENTIFIERS + OUTCOMES + ["cycle_time_s"])`).
* **`clamp_force_kn` is context, not a lever.** `src/utils.py:50` — "Note: `clamp_force_kn` is NOT
  listed as a controllable lever in the paper." It appears only as a covariate in the adjustment set
  for `injection_pressure_bar` (`src/utils.py:95`).

### Colliders

No variable is labelled "collider" anywhere in the repository. The mechanism the code guards
against is stated as post-treatment / mediator conditioning, not collider stratification
(`notebooks/02_dag_causal_analysis.ipynb` cell 5). `defect_type` and `pass_fail_flag` are
descendants of both the levers and scrap and are placed in `OUTCOMES`, which is the practical
equivalent of collider avoidance. *Hypothesis:* `defect_type` is the clearest latent collider
candidate — it is caused by both process settings and by the same latent defect intensity that
drives `scrap_rate_pct`. The repository never conditions on it, so no bias is introduced, but the
reasoning is never made explicit.

### Time and grouping

`timestamp` (75 days, half-hourly), `plant_id`, `machine_id`, `mold_id`, `resin_lot_id`,
`operator_shift`. Grouping is handled **only** through fixed effects in the regressions
(`src/utils.py:72`); no grouped or time-based resampling is used anywhere. See §7.

## 3. Causal assumptions and identification

Stated target estimand (`notebooks/02_dag_causal_analysis.ipynb` cell 8):

```
ACE(T, t1, t0) = E[Y | do(T=t1)] − E[Y | do(T=t0)]
```

identified by the **back-door criterion** — condition on a set Z that blocks all non-causal paths
from T to Y and contains no descendant of T (Pearl 2000 §3.3, cited at `README.md:142`).

Adjustment sets are hand-specified per lever from the challenge ontology, not discovered from data
(`src/utils.py:80-113`). Fixed effects are always added on top.

| Lever | Adjustment set (plus machine/mould/variant/shift FE) |
|---|---|
| `cooling_time_s` | `mold_temperature_c`, `part_weight_g`, `shot_size_g`, `ambient_humidity_pct`, `ambient_temperature_c`, `maintenance_days_since_last` |
| `mold_temperature_c` | `cooling_time_s`, `barrel_temperature_c`, `part_weight_g`, `ambient_humidity_pct`, `ambient_temperature_c` |
| `barrel_temperature_c` | `injection_pressure_bar`, `mold_temperature_c`, `resin_batch_quality_index`, ambient ×2 |
| `injection_pressure_bar` | `tool_wear_index`, `clamp_force_kn`, `barrel_temperature_c`, `resin_batch_quality_index`, `part_weight_g`, `hold_pressure_bar`, ambient ×2 |
| `hold_pressure_bar` | `injection_pressure_bar`, `tool_wear_index`, ambient ×2 |
| `dryer_dewpoint_c` | `ambient_humidity_pct`, `ambient_temperature_c` |
| `maintenance_days_since_last` | `ambient_humidity_pct`, `ambient_temperature_c` |
| `screw_speed_rpm` | `shot_size_g`, `barrel_temperature_c`, ambient ×2 |

Note the deliberate asymmetry: the sets for the two levers whose effects run **through a mediator**
(`dryer_dewpoint_c` → moisture, `maintenance_days_since_last` → drift) are minimal — exactly two
ambient confounders — precisely so the mediated path stays open and a *total* effect is recovered.

The key causal pathways come from the challenge ontology
(`data/injection_molding_ontology.json` → `key_causal_pathways`):

```
ambient_humidity → dryer_dewpoint → resin_moisture → splay_moisture → scrap
mold_temperature + cooling_time → warpage risk → scrap
maintenance_days → calibration_drift → instability → scrap
tool_wear + injection_pressure + clamp_force → flash risk → scrap
resin_batch_quality + barrel_temperature + injection_pressure → short_shot risk → scrap
operator_shift → calibration_drift / operator_experience → cooling_time and stability → scrap
```

## 4. Estimator and uncertainty

`src/causal_helpers.py:36-86` — `estimate_adjusted_effect()`:

* One separate OLS per lever, with that lever's own adjustment set plus FE dummies
  (`add_fixed_effects` → `pd.get_dummies(..., drop_first=True)`, line 29).
* Two coefficients are reported, and the file documents *why they are inconsistent with each other*:
  * `beta_unstd` — direct OLS in original units, natural-unit effect (p.p. per unit), line 59.
  * `beta_std` — Frisch–Waugh partialling with a **z-scored lever and an unscaled outcome**,
    lines 62-67. Units are therefore **p.p. per SD of lever**, *not* dimensionless.
* The module docstring (`src/causal_helpers.py:6-17`) states plainly that these two are numerically
  inconsistent with the paper's Equation 5 and that "the same inconsistency exists in the paper. We
  report both values honestly." This is unusually good practice and is a preservation candidate.
* **Uncertainty:** 300-replicate **row-wise i.i.d. bootstrap** of the FW residual product,
  percentile 2.5/97.5 CI (lines 70-77). The repository repeatedly flags that machine-clustered
  errors would be wider (`README.md:158`, `notebooks/02` cell 15, `notebooks/03` cell 25).

### Reproduced estimates (`notebooks/02_dag_causal_analysis.ipynb` cell 10 output)

| Lever | β (p.p./SD) | 95 % CI | β̃ (p.p./unit) | Paper β |
|---|---|---|---|---|
| `cooling_time_s` | **−1.743** | [−1.851, −1.622] | −0.4063 | −1.75 |
| `mold_temperature_c` | +0.879 | [+0.789, +0.963] | +0.0812 | +0.88 |
| `injection_pressure_bar` | +0.324 | [+0.221, +0.416] | +0.0023 | +0.31 |
| `barrel_temperature_c` | −0.136 | [−0.195, −0.083] | −0.0101 | — |
| `maintenance_days_since_last` | +0.103 | [+0.066, +0.137] | +0.0127 | +0.10 |
| `dryer_dewpoint_c` | +0.095 | [+0.060, +0.133] | +0.0275 | +0.09 |
| `hold_pressure_bar` | −0.087 | [−0.180, **+0.007**] | −0.0009 | — |
| `screw_speed_rpm` | +0.002 | [−0.037, +0.045] | +0.0002 | — |

I re-ran this estimator independently during the audit and reproduced `cooling_time_s = −1.7429`
p.p./SD on all 5,000 rows (see `ML_EVALUATION_AUDIT.md` §1 for the command).

## 5. The headline finding — the cooling-time sign reversal

`notebooks/02_dag_causal_analysis.ipynb` cells 17-19:

| Quantity | Value |
|---|---|
| Raw Pearson ρ(cooling, scrap) | **+0.278** |
| Partial ρ after adjusting for `mold_temperature_c` only | −0.122 (paper: −0.37) |
| Fully adjusted β | **−1.743** p.p./SD |

Mechanism given (cell 19): operators *reactively extend cooling when they observe high mould
temperatures*. The raw positive correlation is that compensation behaviour, not a forward effect.
This is the analysis's central scientific claim, and it is stated with the right epistemic hedge:
"a DAG-informed estimate under the stated identification assumptions — not an experimental
coefficient."

Supporting interaction evidence, `notebooks/02` cell 22 (warpage subset, n = 1,650):

| mould temp \ cooling | ≤12 s | 12–18 s | >18 s |
|---|---|---|---|
| ≤65 °C | 3.62 | 3.81 | 4.54 |
| 65–75 °C | 5.90 | 4.33 | 4.44 |
| >75 °C | **8.20** | 5.81 | **5.06** |

−3.1 p.p. from extending cooling where mould temperature is highest.

## 6. Predictive model and its role

`src/causal_helpers.py:125-157` — `train_gbr()`:

* `GradientBoostingRegressor(n_estimators=400, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=42)`
* Feature set = all numeric columns **minus** identifiers, outcomes, and `cycle_time_s` (line 137).
  **Mediators are deliberately included** (line 131 docstring) — because the GBR's job is to be a
  world-approximator for simulation, not an effect estimator.
* Evaluation: `cross_val_score(..., cv=5, scoring="r2")` → **0.629 ± 0.029** (`notebooks/03` cell 4).
  Plain 5-fold KFold; **no grouping, no time ordering**.
* Feature importance is displayed with the title "Predictive gain; **NOT a causal ranking**"
  (`notebooks/03` cell 5) and cell 6 explains why `ambient_humidity_pct` ranking high does not imply
  intervening on humidity would help. This separation of predictive from causal importance is the
  single clearest piece of scientific communication in either repository.

## 7. Mediator sub-models and chain propagation

`src/causal_helpers.py:105-122` (`train_sub_model`) and `src/intervention_helpers.py:59-106`.

| Chain | Sub-model | CV R² |
|---|---|---|
| dewpoint → moisture → scrap | `resin_moisture_pct ~ ambient_humidity_pct + dryer_dewpoint_c + resin_batch_quality_index + ambient_temperature_c` | **0.0896** |
| maintenance → drift → scrap | `calibration_drift_index ~ maintenance_days_since_last + ambient_humidity_pct + ambient_temperature_c` | **0.7845** |

The **delta-propagation method** (`src/intervention_helpers.py:21-24, 76-86`) is the most
sophisticated idea in the Datathon repository and deserves to survive verbatim:

1. Predict the mediator with the **original** lever values.
2. Predict the mediator with the **shifted** lever values.
3. `delta = shifted − original`.
4. **Add the delta to each row's observed mediator**, rather than replacing the observed value with
   the sub-model's absolute prediction.

Rationale, quoted from the source: this "respects the R²=0.09 of the sub-model by only propagating
the CHANGE, not replacing the measured moisture with the sub-model's (noisy) absolute prediction."
For the dewpoint intervention the dewpoint column itself is *not* changed in the main GBR — only
the propagated moisture delta is applied — because the DAG says dewpoint acts only through moisture
(`src/intervention_helpers.py:14-16`).

## 8. Counterfactual simulation and process constraints

`src/intervention_helpers.py:109-176` — `counterfactual_shift()`:

* Per-row intervention: `delta` (additive shift) **or** `target_value` with optional `cap_only`
  (only rows already above the cap are moved) — lines 146-157.
* Conditional application: `condition_col` / `condition_threshold` / `condition_direction`,
  so a rule can be "only when humidity ≥ 65 %" — lines 136-141.
* **Hard physical bounds** from the ontology, `LEVER_RANGES` at lines 32-43:
  cooling 5–40 s, mould temp 40–110 °C, barrel 180–310 °C, injection 600–1800 bar,
  hold 300–1200 bar, screw 20–150 rpm, dewpoint −50…−20 °C, shot 50–2000 g,
  clamp 500–4400 kN, maintenance 1–60 d. Applied via `_clip()` before prediction.
* PATE = mean of (ŷ_intervened − ŷ_observed) **over the intervened rows only** (line 167).
* `simulate_combined_package()` (lines 179-242) applies all five levers simultaneously to one
  matrix and then runs the chains **once**, so lever interactions are captured by the GBR's
  response surface rather than assumed additive.

**Limitation of the original that LeverGuide inherits:** the GBR used for simulation is fit on all
rows (`src/causal_helpers.py:153`) and the counterfactual is evaluated on those same rows. The
5-fold CV R² is reported honestly alongside, but the PATEs themselves carry no interval.

## 9. Recommendations, trade-offs, limitations

`notebooks/03_intervention_tradeoff_analysis.ipynb` cells 10-11:

| Action | Specification | Repo PATE | Paper PATE | Confidence |
|---|---|---|---|---|
| Extend cooling | +1.5 s plant-wide | −0.389 | −0.44 | High |
| Cap mould temp | ≤78 °C, cap-only | −0.139 | −0.21 | Med–High |
| Lower dryer dewpoint | −5 °C **when humidity ≥ 65 %** (chained through moisture) | −0.090 | −0.09 | Medium |
| Reduce injection pressure | −30 bar **when tool_wear_index ≥ 0.45** | −0.065 | −0.09 | Medium |
| Tighten maintenance | cap at 14 d (chained through drift) | −0.019 | −0.07 | Medium |
| **Combined package** | all five at once | **4.442 % → 3.884 %, −12.6 %** | −13 % | — |

Explicitly **not** recommended (`notebooks/03` cell 23): shortening cooling; plant-wide HVAC for
humidity; reassigning experienced operators (positive β is assignment bias); blanket pressure
reduction on lightly worn tooling. Stating the anti-recommendations is a strong feature.

Stated limitations (`README.md:155-162`, `notebooks/03` cell 25): observational data requiring a
pilot; row-wise bootstrap understates uncertainty vs machine-clustered; moisture sub-model R²=0.09
makes the dryer estimate a lower bound; Jan–Mar temporal scope; no cost data; DAG assumed correctly
specified.

## 10. Defects found **in the Datathon repository itself**

These are confirmed inconsistencies in the source of truth, and they should be fixed before the
Datathon analysis is used as the reference implementation for a rebuild.

| # | Defect | Evidence | Severity |
|---|---|---|---|
| D1 | Cycle-time trade-off is **hard-coded**, contradicting the computed value. Cell 18 computes `+2.79 %` then prints "Paper reports: +1.0 %", and the summary table at cell 22 writes the literal string `≈{baseline+1.5} s (+1.0%)`. The +1.0 % figure is not reproducible from this dataset (1.5 / 53.75 = 2.79 %). | `notebooks/03` cells 18, 22 | High |
| D2 | The headline trade-off verdict contradicts its own output. Cell 19 computes a scrap-to-cycle ratio of **0.2×** (paper: ~10×) and then prints "Assessment: highly favourable trade-off … pays back many times over". The narrative is hard-coded and unaffected by the number above it. | `notebooks/03` cell 19 | **Critical** — this is the trade-off conclusion |
| D3 | EDA summary table claims `mold_temperature_c` ρ ≈ +0.55; the computed value in the cell directly above is **+0.412**. | `notebooks/01` cells 22 vs 29 | Medium |
| D4 | Partial ρ after adjusting for mould temperature reproduces at −0.122 vs the paper's −0.37 — a 3× gap that the notebook prints but never explains, unlike the other gaps which are all discussed. | `notebooks/01`/`02` cell 17 | Medium |
| D5 | `train_gbr` is reused for the energy outcome (cell 18) and prints "paper reports 0.64" for an unrelated model that scores 0.974 — a hard-coded log string leaking into a different context. | `src/causal_helpers.py:155-156` | Low |
| D6 | No tests of any kind. No `tests/` directory, nothing in `requirements.txt` for testing. | repository listing | High for reuse |
| D7 | Bootstrap resamples the **residualised** vectors, not the rows, and refits nothing. This is a fixed-design bootstrap: it captures sampling noise in the FW second stage but not in the first-stage partialling. The README calls it "row-wise bootstrap", which overstates what is resampled. | `src/causal_helpers.py:70-77` | Medium |
| D8 | Notebooks depend on `os.chdir(repo_root)` walking up from the kernel CWD, and the stored outputs contain an absolute path from the author's machine. Not reproducible from a clean checkout without Jupyter. | `notebooks/0*` cell 2 | Medium |

None of D1–D8 undermine the **direction** of the five recommendations, which I independently
reproduced. D1/D2 do undermine the *trade-off* conclusion as stated.

## 11. Verdict

The Datathon analysis is scientifically serious. Its distinguishing strengths — the ones worth
migrating — are, in order of value:

1. The variable-role taxonomy with its per-lever adjustment sets (`src/utils.py`).
2. Delta-propagation through mediator sub-models (`src/intervention_helpers.py`).
3. Constrained, conditional, cap-only interventions with ontology-derived physical bounds.
4. The explicit separation of predictive importance from causal effect, stated in the UI-facing text.
5. The habit of documenting a discrepancy rather than hiding it (`src/causal_helpers.py:6-17`,
   `README.md:83-109`).

Its weaknesses are: no grouped/temporal validation, understated uncertainty, no tests, no
reproducible entry point outside Jupyter, and the hard-coded narrative defects D1–D3.
