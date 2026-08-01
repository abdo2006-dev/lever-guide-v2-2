# Intervention Engine Audit

Subject: `apps/api/app/models/intervention.py` (180 lines), invoked from `analysis.py:337`.
Reference: `datathon-CUB-2026-main/src/intervention_helpers.py`.

---

## 1. What kind of object is a LeverGuide "intervention"?

**Classification: a predictive what-if, ranked heuristically, presented with a causal badge.**

The generative logic (`intervention.py:104-117`) is:

```python
for sign, direction in [(+1, "increase"), (-1, "decrease")]:
    target_val = clip(cur_mean + sign*cur_std, lo_bound, p90*1.5)
    X_shifted = X.copy(); X_shifted[:, j] = target_val      # every row → same constant
    sim_impact = mean(gbr.predict(X_shifted)) - base_mean
    improves = (improve_direction == "decrease" and sim_impact < 0) or ...
    if improves and abs(sim_impact) > abs(best): keep
```

Three properties follow:

1. **The direction is chosen by the predictive GBR**, by trying both and keeping whichever the model
   likes. The adjusted OLS estimate is *never consulted for direction* — it is attached afterwards
   as an annotation (`intervention.py:122-139`).
2. **The magnitude is a difference of two in-sample GBR predictions**, not an identified causal
   quantity.
3. **The `evidence_type="causal"` badge** is awarded on `causal_row.p_value < 0.05`
   (`intervention.py:124`) with **no check that the OLS sign agrees with the recommended
   direction**.

So: neither a causal effect estimate (no identification is used in producing it), nor a structural
counterfactual (no structural equations, no mediator propagation), nor a pure heuristic (a fitted
model is involved). It is a **predictive what-if with an unvalidated causal label**.

**Measured consequence — sign conflicts shipped as recommendations.** Running the shipped engine on
the demo, two of seven recommendations point the opposite way from their own adjusted OLS estimate:

| Rank | Feature | GBR direction | OLS β/SD | OLS implies | Badge |
|---|---|---|---|---|---|
| **1** | `shot_size_g` | decrease | −0.0440 | **increase** | `mixed / weak` |
| 2 | `mold_temperature_c` | decrease | +0.2924 | decrease ✓ | `causal / strong` |
| 3 | `cooling_time_s` | increase | −0.7816 | increase ✓ | `causal / strong` |
| 4 | `injection_pressure_bar` | decrease | +0.1436 | decrease ✓ | `causal / strong` |
| 5 | `hold_pressure_bar` | decrease | +0.1008 | decrease ✓* | `causal / moderate` |
| 6 | `clamp_force_kn` | increase | −0.0569 | increase ✓ | `causal / moderate` |
| **7** | `screw_speed_rpm` | increase | +0.0114 | **decrease** | `mixed / weak` |

\* agrees with LeverGuide's own OLS, which itself has the wrong sign — see `SCIENTIFIC_DISCREPANCIES.md` D-2.

The **rank-1 recommendation is a sign conflict.** The `mixed/weak` badge is the only signal, and it
is a small grey chip beside a bold green "−9.4 % scrap_rate_pct".

---

## 2. Checklist

| Check | Result | Evidence |
|---|---|---|
| Evaluated out of sample? | **No** | `intervention.py:84` `gbr.fit(X, y)` on all rows; `line 88` `base_mean` from `gbr.predict(X)`; `line 109` counterfactual on the same `X`. |
| Same data for training and evaluation? | **Yes** | as above |
| Changed values inside observed support? | **Partly** — see I-3 | `intervention.py:105-106` |
| Coupled variables updated? | **No** | only column `j` is written |
| Mediator propagation matches original? | **No — absent entirely** | no analogue of `_apply_moisture_chain` / `_apply_drift_chain` |
| Impossible combinations generatable? | **Yes, and generated** — see I-4 | measured |
| Uncertainty reported? | **No** | `Intervention` schema has no CI/SE field |
| Domain constraints respected? | **No** | percentile heuristic only, no ontology bounds |
| Combined interventions / interactions? | **No** | one at a time; no package simulation |

### I-1 — In-sample counterfactuals, optimism never surfaced — **Critical**

**Measured:** the intervention GBR (`n_estimators=150, lr=0.08, depth=4, min_samples_leaf=10,
subsample=0.8`) scores **in-sample R² = 0.8006** but **5-fold CV R² = 0.5405 ± 0.0250**. Optimism
**+0.2601**. Neither number appears anywhere in the API response or the UI. Every KPI-change figure
the product headlines is read off a surface that is 26 R²-points better on these rows than on new
ones.

Note the Datathon has the same in-sample structure (`src/causal_helpers.py:153` fits on all rows)
but at least reports the CV R² of that exact model in the notebook (0.629 ± 0.029). LeverGuide
reports the CV of a *different* model (the predictive-pipeline RF) and never mentions this one.

**Fix:** fit the simulation model inside a `GroupKFold` and report out-of-fold PATEs, or at minimum
report the simulation model's own CV R² next to every PATE.

### I-2 — No uncertainty on any intervention — **Critical**

`schemas.py` `Intervention` fields: `rank, feature, direction, current_mean, current_p10,
current_p90, suggested_value, delta, delta_pct, expected_kpi_change, expected_kpi_change_pct,
evidence_strength, evidence_type, tradeoff, rationale, assumptions, caveat`. **No interval, no
standard error, no bootstrap.** Confirmed programmatically.

**Measured:** a 40-replicate bootstrap of the rank-1 counterfactual gives point **−0.4153** with a
95 % interval of **[−0.5149, −0.1663]** — a 3× spread, strongly asymmetric. The UI shows
`−0.4153` and `−9.4 %` as a bare number in bold.

Meanwhile `HomeClient.tsx:32-35` advertises "**Honest uncertainty** — Confidence intervals,
p-values, and model quality metrics always visible." True of the effect-estimates tab; false of the
interventions tab, which is the product's headline output.

### I-3 — Support bounds are a percentile heuristic, not physics — **High**

```python
lo_bound  = p10 - abs(p10)*0.5 if p10 <= 0 else p10*0.5     # intervention.py:105
target_val = clip(cur_mean + sign*cur_std, lo_bound, p90*1.5)
```

`p90 * 1.5` and `p10 * 0.5` are arbitrary. On the demo all seven suggested values happened to land
inside the observed min–max, so **no violation was produced this run** — but the bound is 50 %
outside the observed 10–90 range by construction and the sign-dependent `lo_bound` branch is
incoherent for negative-valued columns: for `dryer_dewpoint_c` (p10 ≈ −41 °C) it yields −61.5 °C,
below the ontology floor of −50 °C. That column is currently spared only because it is mislabelled
as a confounder and never reaches the engine (`SCIENTIFIC_DISCREPANCIES.md` D-4). Fix the role and
the bug becomes live.

The Datathon by contrast clips to declared physical ranges from the ontology
(`src/intervention_helpers.py:32-43`): cooling 5–40 s, mould 40–110 °C, dewpoint −50…−20 °C, etc.

### I-4 — Coupled variables not updated → physically impossible recommendations — **Critical**

The engine writes one column and leaves every physically coupled column at its observed value.

**Measured on the rank-1 recommendation.** `shot_size_g` is the material mass injected per cycle; it
is mechanically determined by cavity count and part weight.

```
corr(shot_size_g, cavity_count × part_weight_g) = +0.9989
shot_size_g / (cavity_count × part_weight_g): mean 1.080, min 0.972, max 1.214
```

LeverGuide's rank-1 action sets `shot_size_g = 66.344 g` for **every row** while `cavity_count` and
`part_weight_g` stay unchanged. **4,627 of 5,000 rows (92.5 %) require more than 66.3 g of material
to fill their cavities.** The engine's top recommendation is, for 92.5 % of production, "inject less
material than the part weighs" — a guaranteed short shot, which is itself one of the dataset's
defect classes. The GBR extrapolates happily because it has never seen this region.

This single finding is the clearest demonstration in the whole audit of why a constrained,
domain-aware simulator is not optional.

Secondary coupling the engine also ignores: `cooling_time_s` → `cycle_time_s` (the Datathon's entire
trade-off analysis); `cooling_time_s`/`mold_temperature_c` → `energy_kwh_interval`;
`injection_pressure_bar` ↔ `hold_pressure_bar` (packing profile).

### I-5 — "Hold others at their mean" is stated but not done — **Medium**

`intervention.py:148` emits the assumption "Other variables remain at their current mean values."
`analyze/page.tsx:424` repeats "holding all others at their mean". **The code does neither** — it
retains each row's observed values and overwrites only column `j` (`intervention.py:107-108`), then
averages. Retaining observed values is the *better* method (it is a marginal effect over the
empirical distribution, closer to a PATE). The documentation describes a worse method than the one
implemented. Both statements should be corrected to "each row keeps its own observed covariates;
only this lever is changed, and the change is averaged across rows."

### I-6 — No mediator chain propagation — **High**

The Datathon's most distinctive machinery has no counterpart:

| Datathon | LeverGuide |
|---|---|
| `_apply_moisture_chain` — dewpoint shift → moisture sub-model → **delta** added to observed moisture → main GBR; dewpoint itself not changed | absent |
| `_apply_drift_chain` — maintenance cap → drift sub-model → delta added to observed drift | absent |
| Delta rather than absolute prediction, to respect R² = 0.09 residual variance | absent |

Consequence, quoted from the Datathon's own notebook (`03` cell 12): "Without the chain, the direct
GBR effect of dewpoint is only −0.03 p.p." versus −0.0895 with it — a **3× understatement**. Any
future LeverGuide version that promotes `dryer_dewpoint_c` to a lever without porting the chain will
reproduce that understatement.

### I-7 — No conditional, cap-only or package interventions — **High**

The API schema cannot express any of the Datathon's five actions faithfully:

| Datathon action | Representable in LeverGuide? |
|---|---|
| cooling **+1.5 s** (additive delta, every row) | ✗ — only "set all rows to a constant" |
| mould temp **≤78 °C, cap-only** (move only rows above) | ✗ |
| dewpoint −5 °C **when humidity ≥ 65 %** | ✗ — no conditional |
| pressure −30 bar **when tool_wear ≥ 0.45** | ✗ — no conditional |
| maintenance **cap at 14 d** | ✗ — variable isn't a lever |
| **all five simultaneously**, chains applied once | ✗ — no package endpoint |

`simulate_combined_package` (`src/intervention_helpers.py:179-242`) is the piece that captures
lever *interactions* — it applies all shifts to one matrix and lets the GBR's response surface do
the compounding. LeverGuide has nothing equivalent, and its UI ranks seven independent
single-lever estimates that a reader will naturally sum. On the demo those seven sum to −1.33 p.p.
against a 4.44 p.p. baseline — a −30 % claim that no simulation supports.

### I-8 — Trade-offs are a hard-coded lookup — **Medium**

`intervention.py:31-42` is a five-entry dict keyed on `(direction, feature)` with a generic fallback
"Monitor downstream effects of {direction}ing {feature}." Nothing is computed. In the run above,
five of seven recommendations got the generic string. The Datathon quantifies its trade-offs
(cycle-time seconds, energy kWh, short-shot monitoring gates) — imperfectly (see
`DATATHON_METHODOLOGY.md` D1/D2) but numerically.

### I-9 — Ranking by predicted magnitude alone — **Medium**

`intervention.py:177`: `sort(key=lambda x: abs(x.expected_kpi_change))`. Not by effect size relative
to uncertainty, not by identification quality, not by feasibility. This is what floats a
sign-conflicted, physically impossible `shot_size_g` change to the top while `cooling_time_s` — the
one lever with strong, well-identified, mechanism-backed evidence — sits at rank 3.

### I-10 — Direction search is model-shopping — **Medium**

Trying both directions and keeping whichever the model prefers is a one-parameter optimisation over
a noisy surface, then reported as a finding. With CV R² 0.54 and no interval, for weak levers this
is close to selecting on noise: `screw_speed_rpm` (β = +0.011, p = 0.60) still produced a confident
"increase" recommendation.

---

## 3. Side-by-side

| Property | Datathon | LeverGuide |
|---|---|---|
| Object simulated | per-row shift, `delta` or cap, conditional | all rows set to one constant |
| Physical bounds | ontology `LEVER_RANGES` | `p90×1.5` / `p10×0.5` heuristic |
| Mediator propagation | delta method through two sub-models | none |
| Coupled variables | not updated either — **shared weakness** | not updated |
| Combined package | yes, chains applied once | no |
| Conditional rules | yes | no |
| Model fit | all rows, CV R² reported (0.629) | all rows, CV R² never computed (0.541 measured) |
| Uncertainty on PATE | none — **shared weakness** | none |
| Direction | from the identified DAG-adjusted estimate | from the predictive model, both tried |
| Ranking | by mechanism + effect + confidence judgement | by \|predicted Δ\| |
| Anti-recommendations | 4 explicit "do not do this" | none |

Note the two shared weaknesses — neither codebase puts an interval on a PATE, and neither updates
coupled variables. A rebuild should fix both rather than treating the Datathon as fully correct.

---

## 4. Recommended target design

1. **Separate the three objects and label them in the UI**: (a) *identified effect* — DAG-adjusted
   β with a clustered bootstrap CI; (b) *simulated policy* — constrained counterfactual on an
   out-of-fold model with a bootstrap interval; (c) *operational judgement* — feasibility, cost,
   monitoring gates. Never merge them into one number.
2. **Take direction from (a), magnitude from (b).** Refuse to emit a recommendation when they
   disagree; surface the disagreement as a finding — it is scientifically interesting and it is
   exactly the cooling-time lesson.
3. **Declare constraints in data, not code**: per-lever `[min,max]` from the ontology, coupling
   identities (`shot_size ≈ cavity_count × part_weight × k`), and monotone feasibility rules.
   Reject any counterfactual row that violates one, and report the rejection count.
4. **Port `_apply_moisture_chain` / `_apply_drift_chain` unchanged.** The delta method is correct
   and the reasoning behind it is the best teaching moment in the source material.
5. **Support the five Datathon action shapes**: additive delta, cap-only, conditional-on-column,
   package, and per-row target.
6. **Bootstrap every PATE** (≥200 replicates over rows, resampled by machine cluster) and render the
   interval, not the point.
7. **Never extrapolate**: clip to the joint observed support, and warn when a proposed row falls in
   a region with fewer than *k* observed neighbours.
