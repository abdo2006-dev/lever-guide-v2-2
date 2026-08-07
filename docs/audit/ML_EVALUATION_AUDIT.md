# ML Pipeline & Evaluation Audit

All measurements from the demo dataset (`apps/web/public/demo/injection_molding_demo.csv`,
5,000 rows) using the committed venv at `apps/api/.venv` (Python 3.10.4, scikit-learn 1.6.0).

---

## 1. Findings

### M-1 — Preprocessing fitted before the split — **Medium (structural), no measured harm here**

`preprocess.py:119` runs `ct.fit_transform(df_feat)` on **all** rows — median imputation and
`StandardScaler` — and only then does `pipeline.py:56` call `train_test_split`. Test-set statistics
leak into the transformer.

**Measured:** scale-then-split test R² = **0.4776**; split-then-scale (`make_pipeline`) test R² =
**0.4776**. Difference **0.0000**. With 2,000 rows and well-behaved distributions, mean/σ estimated
on 100 % vs 90 % of rows are indistinguishable. Reporting this honestly: the defect is real and
would matter with small n, heavy tails, or fitted imputation, but on this dataset it costs nothing
measurable. Fix it because it is wrong, not because it is currently harmful.

**Fix:** wrap the `ColumnTransformer` in a `Pipeline` with the estimator and fit inside each fold.

### M-2 — Winner selected on the test set — **High**

`pipeline.py:172-173`: `results.sort(key=lambda r: r.metrics.r2, reverse=True); results[0].is_winner = True`.
`metrics.r2` is computed on `y_te` (`pipeline.py:68, 94, 113, …`). There is no validation split. The
reported "Best Model R²" that headlines the Overview tab and the executive summary is therefore the
score of the model chosen *because* it scored highest on that same held-out set — an optimistically
biased number, by construction.

**Measured:** RF 0.4807 / Ridge 0.4119 / OLS 0.4118 on n_test = 200. With 200 test rows the standard
error on R² is roughly ±0.05, so the ranking between Ridge and OLS (Δ = 0.0001) is pure noise being
presented as a winner selection.

**Fix:** three-way split, or nested CV — select on inner-fold CV, report the outer held-out score.

### M-3 — CV computed on the full matrix including test rows — **Medium**

`pipeline.py:70, 96, 116`: `cross_val_score(model, X, y, cv=3)` uses the **whole** `X, y`, not
`X_tr, y_tr`. The "CV R²" column shown next to the test R² in the UI (`analyze/page.tsx:149-152`)
therefore includes the test rows in its training folds.

**Measured:** cv_r2 as shipped = **0.4455**; on training rows only = **0.4570**. The contamination
is small and here it happens to *lower* the number, but the two quantities are being displayed
side-by-side as if independent, which they are not.

### M-4 — Random row split on grouped, time-ordered panel data — **High**

`pipeline.py:56` uses `train_test_split(..., random_state=seed)`. The data is 5,000 half-hourly
intervals across 12 machines / 18 moulds / 4 plants over 75 days. Rows from the same machine on the
same day land on both sides of the split.

**Measured heterogeneity:** mean scrap by machine spans 3.824 % (DEN_IM_03) → 5.533 % (NAM_IM_02).
Rows per machine 348–552.

**Measured optimism** (RandomForest, identical hyperparameters, all 5,000 rows):

| Scheme | R² | σ |
|---|---|---|
| random row split (KFold-5, shuffled) | **+0.4913** | 0.0275 |
| repeated CV (5-fold × 4 repeats) | +0.4933 | 0.0196 |
| grouped CV by `mold_id` (GroupKFold-5) | +0.4824 | 0.0281 |
| time-based forward chaining (4 expanding folds) | +0.4610 | 0.0425 |
| **grouped CV by `machine_id` (GroupKFold-5)** | **+0.4369** | 0.0591 |
| group hold-out by machine (GroupShuffleSplit-5) | +0.4131 | **0.0728** |

The random split overstates generalisation by **+0.05 to +0.08 R²** and understates its own
variability by **2.6×** (σ 0.028 vs 0.073). For a product whose promise is "this lever will work on
your line", machine-level generalisation is the quantity that matters.

### M-5 — Test set passed into `fit()` — **Medium**

`pipeline.py:135`: `xgb_m.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)`. Without
`early_stopping_rounds` this does not alter the fitted model, so it is not leakage today — but it is
a one-flag change away from becoming leakage, and it means the test set is an input to the training
call. Remove it.

### M-6 — Silent model failures — **High**

Five bare `except Exception: pass` blocks (`pipeline.py:86, 105, 126, 145, 166`) plus two import
guards (`pipeline.py:15-23`). **Verified live:** in the committed venv both boosters fail —
`xgboost` → `XGBoostError: libxgboost.dylib could not be loaded … libomp.dylib`; `lightgbm` →
`OSError: Library not loaded: @rpath/libomp.dylib`. The pipeline returned `['rf','ridge','ols']`,
the bundle contained **no warning**, and the UI kept displaying "Running: OLS · Ridge · Random
Forest · XGBoost · LightGBM" (`setup/page.tsx:411`) and "Trains 5 models" (line 374). The user
cannot tell that 40 % of the advertised comparison did not happen.

**Fix:** catch, append to `bundle.warnings`, and render a per-model status.

### M-7 — Ordinal encoding of nominal categoricals — **Medium**

`preprocess.py:110-113` applies `OrdinalEncoder` then `StandardScaler` to every categorical.
`product_variant` (8 nominal part families) becomes 0–7 with an imposed ordering, and
`operator_shift` (A/B/C) becomes 0–2. Tree models can partially recover from this with enough splits;
OLS and Ridge cannot — they fit a single slope across an arbitrary alphabetical ordering. Since
`operator_shift` is one of the Datathon's three named confounders, mis-encoding it degrades exactly
the adjustment the product depends on.

**Fix:** one-hot for low-cardinality nominals; keep ordinal only for genuinely ordered scales
(`operator_experience_level`, `cavity_count`).

### M-8 — Feature importances compared across incompatible scales — **Medium**

`_importance_list` (`pipeline.py:37-43`) normalises each model's values by its own max and the UI
plots them on a shared 0–1 axis (`analyze/page.tsx:253`). But `|OLS standardised coefficient|`,
`RandomForest` mean impurity decrease, `XGBoost` gain and `LightGBM` split count are four different
quantities. The chart caption ("For tree models = mean impurity decrease. For linear = |standardised
coefficient|") is honest about *what* they are but the shared normalisation invites the comparison
anyway. Impurity-based importance is additionally biased toward high-cardinality continuous
features, which is most of this dataset.

**Fix:** permutation importance on held-out folds, with a CI; or drop the cross-model chart.

### M-9 — Silent 60 % row discard — **Medium**

`analysis.py:258-259`: `df.sample(2_000, random_state=req.random_seed)` on the 5,000-row demo. It is
disclosed in the setup UI (`setup/page.tsx:250`) and the README, so it is not hidden — but it is a
**simple random sample of a time-ordered panel**, which destroys the temporal ordering and the
per-machine balance. Every published Datathon number is computed on all 5,000 rows, so no
LeverGuide figure is directly comparable to the paper.

**Fix:** stratify by machine × time-bucket, or precompute on the full dataset (see
`TARGET_ARCHITECTURE.md`).

### M-10 — No residual diagnostics — **Medium**

`predictions` carries at most 400 `(actual, predicted, residual)` triples (`pipeline.py:82`) and the
UI renders only an actual-vs-predicted scatter of the first 300 (`analyze/page.tsx:213`). There is
no residual-vs-fitted plot, no QQ plot, no heteroskedasticity check, no residual-vs-machine or
residual-vs-time panel — despite the OLS confidence intervals in the causal tab depending on
homoskedasticity, which the UI itself states (`analyze/page.tsx:345`).

### M-11 — No subgroup or drift analysis — **Medium**

Nothing computes performance or effects by plant, machine, mould, shift, or time window. Yet the
Datathon's pilot recommendation (VN_QUANGNAM) is entirely a subgroup argument, its dryer rule is
conditional on humidity, and its stated limitation is seasonal drift (Jan–Mar data). The single most
recruiter-legible chart the product could show — "does this lever work on every machine?" — does not
exist.

### M-12 — Reproducibility gaps — **Medium**

`random_seed` is threaded correctly through the split, RF, XGB, LGBM, GBR and the row sample. What
is *not* pinned: BLAS thread count, `n_jobs=-1` for RF (non-deterministic reduction order in some
builds), library versions between the pinned `requirements.txt` (Python 3.12 on Render) and the
committed venv (Python 3.10.4), and `wandb` which is unpinned entirely. No lockfile for Python. No
CI to detect drift. No artifact recording which model actually ran.

### M-13 — No target-leakage guard — **Medium**

`cycle_time_s` is excluded today only as a side effect of being labelled `mediator`
(`analysis.py:302-305` selects `controllable|confounder|context`). A user relabelling it in the
setup dropdown introduces immediate leakage — it mechanically contains `cooling_time_s` — with no
warning. `scrap_count`, `parts_produced` and `pass_fail_flag` are guarded the same way (role
`ignore`) and are equally one dropdown away from leaking. `pass_fail_flag` is a **deterministic
function of the target** (`scrap_rate_pct > 3.2`).

**Fix:** a leakage checker that flags near-deterministic relationships to the target and refuses
predictor roles for them.

---

## 2. Split-scheme comparison and recommendation

| Scheme | What it estimates | R² (measured) | σ | Verdict for this dataset |
|---|---|---|---|---|
| Random row split | performance on a *new interval from a machine already seen* | 0.491 | 0.028 | Optimistic; the current default. Keep only as a labelled "in-distribution" reference. |
| Group split by machine (hold-out) | performance on a *machine never seen* | 0.413 | 0.073 | The honest headline number, but a single split of 12 groups is high-variance. |
| Time-based split | performance on a *future period* | 0.461 | 0.043 | Essential as a **secondary** check — the data spans one season and drift is a stated limitation. |
| **Grouped CV by machine (GroupKFold-5)** | machine-level generalisation, averaged over 5 partitions | **0.437** | 0.059 | **Recommended primary.** |
| Repeated CV | tightens the estimate of the *random-split* quantity | 0.493 | 0.020 | Reduces the wrong estimator's variance. Not a substitute. |

**Recommended primary scheme: `GroupKFold(n_splits=5, groups=machine_id)`**, with
`RepeatedStratifiedGroupKFold` if scrap terciles need balancing.

Reasoning, specific to this dataset:

1. **The unit of deployment is a machine.** The product's claim is "apply this setpoint change on
   your line." A random split answers a question nobody asked.
2. **Machines are the dominant heterogeneity.** 3.82 %–5.53 % mean scrap between machines; the
   Datathon absorbs `machine_id` as a fixed effect for exactly this reason.
3. **12 groups × ~420 rows** is enough for 5 folds without any fold being degenerate.
4. **It is the conservative choice.** 0.437 vs 0.491 — reporting the lower number and explaining
   why is the single strongest credibility signal an ML case study can send.

**Supporting scheme: a time-based forward-chaining split**, reported alongside, not instead. The
dataset covers 75 days of one season; the Datathon explicitly flags seasonal recalibration as a
limitation. A "trained on Jan–Feb, tested on Mar" number directly addresses it and is cheap.

**Reject `mold_id` grouping as primary** (R² 0.482, barely different from random) — moulds are
shared across machines so the grouping does not isolate the deployment unit.

### Target evaluation design

```
GroupKFold(5, groups=machine_id)
  └─ per fold:
       fit ColumnTransformer  on TRAIN ONLY   (fixes M-1)
       inner GroupKFold(4) for model selection (fixes M-2)
       outer fold → held-out metrics
  report: mean ± σ across the 5 outer folds
  plus:   per-machine held-out R² (fixes M-11)
  plus:   forward-chaining time split, reported separately (fixes M-11 drift)
  plus:   residual-vs-fitted, residual-vs-machine, residual-vs-time (fixes M-10)
  plus:   permutation importance on held-out folds with CI (fixes M-8)
  plus:   explicit per-model status incl. skipped/failed (fixes M-6)
```

---

## 3. Commands run and outcomes

| Command | Outcome |
|---|---|
| `cd apps/api && ./.venv/bin/python -m pytest -q` | `27 passed in 23.72s` — **with xgboost and lightgbm both unimportable** |
| `./.venv/bin/python -c "import xgboost"` | `XGBoostError: libxgboost.dylib could not be loaded … libomp.dylib` |
| `./.venv/bin/python -c "import lightgbm"` | `OSError: Library not loaded: @rpath/libomp.dylib` |
| `verify_leverguide.py` | 3/5 models returned, no warning; test_size resolved to **0.1** (90/10) while the UI claims "80/20" |
| `verify_decomp.py` | reconciliation ladder in `SCIENTIFIC_DISCREPANCIES.md` §5 |
| `verify_intervention.py` | split-scheme table above; intervention optimism +0.26 R² |

Scripts live in
`/private/tmp/claude-502/-Users-abdulrahmanahmad-Desktop-My-Projects/5b4981c1-c79c-4d81-bdd1-8e1cec784f93/scratchpad/`.
No repository file was modified. `libomp` was **not** installed — the missing-dependency path is
reported as found, not worked around.

**Note on the 80/20 claim:** `pipeline.py:55` computes `test_size = min(0.2, max(0.1, 200/n))`. For
n = 2,000 this is 0.1. `analyze/page.tsx:139` tells the user "All models on the same 80/20
train/test split". Confirmed 1,800/200. The README is correct here ("between 10 % and 20 %"); only
the UI is wrong.
