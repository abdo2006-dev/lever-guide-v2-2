# Open Questions & Unresolved Uncertainties

Everything below is something this audit could **not** settle. Each item states what is uncertain,
why it could not be resolved, what would resolve it, and what changes if the answer goes one way or
the other.

---

## A. Things I could not verify

### A-1 — The paper itself was never read

`report/Injection_Molding_Paper_v3_Final.pdf`, `data/Injection_Molding_DAG_Notes.pdf` and
`data/Datathon_Student_Guide.pdf` were not parsed. Every "paper value" in this audit is quoted from
the Datathon README and notebooks, which are secondary. The DAG Notes PDF in particular is described
as containing "why certain variables are confounders vs. mediators, and the identification
assumptions underlying the adjustment sets" (`report/README.md`) — i.e. the primary justification
for `src/utils.py`.

**Resolve by:** reading the three PDFs before Phase 2.
**Changes if different:** if the DAG Notes contradict `src/utils.py`, the ontology port in Phase 2
is wrong at its root and D-1 through D-9 would need re-deriving.

### A-2 — The −0.122 vs −0.37 partial-correlation gap

`notebooks/02` cell 17 computes ρ_partial(cooling, scrap | mould temp) = **−0.122** against a paper
value of **−0.37** — a 3× gap the notebook prints without comment, while every other gap in the repo
is explicitly discussed. Possible explanations: a different partialling set in the paper; a different
dataset seed; or an error on one side.

**Resolve by:** A-1, plus recomputing under several candidate specifications.
**Changes if different:** if the demo CSV is not the paper's dataset, then all magnitude comparisons
in this audit compare a reproduction to a *different* population, and only directions transfer.

### A-3 — Is the demo CSV the paper's dataset?

`data/README.md` and the filename both say **demo**. The Datathon README hypothesises "the paper may
have been produced on a different random seed of the synthetic generator … σ(cooling_time_s) = 4.3 s
here vs an implied ~6.5 s in the paper" (`README.md:106`). If true, several published PATEs are not
reproducible in principle, and the "⚠️ ~33 % gap" rows in the reproduction table are expected rather
than defects.

**Resolve by:** asking the challenge organisers, or checking whether the generator is published.
**Changes if different:** determines whether Phase 4's acceptance criterion should be "same sign and
rank" (my recommendation) or "same magnitude".

### A-4 — Whether the app has ever been deployed

Root `render.yaml` names `lever-guide.onrender.com`; `apps/api/render.yaml` names
`lever-guide.vercel.app`; the README lists "Stable public deployment link" under *Planned Work*.
No live URL was fetched during this audit (no network calls were made to either host).

**Resolve by:** attempting the URLs, or checking the Render/Vercel dashboards.
**Changes if different:** if something is live with the current science, D-1 and D-2 are publishing
wrong numbers now and Phase 1a becomes urgent rather than merely cheap.

### A-5 — XGBoost / LightGBM behaviour in production

Both are unimportable in the committed macOS venv (missing `libomp`). The Dockerfile installs
`libgomp1`, so on Render they probably *do* load — meaning production runs 5 models and local runs 3,
with **no way to tell from the output** which happened. I did not install `libomp`, per the
instruction not to change the environment to make things pass.

**Resolve by:** running the container locally, or adding the per-model status reporting from
Phase 1a.
**Changes if different:** if they load in production, the "winner" and its R² differ between
environments — which makes the reported headline metric environment-dependent.

### A-6 — Real-world sessionStorage failure threshold

`store.ts` persists the full CSV; the demo is 1.13 MB against a ~5 MB quota. I did not run a browser
to find the actual break point for a real upload, and `setup/page.tsx:119-132` already contains a
recovery path suggesting the failure has been observed.

**Resolve by:** a Playwright test uploading progressively larger files.

### A-7 — Groq and W&B paths were not exercised

Neither external service was called (no paid or third-party calls were made). `answer_with_groq`,
the W&B logging path, and the `adj_r2`/`n_train`/`n_test` mismatch were audited **statically**. The
mismatch is certain from reading `schemas.py:97-106` against `wandb_tracking.py:107-112`; the
runtime behaviour of the Groq call is not.

---

## B. Product and scoping decisions that are the author's to make

### B-1 — Does generic CSV upload stay?

My recommendation is *demote, do not delete* (`TARGET_ARCHITECTURE.md` Amendment 1). Counter-argument
worth weighing: it is the only part of the codebase that demonstrates handling arbitrary, hostile
input, which some reviewers value more than a polished case study. **This is a judgement about
audience, not a technical question**, and I do not have enough information about which roles are
being targeted.

### B-2 — Recruiter target: ML engineer, data scientist, or research engineer?

The three want different things. An ML engineer wants leakage-safe pipelines, CI and deployment. A
data scientist wants the identification argument and the refutation tests. A research engineer wants
the modular package and reproducible artifacts. The plan currently serves all three, which risks
serving none exceptionally. **If forced to pick one, pick data scientist** — the causal argument is
the genuinely differentiated asset — but this should be a deliberate choice.

### B-3 — Is the Datathon paper's authority binding?

The Datathon README treats the paper as "the authoritative source of truth" and documents deviations
as *approximations*. But this audit found the paper's own reproduction contains a hard-coded
trade-off narrative contradicting its computed output (D1/D2). **Does the rebuild reproduce the
paper, or correct it?** I recommend correcting it and documenting the correction — a case study that
finds and fixes an error in its own source is stronger than one that reproduces it faithfully. That
is a decision about how to represent prior team work, and it involves co-authors.

### B-4 — Attribution of team work

The Datathon was a four-person team effort (`README.md:6`). LeverGuide appears to be solo. The
rebuilt case study derives its scientific content from the team analysis. How to attribute this is
an ethical and social question outside the scope of a technical audit, but it needs an answer before
anything is published publicly.

---

## C. Scientific questions the data may not be able to answer

### C-1 — Is the cooling-time reverse-causation story identifiable at all?

The mechanism claimed is: operators extend cooling **in response to** observed high mould
temperature. If `mold_temperature_c` at time *t* causes `cooling_time_s` at time *t*, and both are
recorded in the same 30-minute interval, then within-interval simultaneity is not resolvable from
cross-sectional adjustment. The adjustment is valid only if mould temperature is *temporally prior*
within the interval — which the data cannot show at 30-minute granularity.

**Hypothesis, not a finding:** a lagged specification (`mold_temperature_c` at *t−1* → `cooling` at
*t*) would test this. If the effect survives lagging, the story is much stronger; if it does not,
the headline finding rests on an unverifiable timing assumption. **I recommend running this test in
Phase 3.** It is the single most informative additional analysis available, and it could
substantially weaken the project's central claim — which is exactly why it should be run.

### C-2 — Are 12 machines enough for machine-clustered inference?

I recommend clustered SEs by `machine_id` (D-11) and `GroupKFold` by machine (M-4). But 12 clusters
is below the ~30–50 rule of thumb for cluster-robust asymptotics; CRVE with few clusters is
downward-biased. Options: wild cluster bootstrap, CR2/CR3 small-sample corrections, or clustering at
the plant level instead (only **4** clusters — worse). The grouped-CV σ of 0.059 across 5 folds
already shows the instability.

**Unresolved.** My recommendation stands (i.i.d. SEs are certainly too narrow), but the correct
small-sample correction is not settled by this audit.

### C-3 — Should mediators be in the simulation model?

The Datathon deliberately **includes** mediators in the simulation GBR (`src/causal_helpers.py:131`)
so indirect paths are captured, and **excludes** them from the effect-estimation adjustment sets.
That asymmetry is correct in principle. But it means the simulation model's predictions condition on
mediator values that would themselves change under intervention — which is why the delta-propagation
chains exist for two levers. **The three mediators are only chained for two of the five actions.**
For cooling and mould-temperature interventions, the mediators are held at observed values with no
justification given for why they would not move.

**Unresolved and inherited by any port.** Resolving it needs either chains for every mediated path,
or an explicit argument that cooling and mould temperature do not act through moisture/drift/wear.

### C-4 — Does the synthetic data contain a genuine collider?

The audit requirement asks about colliders. Neither repository labels any variable as one.
`defect_type` and `pass_fail_flag` are the structural candidates (caused by both process settings
and the latent defect intensity that drives scrap) and both are correctly excluded — but as
"outcomes", not as "colliders", and no reasoning is recorded.

**Unresolved:** whether the generator actually created a collider structure, or whether the topic
simply does not arise in this dataset. If the latter, the product should teach colliders as a
counter-example rather than claim to demonstrate one (`TARGET_ARCHITECTURE.md` Amendment 2).

### C-5 — Is the 33 % attenuation on cooling stable, or a sample artifact?

D-1 is measured on the 2,000-row sample LeverGuide analyses, with one seed. The gap (−1.171 vs
−1.743 p.p./SD) is large and the mechanism is understood, so I am confident in the direction and
rough size. I did **not** repeat it across seeds or sample sizes.

**Resolve by:** repeating over 20 seeds and on the full 5,000 rows. Cheap; should be done before the
number is quoted publicly.

---

## D. Corrections to my own working hypotheses

Recorded because the reasoning matters more than the conclusions.

| Hypothesis before measuring | What measurement showed |
|---|---|
| LeverGuide's omission of `mold_temperature_c` **flips** the cooling sign, reproducing the exact error the paper warns about | **Refuted.** The sign survives (−1.17 p.p./SD, still protective). The defect is a 33 % magnitude attenuation, not a wrong direction. |
| Mediator over-adjustment is the dominant source of bias | **Refuted for cooling.** Removing all three mediators moves β by <1 %. The dominant term is the *omitted* `mold_temperature_c`. The mediator defect is real and structural but small here. |
| Fitting the scaler before the split materially inflates the reported score | **Refuted.** Measured difference 0.0000 on this dataset. Real defect, no measured harm. |
| The intervention engine would generate out-of-support values on the demo | **Not observed this run.** All 7 suggestions fell inside observed min–max. The support bug is latent (it activates for negative-valued levers) and the *real* infeasibility is coupling violation, not range violation. |
| `cycle_time_s` labelled `mediator` causes active leakage | **Refuted for the current config.** Mediators are excluded from `pred_features`, so it is currently safe — but only incidentally, and one dropdown change away from leaking. |

Three of my five prior hypotheses were wrong in the specifics. They are listed so the confirmed
findings can be read as measurements rather than as confirmations of what I expected to find.
