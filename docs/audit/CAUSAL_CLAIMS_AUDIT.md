# Causal Claims Audit

*Supplement to the eight required deliverables — covers audit requirement §4.*

Every claim in the UI, API, docs, symbol names and generated text that touches causality, classified
as **defensible / defensible with stronger wording / methodologically incomplete / misleading /
incorrect**. Replacement wording is proposed. **No source file was edited.**

---

## Headline: the README is more honest than the product

`README.md` is genuinely careful. It states that effect estimates are "observational adjustment, not
proof of causality" (line 20), that they are "not do-calculus, not causal discovery, and not
guaranteed minimal valid adjustment sets" (line 92), that the default DAG "is not learned from data"
(line 116), that the intervention engine "can propose unrealistic changes when variables are
physically or operationally coupled" (line 131), and it lists nine real limitations (lines 346-357).

The overstatements are almost entirely in the **UI**, the **symbol names**, and the **generated
summary strings** — i.e. exactly the surfaces a recruiter or a user actually reads. That is the gap
to close.

---

## C-1 — Tab label "Effect Estimates" + method box — **defensible**

`analyze/page.tsx:39, 339-347`. The box says "Back-door adjusted OLS", gives the regression formula,
defines β as "effect on {target} per +1 SD increase … with all adjustment variables held constant",
names the homoskedasticity assumption, and ends "This is **observational** — unobserved confounders
may bias estimates." The tab is not called "Causal Effects". This is good practice; keep it.

**One factual error inside it (see C-2).**

## C-2 — "Mediators are excluded (blocking the causal path would absorb the effect)" — **incorrect**

`analyze/page.tsx:344`. True of the *mechanism* (`dag.py:122-123`), false of the *analysis being
displayed*: the demo labels all three genuine mediators as confounder/context, so all three are in
every adjustment set (`SCIENTIFIC_DISCREPANCIES.md` D-3). The sentence tells the user a guarantee
holds when it does not.

> **Replacement:** "Any column you label *mediator* is excluded from the adjustment set, because
> conditioning on it would block the causal path and turn a total effect into a direct effect.
> Labelling mediators correctly is your responsibility — this analysis currently treats
> `resin_moisture_pct`, `calibration_drift_index` and `tool_wear_index` as adjusters, which yields
> direct rather than total effects for any lever acting through them."

## C-3 — Intervention badge `evidence_type: "causal"` → "adjusted evidence" — **misleading**

`intervention.py:15-20` sets `"causal"` on `p < 0.05` alone; `analyze/page.tsx:438` renders it as
"adjusted evidence". The rendered label is decent. The problems are (a) the payload field, which any
API consumer sees as `"causal"`, and (b) that the badge never checks the OLS sign against the
recommended direction — measured: **2 of 7 recommendations, including rank 1, point the opposite way
from their own estimate** (`INTERVENTION_AUDIT.md` §1).

> **Replacement:** rename the field `adjustment_support: "aligned" | "conflicting" | "none"`, and
> render "adjusted estimate agrees with this direction (p=…)" / "**adjusted estimate points the
> other way** — treat as predictive only".

## C-4 — `expected_kpi_change` as a bare bold number — **misleading**

`analyze/page.tsx:451, 477-481`. `−0.4153` / `−9.4 % scrap_rate_pct` in bold green, no interval. The
measured bootstrap interval for that exact figure is **[−0.515, −0.166]**. The word "expected"
implies an expectation over a distribution that is never shown.

> **Replacement:** "Simulated change: **−0.42 p.p.** (bootstrap 95 %: −0.51 to −0.17). Simulation
> only — the model's out-of-fold R² is 0.54, so treat magnitudes as indicative."

## C-5 — "Honest uncertainty — Confidence intervals … always visible" — **misleading**

`HomeClient.tsx:32-35`. True on the effect-estimates tab; false on the interventions tab, which is
the product's headline output and has no interval of any kind.

> **Replacement:** "Uncertainty where we have it — confidence intervals and p-values on every
> adjusted effect estimate; simulated interventions are point estimates and are labelled as such."

## C-6 — Module named `causal.py`, function `run_causal_analysis`, field `causal[]` — **defensible with stronger wording**

`apps/api/app/models/causal.py`, `schemas.py` `AnalysisBundle.causal`. The docstring is careful
(lines 8-9: "This is NOT full structural causal modelling"), but the names are what appear in the
OpenAPI schema, in the W&B artifacts, in the Qdrant `causal_findings` artifact, and in every code
reading. `adjusted_effects.py` / `run_adjusted_effect_estimation` / `adjusted_effects[]` would carry
the same information without the claim.

## C-7 — "Predictive importance … Causal estimates use back-door adjusted OLS" — **methodologically incomplete**

`analysis.py:224-228`, the `methodology_note` shipped in every executive summary and indexed into
the Copilot corpus. It calls the OLS output "causal estimates" without any of the README's hedging,
and the sentence "Intervention magnitudes are counterfactual simulations from the predictive model"
does not say the model is fit and evaluated on the same rows.

> **Replacement:** "Feature importance is gradient-boosted-tree gain — a predictive diagnostic, not
> a causal ranking. Effect estimates are back-door-adjusted OLS under an assumed DAG; they are
> observational and rest on the assumption that the adjustment set is correct and complete.
> Intervention magnitudes are what-if predictions from a model fitted on these same rows and are not
> identified causal effects."

## C-8 — `auto_dag` output presented as a validated graph — **methodologically incomplete**

`dag.py:132-154` emits 60 template edges from role labels. `validate_dag` then checks it for cycles
and returns `dag_validation.valid: true`, which flows into the bundle, the UI warnings block, and
the Qdrant artifact `"Valid DAG: True"`. "Valid" here means *acyclic and well-formed*, not
*scientifically defensible*. A user reading "Valid DAG: True" will not draw that distinction. The
README says the right thing (line 116); nothing in the runtime does.

> **Replacement:** report `dag_source: "assumed_from_roles" | "user_supplied"` alongside
> `structurally_valid: true`, and render "Structurally valid (acyclic). This graph was generated
> from your role labels — it is an assumption, not a discovered causal structure."

**Nothing in either repository claims to have discovered a causal graph from data.** The requirement
to check for "automatically generated graphs presented as discovered causal structures" is answered:
the generation is automatic, the presentation is *ambiguous*, but no discovery claim is made.

## C-9 — "Top Levers to Pull" — **misleading**

`analyze/page.tsx:528`. Imperative. Green chips. No caveat in the component. It sits two cards above
the caveats card, and on the demo it would name `shot_size_g` — the physically impossible one.

> **Replacement:** "Highest-ranked candidate levers (simulation-ranked; validate before acting)".

## C-10 — Executive bullets "is estimated to reduce {target} by ~X %" — **misleading**

`analysis.py:199-206`. Generated per intervention with `evidence_type` and `evidence_strength`
interpolated — so the demo produces sentences like "Decreasing shot_size_g is estimated to reduce
scrap_rate_pct by ~9.4 % (mixed evidence, weak strength)". The hedge is real but is trailing
parenthetical text after a confident main clause.

> **Replacement:** "Simulation suggests decreasing `shot_size_g` could reduce `scrap_rate_pct` by
> ~9.4 %, but the adjusted estimate for this variable points the other way — do not act on this
> without a controlled test."

## C-11 — "Don't just predict. Decide what to change." + "predictive and causal evidence side by side" — **defensible with stronger wording**

`HomeClient.tsx` hero and line 92. The prediction-vs-intervention framing is the correct and
valuable message — it is the Datathon's own thesis. The weakness is "causal evidence", which as
shipped is adjusted-OLS-under-an-assumed-DAG. Under the proposed *Causal Process Studio*
repositioning this line becomes an asset rather than a liability.

> **Replacement:** "…with predictive signal and DAG-adjusted effect estimates shown side by side —
> and the difference between them made explicit."

## C-12 — "Upload CSV files up to 50 MB" — **incorrect** (not causal, but a factual claim)

`HomeClient.tsx:38`, `setup/page.tsx:85, 229`, `README.md:14`. The full CSV is (a) persisted into
`sessionStorage` (`store.ts:77`), quota ≈ 5 MB and JSON-escaped, and (b) POSTed as a JSON string
body to a free-tier service with ~512 MB RAM. The 1.13 MB demo already occupies a fifth of the
sessionStorage budget. A 50 MB upload cannot work.

> **Replacement:** "CSV up to ~5 MB (browser storage limit). Datasets over 2,000 rows are randomly
> sub-sampled to 2,000 for analysis."

## C-13 — "All models on the same 80/20 train/test split" — **incorrect**

`analyze/page.tsx:139`. `pipeline.py:55` resolves to `test_size=0.1` for n = 2,000. Verified:
1,800/200. The README is correct; only the UI is wrong.

## C-14 — Copilot system prompt — **defensible**

`rag.py:400-405`: "Answer only from the provided analysis artifacts. Do not invent causal claims,
data values, model results, or recommendations. … mention caveats when causal or intervention claims
are involved." This is a well-written grounding prompt. Note however that it grounds the model in
artifacts that *themselves* contain C-3, C-7 and C-10, so the Copilot will faithfully repeat
overstatements. Fixing the artifact text fixes the Copilot.

## C-15 — Datathon: "Assessment: highly favourable trade-off" — **incorrect**

`notebooks/03` cell 19 prints a computed ratio of **0.2×** (paper: ~10×) and then asserts the
trade-off "pays back the cycle-time cost many times over". The narrative is a hard-coded string
independent of the number above it. Since LeverGuide's repositioning is meant to be built *on* the
Datathon, this must be resolved in the source before it is inherited. See
`DATATHON_METHODOLOGY.md` D1/D2.

---

## Tally

| Classification | Count | IDs |
|---|---|---|
| Defensible | 2 | C-1, C-14 |
| Defensible with stronger wording | 2 | C-6, C-11 |
| Methodologically incomplete | 2 | C-7, C-8 |
| Misleading | 5 | C-3, C-4, C-5, C-9, C-10 |
| Incorrect | 4 | C-2, C-12, C-13, C-15 |

**Pattern:** the product is honest in its long-form prose and overconfident in its short-form UI.
Every misleading item is a badge, a bold number, a heading, or a generated sentence — the elements
with the highest read-rate and the least room for a caveat. The fix is not more disclaimers at the
bottom; it is **putting the uncertainty inside the number** (C-4), and **refusing to emit a
recommendation when the evidence conflicts with itself** (C-3).
