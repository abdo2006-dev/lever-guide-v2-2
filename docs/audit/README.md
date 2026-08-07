# Phase 0 Audit — Index

Audit of `datathon-CUB-2026` and `lever-guide-v2-2`, completed 2026-08-01.
**No file in either repository was modified. No commit was made.**

## Documents

| # | Document | Covers |
|---|---|---|
| 1 | [CURRENT_ARCHITECTURE.md](CURRENT_ARCHITECTURE.md) | Requirement §2 — LeverGuide runtime data flow, responsibility split, API surface, deployment, error handling, tests, hygiene |
| 2 | [DATATHON_METHODOLOGY.md](DATATHON_METHODOLOGY.md) | Requirement §1 — reconstructed methodology, roles, identification, estimators, chains, recommendations, plus 8 defects found in the Datathon repo itself |
| 3 | [SCIENTIFIC_DISCREPANCIES.md](SCIENTIFIC_DISCREPANCIES.md) | Requirement §3 — 12-row discrepancy matrix with measured magnitudes |
| 4 | [ML_EVALUATION_AUDIT.md](ML_EVALUATION_AUDIT.md) | Requirement §5 — 13 pipeline findings; 6 split schemes compared empirically; recommended design |
| 5 | [INTERVENTION_AUDIT.md](INTERVENTION_AUDIT.md) | Requirement §6 — classification of the engine, 10 findings, side-by-side vs the original |
| 6 | [TARGET_ARCHITECTURE.md](TARGET_ARCHITECTURE.md) | Requirements §8 + §9 — critique of the proposed direction, preserve/refactor/replace/remove table, target design |
| 7 | [MIGRATION_PLAN.md](MIGRATION_PLAN.md) | Requirement §10 — 6 phases with scope, non-goals, files, risks, acceptance, tests, rollback |
| 8 | [OPEN_QUESTIONS.md](OPEN_QUESTIONS.md) | Unresolved items, product decisions, and corrections to my own hypotheses |
| — | [CAUSAL_CLAIMS_AUDIT.md](CAUSAL_CLAIMS_AUDIT.md) | Requirement §4 (supplement) — 15 claims classified, replacement wording |
| — | [FRONTEND_REPO_QUALITY.md](FRONTEND_REPO_QUALITY.md) | Requirement §7 (supplement) — 15 frontend/repo findings, all tied to comprehension/trust/recruiter evaluation |

Requirements §4 and §7 were given their own documents rather than being folded into the eight named
deliverables; both are cross-referenced throughout.

## Evidence standard

* Every finding cites `file:line`.
* Findings are marked **confirmed** (measured or read) or ***hypothesis***.
* Five findings were reproduced by executing LeverGuide's own modules on the demo CSV.
* Verification scripts (read-only) are in the sibling `scratchpad/` directory:
  `verify_leverguide.py`, `verify_decomp.py`, `verify_intervention.py`.
* Failures found and reported rather than worked around: `xgboost` and `lightgbm` are unimportable in
  the committed venv (`libomp` missing). `libomp` was **not** installed.
* No paid or external service was called. No secret was read or reproduced.
