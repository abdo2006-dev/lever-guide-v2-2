# Audit import notes

The ten documents in this directory (plus their index, [README.md](README.md)) are the **Phase 0
audit**, imported verbatim. No sentence of any audit document was changed during the import — this
file exists only to resolve path references that were written against the auditor's working
environment and do not resolve inside this repository.

## Path reference mapping

| As written in the audit documents | Meaning inside this repository |
|---|---|
| `datathon-CUB-2026-main/` | The original Datathon analysis repository (`datathon-CUB-2026`). It is a **reference, not a dependency** — it is not vendored here and is not a submodule. |
| `apps/api/app/...`, `apps/web/...` | Repository-relative; resolves as written. |
| `/private/tmp/claude-502/.../scratchpad/verify_*.py` | Read-only verification scripts written during the audit. They lived outside both repositories and were **not** imported; the measurements they produced are quoted in the documents themselves. |
| `~/Downloads/lever-guide-v2-2-main.zip` | A distribution zip the auditor diffed against the working tree. Not part of this repository. |

## Line references are pinned to the pre-Phase-1A tree

Every `file:line` citation in these documents refers to the state of the repository at commit
`2bd854f` (branch `main`), which is the commit Phase 1A branched from. Phase 1A changes many of the
cited files, so line numbers will drift. To read a citation against the code it was written about:

```bash
git show 2bd854f:apps/api/app/routers/analysis.py
```

## Status of the findings

The audit is a record of what was true at `2bd854f`; it is **not** updated as findings are fixed.
Which findings Phase 1A addresses, which it partially addresses, and which it defers are recorded in
[../implementation/PHASE_1A_TRUTH_IN_LABELLING.md](../implementation/PHASE_1A_TRUTH_IN_LABELLING.md).
