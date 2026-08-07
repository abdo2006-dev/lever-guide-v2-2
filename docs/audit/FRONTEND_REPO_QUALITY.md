# Frontend & Repository Quality Audit

*Supplement to the eight required deliverables — covers audit requirement §7.*

Every UI recommendation below is tied to **comprehension**, **trust**, or **recruiter evaluation**.
No recommendation is aesthetic.

---

## F-1 — 705-line page component holding all six tabs — *recruiter evaluation*

`apps/web/app/analyze/page.tsx` defines `AnalyzePage`, `OverviewTab`, `PredictiveTab`, `CausalTab`,
`InterventionsTab`, `IvCard`, `ExecutiveTab`, `CopilotTab`, plus `Card`, `KpiCard`, `MetCard`,
`Row`, `Badge`, `Empty`. `setup/page.tsx` is another 483 lines with `Section`, `Check`, `Chip`
inline.

A reviewer opening the flagship file of a portfolio project sees one 705-line module. Split into
`components/analyze/<Tab>.tsx` + `components/ui/`. **This is not a rewrite** — the components
already exist as files (F-2); the work is reconciling them.

## F-2 — Dead duplicate components — *recruiter evaluation, trust*

`apps/web/components/analyze/` contains `CausalTab.tsx` (127), `PredictiveTab.tsx` (145),
`InterventionsTab.tsx` (150), `ExecutiveTab.tsx` (120), `EdaTab.tsx` (**1 line, empty stub**).
Confirmed by grep: **nothing imports any of them.** 543 lines of unreachable code duplicating
logic that lives inline in `analyze/page.tsx`.

Two versions of the same component that can silently diverge is the specific failure mode a
reviewer is looking for. Delete or promote — do not keep both.

## F-3 — No EDA in the UI at all — *comprehension*

`EdaTab.tsx` is empty and no tab references it. Meanwhile the backend computes up to 100
`correlations` (`analysis.py:120-137`) and `distributions` for 30 columns
(`analysis.py:140-173`), Pydantic-validates them, and ships them in every response — where they are
**never rendered**. Wasted payload and a missing chapter.

This matters disproportionately for the proposed repositioning: "correlation → intervention" is the
product's thesis, and the correlation half is invisible. The raw ρ(cooling, scrap) = **+0.278** — the
number the entire case study exists to overturn — is computed, serialised, and thrown away.

## F-4 — `reactflow` declared, no DAG editor exists — *trust*

`package.json:38` declares `reactflow@^11.11.4`. Zero imports. `store.dagEdges` is initialised `[]`,
`setDagEdges` is never called, and `setup/page.tsx:149` therefore always posts `dag_edges: []`, so
`auto_dag` runs on every request (`CURRENT_ARCHITECTURE.md` §5).

The README is honest about this ("There is no visual DAG editor yet", line 346). But shipping the
dependency creates the impression of a feature in `package.json`, and the DAG is the single most
important missing UI for the intended product.

## F-5 — 23 of 31 runtime dependencies unused — *recruiter evaluation*

All 16 `@radix-ui/*`, `reactflow`, `zod`, `@tanstack/react-query`, `class-variance-authority`,
`tailwind-merge`, `clsx`. In use: `next`, `react`, `react-dom`, `recharts`, `lucide-react`, `sonner`,
`zustand`, `papaparse`, `next-themes`, `tailwindcss-animate`.

This inflates install time and lockfile size, and reads as scaffolding left behind. Removing 23
dependencies is a 10-minute change with a visible payoff.

## F-6 — sessionStorage holds the entire CSV — *trust*

`store.ts:72-78` persists `dataset` including `csv_content`. The comment says "sessionStorage
handles ~5MB fine". The demo alone is 1.13 MB; JSON-escaped inside a larger object it is a
meaningful fraction of the quota, and a real upload near the advertised 50 MB throws
`QuotaExceededError`. `setup/page.tsx:119-132` already contains a recovery path for "if csv_content
was somehow lost from sessionStorage" — evidence that this failure has been hit in practice, handled
for the demo (re-fetch) but **unrecoverable for a user upload**.

Compounding: state lives only in sessionStorage, so a results URL cannot be shared. For a
recruiter-facing case study, a shareable permalink to a specific analysis is a high-value, low-cost
feature.

## F-7 — Misleading upload-size claim — *trust*

"Max 50 MB" in three places (`HomeClient.tsx:38`, `setup/page.tsx:85`, `setup/page.tsx:229`) and the
README. Impossible for the reasons in F-6 plus the JSON POST body against a free-tier 512 MB
service. See `CAUSAL_CLAIMS_AUDIT.md` C-12 for replacement wording.

## F-8 — Uploading a fresh CSV always fails first time — *comprehension*

`csv.ts:92-98` `inferDefaultRole()` returns `"confounder"` for every non-ID numeric column. The API
returns 422 "No columns assigned 'controllable' role" (`analysis.py:272-276`) whenever nothing is
controllable. A user who uploads their own file and clicks through **cannot** succeed without
manually re-labelling. The wizard's step-3 subtitle mentions it in passing; the analyse button is
simply disabled with no explanation of *which* columns to change.

Given the recommended product direction (curated case study, generic upload demoted), this is
another argument for removing generic upload from the core path.

## F-9 — Charts lack accessibility and interpretive scaffolding — *comprehension*

* Recharts `ResponsiveContainer` output has no `role`, `aria-label`, or text alternative; the
  content is unavailable to a screen reader and to anyone reading a PDF export of the page.
* The causal forest-equivalent (`analyze/page.tsx:350-365`) plots β as bars **without the
  confidence intervals** that are present in the data (`conf_int_lo/hi`) and shown in the adjacent
  table. A forest plot without whiskers is the chart that most invites over-reading a point
  estimate — and the Datathon's own `figures/forest_plot.png` *does* draw them.
* Colour alone encodes significance (grey vs red/green, line 360) — fails for colour-vision
  deficiency and prints identically in greyscale.
* Feature names are truncated at 20-22 chars with "…" (lines 209, 332) and no tooltip restores the
  full name.

**Highest-value single fix: add the CI whiskers.** It converts the chart from a ranking into an
uncertainty display and directly supports the trust argument.

## F-10 — Loading and error states — *mostly good*

Genuinely well done: rotating status messages with elapsed timer and progress bar
(`setup/page.tsx:392-414`), 90 s and 45 s abort timeouts, typed `ApiError` with `INVALID_DAG`
special-casing (lines 155-173), `Empty` fallbacks on every tab, null-safe formatters
(`analyze/page.tsx:19-23`).

Two gaps: (a) the progress bar is **fake** — driven by a 4-second interval, not by server progress,
so it advances identically whether the server is working or hung; (b) `analyze/page.tsx:53-55`
redirects to `/setup` when `analysis` is null, so a refreshed or shared `/analyze` URL silently
bounces the visitor with no explanation.

## F-11 — Missing explanatory content — *comprehension, recruiter evaluation*

The app has good *metric* explainers ("How to read these metrics", `analyze/page.tsx:243-250`;
"How to read" for coefficients, lines 289-293) and a solid method box on the causal tab. What is
absent is the **argument**:

* no statement of the business problem or the 3.2 % threshold;
* no before/after: raw correlation → adjusted estimate for the same variable;
* no DAG, so the reader never sees why one variable is adjusted for and another is not;
* no anti-recommendations (the Datathon's "do NOT shorten cooling" is its most memorable result);
* no reproducibility or provenance panel.

A recruiter spends 3–5 minutes. Currently those minutes are spent on model-comparison bars. They
should be spent on the sign reversal.

## F-12 — Deployment inconsistencies — *trust*

* Two contradictory `render.yaml` files: root = single combined free service on
  `lever-guide.onrender.com`; `apps/api/render.yaml` = API-only, **`plan: starter` ($7/mo)**,
  `region: frankfurt`, CORS pointing at `lever-guide.vercel.app`. Neither is marked authoritative.
* Two `Dockerfile`s, **neither used** — both render.yaml files use `runtime: python`.
* README "Planned/Future Work" lists "Stable public deployment link" — i.e. there is no live URL.
  For a recruiter-facing artefact this is the single highest-value gap.
* `PYTHON_VERSION 3.12.0` in Render vs Python 3.10.4 in the committed venv.
* `wandb` unpinned in `requirements.txt`.

## F-13 — README reads as an implementation inventory — *recruiter evaluation*

365 lines. Structure: "What Works Today" (feature table) → Architecture (file tree) → Backend
Behavior (11 numbered pipeline steps) → per-model hyperparameter table → API JSON samples → RAG
design → design trade-offs → **"Repository Polish Notes"** → Limitations → Planned Work.

The prose quality and intellectual honesty are high — the trade-off section is genuinely good
writing. But the document answers "what did you build?" and never answers "what did you find?".
There is no scientific result anywhere in it. A reader learns the LightGBM `num_leaves` before
learning what the app is for.

Worse, lines 316-338 ("Repository Polish Notes") instruct the reader to run `wc -l` to verify the
files are not minified one-line blobs. That is an artefact of a past packaging incident, addressed
to the author rather than the audience, and it is the least confidence-inspiring passage in either
repository.

**Recommended README spine:** the question → the counter-intuitive finding (ρ = +0.28 → β = −1.74,
with the figure) → how the DAG produces that reversal → what was recommended and what was explicitly
*not* → how it was validated (grouped CV, refutation) → how to reproduce → architecture → limits.

## F-14 — Repository hygiene — *recruiter evaluation*

* **Stray brace-expansion directories** on disk from a failed `mkdir -p`: `{apps/`,
  `{apps/web,apps/`, `{apps/web,apps/api}`, `apps/web/{app/{setup,analyze,api/health,api/analyze},components/{ui,setup,analyze},lib,hooks,public}`,
  `apps/api/{app/{routers,models,utils},tests}`. Untracked by git, but present in the working tree
  and visible to anyone who clones and lists.
* `apps/web/out/` (built export) and `apps/api/.qdrant/` (SQLite) on disk — correctly gitignored.
* `apps/api/wandb/offline-run-20260501_000458-jbyblv3d/logs/debug.log` — a stray run artifact
  present in the distributed zip.
* Two `.venv` directories.
* **Secrets: clean.** `apps/api/.env` exists locally with a real `GROQ_API_KEY` and is correctly
  gitignored (`git check-ignore` confirms, `git ls-files` returns nothing). Its contents were not
  read or reproduced in this audit. No secret is in the repository.
* Commit messages are terse but descriptive ("fixed load issue in main.py", "integrated W&B
  tracking"). No CI configuration in the repository.

## F-15 — Schema duplication — *trust*

`DEMO_ROLES` exists in two files that must agree by hand: `apps/api/app/routers/analysis.py:37-77`
and `apps/web/lib/csv.ts:6-40`. They currently agree. `schemas.py` claims to be "the single source of
truth — the frontend TypeScript types are generated from / kept in sync with these" (lines 2-4), but
`lib/types.ts` (248 lines) is hand-written; nothing generates it. Column-kind inference is also
implemented twice with different heuristics (`csv.ts:43-52` vs `preprocess.py:18-26`) and can
disagree on a borderline column.

---

## Priority for a recruiter-facing rebuild

| Rank | Item | Effort | Why |
|---|---|---|---|
| 1 | F-11 explanatory narrative + F-3 EDA + F-13 README spine | M | This is what a reviewer actually reads. |
| 2 | F-4 real DAG editor | M–L | The missing centrepiece of the intended product. |
| 3 | F-9 CI whiskers + accessible charts | S | Converts charts from rankings into evidence. |
| 4 | F-12 one live deployment URL | S | An unreachable case study cannot be evaluated. |
| 5 | F-2 delete dead components, F-5 drop 23 deps, F-14 clean stray dirs | S | Cheap; removes the "unfinished" signal. |
| 6 | F-6 shareable permalinks | M | Lets the work be linked in an application. |
| 7 | F-1 split the 705-line page | M | Code-quality signal. |
| 8 | F-7/F-8 fix or remove generic upload | S | Currently an advertised path that fails. |
