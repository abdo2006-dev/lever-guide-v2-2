import Papa from "papaparse";
import type { ColumnKind, ColumnMeta, ColumnRole, ParsedDataset } from "./types";
import { DEMO_ONTOLOGY } from "./ontology";

// ── Demo role assignments ─────────────────────────────────────────────────────
// Derived from the generated ontology, never hand-maintained. The backend's
// Python ontology is authoritative; `apps/api/scripts/export_ontology.py`
// regenerates the JSON and a backend test fails if the two drift apart.
export const DEMO_TARGET = DEMO_ONTOLOGY.target;
export const DEMO_ROLES: Record<string, ColumnRole> =
  DEMO_ONTOLOGY.column_roles as Record<string, ColumnRole>;

// ── Type inference ────────────────────────────────────────────────────────────
function inferKind(values: unknown[]): ColumnKind {
  const sample = values.filter((v) => v !== null && v !== "" && v !== undefined).slice(0, 100);
  if (sample.length === 0) return "text";
  const numericCount = sample.filter((v) => !isNaN(Number(String(v)))).length;
  if (numericCount / sample.length > 0.8) return "numeric";
  if (sample.some((v) => /^\d{4}-\d{2}-\d{2}/.test(String(v)))) return "datetime";
  const unique = new Set(sample.map(String)).size;
  if (unique <= 30) return "categorical";
  return "text";
}

function buildColumnMeta(
  name: string,
  values: unknown[],
  role: ColumnRole,
): ColumnMeta {
  const kind = inferKind(values);
  const nonNull = values.filter((v) => v !== null && v !== "" && v !== undefined);
  const missing = values.length - nonNull.length;

  const meta: ColumnMeta = {
    name, kind, role,
    unique: new Set(values.map(String)).size,
    missing,
    top_values: [],
  };

  if (kind === "numeric") {
    const nums = nonNull.map((v) => Number(String(v))).filter((n) => isFinite(n));
    if (nums.length > 0) {
      const sorted = [...nums].sort((a, b) => a - b);
      meta.min    = sorted[0];
      meta.max    = sorted[sorted.length - 1];
      meta.mean   = nums.reduce((s, v) => s + v, 0) / nums.length;
      meta.std    = Math.sqrt(nums.reduce((s, v) => s + (v - meta.mean!) ** 2, 0) / nums.length);
      meta.median = sorted[Math.floor(sorted.length / 2)];
      meta.p25    = sorted[Math.floor(sorted.length * 0.25)];
      meta.p75    = sorted[Math.floor(sorted.length * 0.75)];
    }
  } else {
    const counts: Record<string, number> = {};
    for (const v of nonNull) { const k = String(v); counts[k] = (counts[k] ?? 0) + 1; }
    meta.top_values = Object.entries(counts)
      .sort((a, b) => b[1] - a[1]).slice(0, 10)
      .map(([value, count]) => ({ value, count }));
  }
  return meta;
}

/**
 * Default role for a column of an uploaded dataset.
 *
 * Identifiers and free text are recognised structurally, which is a claim about
 * the *format* of a column. Everything else is left `unassigned`: being numeric
 * is not evidence that a column confounds anything, and silently calling it a
 * confounder made the app assert a causal role the user never gave it.
 */
function inferDefaultRole(name: string, kind: ColumnKind): ColumnRole {
  if (kind === "datetime") return "identifier";
  if (kind === "text") return "ignore";
  const l = name.toLowerCase();
  if (l.includes("id") || l === "timestamp" || l.includes("_id")) return "identifier";
  return "unassigned";
}

// ── Parse CSV text → ParsedDataset ────────────────────────────────────────────
function csvTextToDataset(
  csvText: string,
  name: string,
  roleOverrides: Record<string, ColumnRole> = {},
  targetOverride?: string,
): ParsedDataset {
  const result = Papa.parse(csvText.trim(), {
    header: true,
    skipEmptyLines: true,
    dynamicTyping: false,
  });

  if (result.errors.length > 0) {
    const fatal = result.errors.filter((e) => e.type === "Delimiter" || e.type === "Quotes");
    if (fatal.length > 0) throw new Error(`CSV parse error: ${fatal[0].message}`);
  }

  const rows = result.data as Record<string, unknown>[];
  if (rows.length === 0) throw new Error("CSV file has no data rows.");

  const colNames = Object.keys(rows[0]).filter((k) => k.trim() !== "");
  if (colNames.length === 0) throw new Error("CSV file has no recognisable columns.");

  const columns: ColumnMeta[] = colNames.map((name) => {
    const values = rows.map((r) => r[name]);
    const kind = inferKind(values);
    const role: ColumnRole = roleOverrides[name] ?? inferDefaultRole(name, kind);
    return buildColumnMeta(name, values, role);
  });

  // Mark target
  if (targetOverride) {
    const idx = columns.findIndex((c) => c.name === targetOverride);
    if (idx !== -1) columns[idx].role = "outcome";
  }

  return {
    name,
    csv_content: csvText.trim(),
    columns,
    preview_rows: rows.slice(0, 10),
    row_count: rows.length,
  };
}

// ── Public API ────────────────────────────────────────────────────────────────

export async function parseCsvFile(file: File): Promise<ParsedDataset> {
  const text = await file.text();
  if (!text.trim()) throw new Error("The file is empty.");
  return csvTextToDataset(text, file.name.replace(/\.[^.]+$/, ""));
}

export async function loadDemoDataset(): Promise<ParsedDataset> {
  const res = await fetch("/demo/injection_molding_demo.csv");
  if (!res.ok) throw new Error(`Failed to fetch demo CSV: HTTP ${res.status}`);
  const text = await res.text();
  if (!text.trim()) throw new Error("Demo CSV file is empty.");
  return csvTextToDataset(text, "Injection Molding Demo", DEMO_ROLES, DEMO_TARGET);
}
