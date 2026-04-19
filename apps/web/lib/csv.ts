import Papa from "papaparse";
import type { ColumnKind, ColumnMeta, ColumnRole, ParsedDataset } from "./types";

// Demo role map — matches DEMO_ROLES in apps/api/app/routers/analysis.py
const DEMO_ROLES: Record<string, ColumnRole> = {
  scrap_rate_pct:            "outcome",
  barrel_temperature_c:      "controllable",
  mold_temperature_c:        "controllable",
  injection_pressure_bar:    "controllable",
  hold_pressure_bar:         "controllable",
  screw_speed_rpm:           "controllable",
  cooling_time_s:            "controllable",
  clamp_force_kn:            "controllable",
  shot_size_g:               "controllable",
  ambient_temperature_c:     "confounder",
  ambient_humidity_pct:      "confounder",
  resin_moisture_pct:        "confounder",
  resin_batch_quality_index: "confounder",
  dryer_dewpoint_c:          "confounder",
  cavity_count:              "context",
  product_variant:           "context",
  operator_experience_level: "context",
  operator_shift:            "context",
  tool_wear_index:           "context",
  calibration_drift_index:   "context",
  maintenance_days_since_last: "context",
  cycle_time_s:              "mediator",
  part_weight_g:             "mediator",
  timestamp:                 "identifier",
  plant_id:                  "identifier",
  machine_id:                "identifier",
  mold_id:                   "identifier",
  resin_lot_id:              "identifier",
  defect_type:               "ignore",
  scrap_count:               "ignore",
  parts_produced:            "ignore",
  energy_kwh_interval:       "ignore",
  pass_fail_flag:            "ignore",
};

const DEMO_TARGET = "scrap_rate_pct";

function inferKind(values: unknown[]): ColumnKind {
  const sample = values.filter((v) => v !== null && v !== "").slice(0, 50);
  const numericCount = sample.filter((v) => !isNaN(Number(v))).length;
  if (numericCount / Math.max(sample.length, 1) > 0.8) return "numeric";
  const unique = new Set(sample.map(String)).size;
  if (unique <= 30) return "categorical";
  // Naive date check
  if (sample.some((v) => /^\d{4}-\d{2}-\d{2}/.test(String(v)))) return "datetime";
  return "text";
}

function buildColumnMeta(
  name: string,
  values: unknown[],
  role: ColumnRole
): ColumnMeta {
  const kind = inferKind(values);
  const nonNull = values.filter((v) => v !== null && v !== "" && v !== undefined);
  const missing = values.length - nonNull.length;

  const meta: ColumnMeta = {
    name,
    kind,
    role,
    unique: new Set(values.map(String)).size,
    missing,
    top_values: [],
  };

  if (kind === "numeric") {
    const nums = nonNull.map((v) => Number(v)).filter((n) => isFinite(n));
    if (nums.length > 0) {
      const sorted = [...nums].sort((a, b) => a - b);
      meta.min = sorted[0];
      meta.max = sorted[sorted.length - 1];
      meta.mean = nums.reduce((s, v) => s + v, 0) / nums.length;
      const variance = nums.reduce((s, v) => s + (v - meta.mean!) ** 2, 0) / nums.length;
      meta.std = Math.sqrt(variance);
      meta.median = sorted[Math.floor(sorted.length / 2)];
      meta.p25 = sorted[Math.floor(sorted.length * 0.25)];
      meta.p75 = sorted[Math.floor(sorted.length * 0.75)];
    }
  } else {
    const counts: Record<string, number> = {};
    for (const v of nonNull) {
      const k = String(v);
      counts[k] = (counts[k] ?? 0) + 1;
    }
    meta.top_values = Object.entries(counts)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 10)
      .map(([value, count]) => ({ value, count }));
  }

  return meta;
}

function inferDefaultRole(name: string, kind: ColumnKind): ColumnRole {
  if (kind === "datetime" || kind === "text") return "ignore";
  const lower = name.toLowerCase();
  if (lower.includes("id") || lower.includes("timestamp")) return "identifier";
  return "confounder"; // conservative default
}

export async function parseCsvFile(file: File): Promise<ParsedDataset> {
  return new Promise((resolve, reject) => {
    Papa.parse(file, {
      header: true,
      skipEmptyLines: true,
      dynamicTyping: false, // keep as strings for now, we'll infer types
      complete: (result) => {
        try {
          const rows = result.data as Record<string, unknown>[];
          if (rows.length === 0) {
            reject(new Error("CSV file is empty."));
            return;
          }
          const colNames = Object.keys(rows[0]);
          const columns: ColumnMeta[] = colNames.map((name) => {
            const values = rows.map((r) => r[name]);
            const kind = inferKind(values);
            const role = inferDefaultRole(name, kind);
            return buildColumnMeta(name, values, role);
          });

          const csv_content = Papa.unparse(rows);

          resolve({
            name: file.name.replace(/\.[^.]+$/, ""),
            csv_content,
            columns,
            preview_rows: rows.slice(0, 10),
            row_count: rows.length,
          });
        } catch (err) {
          reject(err);
        }
      },
      error: (err) => reject(new Error(err.message)),
    });
  });
}

export async function loadDemoDataset(): Promise<ParsedDataset> {
  const res = await fetch("/demo/injection_molding_demo.csv");
  if (!res.ok) throw new Error("Failed to load demo dataset");
  const text = await res.text();

  const result = Papa.parse(text, { header: true, skipEmptyLines: true, dynamicTyping: false });
  const rows = result.data as Record<string, unknown>[];
  const colNames = Object.keys(rows[0]);

  const columns: ColumnMeta[] = colNames.map((name) => {
    const values = rows.map((r) => r[name]);
    const role: ColumnRole = DEMO_ROLES[name] ?? inferDefaultRole(name, inferKind(values));
    return buildColumnMeta(name, values, role);
  });

  // Mark target
  const outcomeIdx = columns.findIndex((c) => c.name === DEMO_TARGET);
  if (outcomeIdx !== -1) columns[outcomeIdx].role = "outcome";

  return {
    name: "Injection Molding Demo",
    csv_content: text,
    columns,
    preview_rows: rows.slice(0, 10),
    row_count: rows.length,
  };
}

export { DEMO_TARGET, DEMO_ROLES };
