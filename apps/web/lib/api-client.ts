import type { AnalysisBundle, AnalysisRequest } from "./types";

// Empty string = same-origin (works when FastAPI serves both frontend + API)
// Set NEXT_PUBLIC_API_URL only if deploying frontend separately
const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "";

class ApiError extends Error {
  constructor(public status: number, message: string, public detail?: unknown) {
    super(message);
    this.name = "ApiError";
  }
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    let detail: unknown;
    try { detail = await res.json(); } catch { detail = await res.text(); }
    throw new ApiError(
      res.status,
      (detail as { detail?: string })?.detail ?? `HTTP ${res.status}`,
      detail,
    );
  }
  return res.json() as Promise<T>;
}

export async function runAnalysis(req: AnalysisRequest): Promise<AnalysisBundle> {
  return post<AnalysisBundle>("/api/analyze", req);
}

export { ApiError };
