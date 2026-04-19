import type { AnalysisBundle, AnalysisRequest } from "./types";

const API_BASE =
  process.env.NEXT_PUBLIC_API_URL ??
  (typeof window !== "undefined" ? "" : "http://localhost:8000");

class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
    public detail?: unknown
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const url = `${API_BASE}${path}`;
  const res = await fetch(url, {
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

export async function checkHealth(): Promise<{ status: string }> {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new ApiError(res.status, "API unavailable");
  return res.json();
}

export { ApiError };
