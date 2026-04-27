import type { AnalysisBundle, AnalysisRequest, CopilotAnswerResponse, CopilotAskRequest } from "./types";

// Empty string = same-origin (works when FastAPI serves both frontend + API)
// Set NEXT_PUBLIC_API_URL only if deploying frontend separately
const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "";

class ApiError extends Error {
  constructor(public status: number, message: string, public detail?: unknown) {
    super(message);
    this.name = "ApiError";
  }
}

async function post<T>(path: string, body: unknown, signal?: AbortSignal): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok) {
    let detail: unknown;
    try { detail = await res.json(); } catch { detail = await res.text(); }
    const apiDetail = (detail as { detail?: unknown })?.detail;
    const message =
      typeof apiDetail === "string"
        ? apiDetail
        : typeof apiDetail === "object" && apiDetail !== null && "message" in apiDetail
          ? String((apiDetail as { message?: unknown }).message)
          : `HTTP ${res.status}`;
    throw new ApiError(
      res.status,
      message,
      detail,
    );
  }
  return res.json() as Promise<T>;
}

export async function runAnalysis(req: AnalysisRequest, signal?: AbortSignal): Promise<AnalysisBundle> {
  return post<AnalysisBundle>("/api/analyze", req, signal);
}

export async function askCopilot(req: CopilotAskRequest, signal?: AbortSignal): Promise<CopilotAnswerResponse> {
  return post<CopilotAnswerResponse>("/api/copilot/ask", req, signal);
}

export { ApiError };
