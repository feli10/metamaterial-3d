/**
 * client.ts — THE DATA LAYER (real backend implementation).
 *
 * This is the "worker" that fulfills the contract in types.ts. The UI imports `api` from here
 * and calls `api.generate(...)`, `api.evaluate(...)`, `api.listModels()`. It never knows how
 * the data is produced — it used to be a mock; now every call is a real `fetch()` against the
 * FastAPI backend (web/backend/main.py). The signatures and types are unchanged, so no other
 * file in the UI had to change.
 *
 * The backend base URL is configurable via the VITE_API_BASE env var (set at build time for
 * production, e.g. the deployed sim-hub URL). It defaults to the local dev server.
 */

import {
  type EvaluateProgress,
  type EvaluateRequest,
  type EvaluateResponse,
  type GenerateProgress,
  type GenerateRequest,
  type GenerateResponse,
  type MetamaterialApi,
  type ModelInfo,
} from "./types";

/**
 * Where the backend lives. Vite replaces `import.meta.env.VITE_API_BASE` at build time.
 * - Local dev: nothing set -> falls back to the uvicorn server on localhost:8000.
 * - Production: set VITE_API_BASE (e.g. in web/frontend/.env.production) to the sim-hub URL.
 */
const API_BASE = (import.meta.env.VITE_API_BASE ?? "http://localhost:8000").replace(/\/$/, "");

/**
 * POST a JSON body to `path` and return the parsed JSON response. Throws a readable Error if
 * the request fails or the server returns a non-2xx status (FastAPI puts the reason in
 * `detail`), so the caller's try/catch shows a useful message.
 */
async function postJson<T>(path: string, body: unknown): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch {
    // Network-level failure (server down, CORS blocked, wrong URL).
    throw new Error(`Cannot reach the backend at ${API_BASE}. Is it running?`);
  }
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const data = await res.json();
      if (data?.detail) detail = typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail);
    } catch {
      /* body wasn't JSON — keep the status text */
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export const api: MetamaterialApi = {
  async listModels(): Promise<ModelInfo[]> {
    let res: Response;
    try {
      res = await fetch(`${API_BASE}/models`);
    } catch {
      throw new Error(`Cannot reach the backend at ${API_BASE}. Is it running?`);
    }
    if (!res.ok) throw new Error(`Failed to load models (${res.status})`);
    return res.json() as Promise<ModelInfo[]>;
  },

  async generate(
    req: GenerateRequest,
    // The backend runs the ~100-iter optimization server-side and returns only the final
    // result (a plain POST can't stream per-iteration progress). onProgress is kept in the
    // contract for a future streaming endpoint; for now it simply isn't called, and the UI
    // shows an indeterminate "Generating…" state instead of a live iteration count.
    _onProgress?: (p: GenerateProgress) => void,
  ): Promise<GenerateResponse> {
    return postJson<GenerateResponse>("/generate", req);
  },

  async evaluate(
    req: EvaluateRequest,
    _onProgress?: (p: EvaluateProgress) => void,
  ): Promise<EvaluateResponse> {
    return postJson<EvaluateResponse>("/evaluate", req);
  },
};
