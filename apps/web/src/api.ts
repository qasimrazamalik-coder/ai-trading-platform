import { fallbackTerminalState } from "./demo.js";
import type { PaperOrderCommand, PaperOrderResponse, TerminalState } from "./types.js";

const API_BASE = document.documentElement.dataset.apiBase ?? "http://127.0.0.1:8000";

export interface LoadResult<T> {
  readonly data: T;
  readonly source: "api" | "offline-demo";
}

export async function loadTerminalState(): Promise<LoadResult<TerminalState>> {
  try {
    const response = await fetch(`${API_BASE}/terminal/demo-state`, {
      headers: { Accept: "application/json" }
    });
    if (!response.ok) throw new Error(`terminal API returned ${response.status}`);
    const data = (await response.json()) as TerminalState;
    return { data, source: "api" };
  } catch {
    return { data: fallbackTerminalState, source: "offline-demo" };
  }
}

export async function loadHealth(): Promise<boolean> {
  try {
    const response = await fetch(`${API_BASE}/health`, { headers: { Accept: "application/json" } });
    if (!response.ok) return false;
    const data = (await response.json()) as { status?: string };
    return data.status === "ok";
  } catch {
    return false;
  }
}

export async function submitPaperOrder(
  command: PaperOrderCommand
): Promise<PaperOrderResponse> {
  const response = await fetch(`${API_BASE}/paper/orders`, {
    method: "POST",
    headers: {
      Accept: "application/json",
      "Content-Type": "application/json"
    },
    body: JSON.stringify(command)
  });
  if (!response.ok) {
    const detail = (await response.json().catch(() => ({}))) as { detail?: unknown };
    const message = typeof detail.detail === "string" ? detail.detail : `HTTP ${response.status}`;
    throw new Error(`paper order rejected: ${message}`);
  }
  return (await response.json()) as PaperOrderResponse;
}
