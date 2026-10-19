import { describeApiError } from "./utils.js";

// "/api" works with the Vite dev proxy (see vite.config.js). For a production build set VITE_API_URL in .env.
const BASE = import.meta.env.VITE_API_URL ?? "/api";

/** Send the text to the FastAPI backend (POST /predict). Returns { data, clientMs }. */
export async function analyzeText(text, explain = true) {
  const started = performance.now();
  let response;
  try {
    response = await fetch(`${BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, explain, details: true }),
    });
  } catch {
    throw new Error(
      "Cannot reach the backend. Start it in a terminal with: uvicorn backend.main:app --host 127.0.0.1 --port 8000"
    );
  }
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(describeApiError(response.status, body));
  }
  const data = await response.json();
  return { data, clientMs: performance.now() - started };
}
