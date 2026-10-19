// Small pure functions (no React) - easy to test on their own.

/** 0.9231 -> "92.3%" */
export function confidencePercent(confidence) {
  return `${(Number(confidence) * 100).toFixed(1)}%`;
}

/** Highlight colour for a word. Same rule as the Python side: nothing below importance 0.10,
 *  otherwise the intensity grows with the importance. Returns undefined for "no highlight". */
export function highlightStyle(importance) {
  const imp = Number(importance);
  if (!Number.isFinite(imp) || imp < 0.1) return undefined;
  const alpha = 0.15 + 0.75 * Math.min(imp, 1);
  return { backgroundColor: `rgba(220, 38, 38, ${alpha.toFixed(2)})` };
}

/** 123.456 -> "123 ms",  1534 -> "1.53 s" */
export function formatMs(ms) {
  const v = Number(ms);
  if (!Number.isFinite(v)) return "-";
  return v >= 1000 ? `${(v / 1000).toFixed(2)} s` : `${Math.round(v)} ms`;
}

export function methodLabel(method) {
  if (method === "integrated_gradients") return "Integrated Gradients";
  if (method === "occlusion") return "Occlusion (fallback: Integrated Gradients did not pass its reliability check)";
  return "none";
}

/** Turn a FastAPI error body into one readable sentence. */
export function describeApiError(status, body) {
  const detail = body && body.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail) && detail.length) {
    return detail.map((d) => d.msg || JSON.stringify(d)).join("; ");
  }
  return `The server answered with HTTP ${status}.`;
}
