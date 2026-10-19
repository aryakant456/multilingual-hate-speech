import { useState } from "react";
import { analyzeText } from "./api.js";
import ResultPanel from "./components/ResultPanel.jsx";

const MAX_CHARS = 2000;
const EXAMPLES = [
  "I hate those disgusting people",
  "Have a wonderful day my friend",
  "आप सब लोग बहुत अच्छे हैं",
];

export default function App() {
  const [text, setText] = useState("");
  const [explain, setExplain] = useState(true);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  async function handleAnalyze() {
    if (!text.trim() || loading) return;
    setLoading(true);
    setError("");
    setResult(null);
    try {
      const { data, clientMs } = await analyzeText(text, explain);
      setResult({ ...data, clientMs });
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="app">
      <h1>Multilingual Hate / Abusive Speech Detector</h1>
      <p className="subtitle">XLM-RoBERTa · English, Hindi, Tamil, Malayalam</p>

      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Enter social media text..."
        rows={5}
        maxLength={MAX_CHARS}
        onKeyDown={(e) => {
          if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) handleAnalyze();
        }}
      />
      <div className="controls">
        <span className="muted">
          {text.length}/{MAX_CHARS}
        </span>
        <label className="check">
          <input type="checkbox" checked={explain} onChange={(e) => setExplain(e.target.checked)} />
          Explain (slower)
        </label>
        <button onClick={handleAnalyze} disabled={loading || !text.trim()}>
          {loading ? "ANALYZING..." : "ANALYZE"}
        </button>
      </div>

      <div className="examples">
        <span className="muted">Try an example:</span>
        {EXAMPLES.map((ex) => (
          <button key={ex} className="link" onClick={() => setText(ex)}>
            {ex}
          </button>
        ))}
      </div>

      {error && <div className="error">{error}</div>}
      {result && <ResultPanel result={result} />}
    </main>
  );
}
