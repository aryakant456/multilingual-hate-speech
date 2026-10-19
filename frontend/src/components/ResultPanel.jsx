import HighlightedText from "./HighlightedText.jsx";
import { confidencePercent, formatMs, methodLabel } from "../utils.js";

export default function ResultPanel({ result }) {
  const isHate = result.prediction === "HATE";
  const words = result.words || [];
  const t = result.timings_ms;

  return (
    <section className="results">
      <div className="row">
        <span className="label">Language</span>
        <span className="value">{result.language}</span>
      </div>
      <div className="row">
        <span className="label">Prediction</span>
        <span className={`badge ${isHate ? "hate" : "safe"}`}>{result.prediction}</span>
      </div>
      <div className="row">
        <span className="label">Confidence</span>
        <span className="value">{confidencePercent(result.confidence)}</span>
      </div>

      <h2>Explainability</h2>
      {words.length > 0 ? (
        <>
          {result.important_words.length > 0 ? (
            <>
              <p className="label">Important words:</p>
              <ul className="chips">
                {result.important_words.map((w) => (
                  <li key={w}>{w}</li>
                ))}
              </ul>
            </>
          ) : (
            <p className="muted">No single word stands out strongly in this text.</p>
          )}
          <p className="label">Highlighted sentence:</p>
          <HighlightedText words={words} />
          <p className="legend">
            Shading = attribution score toward the predicted class (darker = stronger). Method: {methodLabel(result.explanation?.method)}.
            These scores describe how the model reached its output; they are not a judgement about the words themselves.
          </p>
        </>
      ) : (
        <p className="muted">
          {isHate
            ? "The explanation was switched off for this request."
            : "No words are highlighted because the model did not predict HATE for this text."}
        </p>
      )}

      {result.warnings && result.warnings.length > 0 && (
        <ul className="warnings">
          {result.warnings.map((w, i) => (
            <li key={i}>{w}</li>
          ))}
        </ul>
      )}

      {t && (
        <p className="timing">
          Response time {formatMs(result.clientMs)} (server: preprocessing {formatMs(t.preprocessing_ms)} · model {formatMs(t.inference_ms)} ·
          explanation {formatMs(t.explainability_ms)} · total {formatMs(t.total_ms)})
        </p>
      )}
    </section>
  );
}
