import { highlightStyle } from "../utils.js";

/** The analysed sentence with every word shaded by its importance (stronger red = higher attribution). */
export default function HighlightedText({ words }) {
  return (
    <p className="highlighted">
      {words.map((w, i) => (
        <span
          key={i}
          className="word"
          style={highlightStyle(w.importance)}
          title={`attribution ${w.score >= 0 ? "+" : ""}${w.score.toFixed(3)} | importance ${w.importance.toFixed(2)} (${w.level})`}
        >
          {w.word}
        </span>
      ))}
    </p>
  );
}
