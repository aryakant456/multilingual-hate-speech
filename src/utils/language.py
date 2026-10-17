"""Script-based language identification for the prediction pipeline (Phase 13).

The project languages are written in different scripts, so the Unicode block of the letters
tells us the language reliably for NATIVE-script text:
    Devanagari -> Hindi, Tamil script -> Tamil, Malayalam script -> Malayalam, ...
Known limits (mention them in the report):
    * Devanagari is also used by Marathi / Nepali -> reported as "Hindi".
    * Latin script is reported as "English" even if it is romanised Hindi/Tamil ("code-mixed" text).
"""
from __future__ import annotations

import unicodedata

# (first code point, last code point, script name, language name)
_BLOCKS = [
    (0x0900, 0x097F, "Devanagari", "Hindi"),
    (0x0980, 0x09FF, "Bengali", "Bengali"),
    (0x0A00, 0x0A7F, "Gurmukhi", "Punjabi"),
    (0x0A80, 0x0AFF, "Gujarati", "Gujarati"),
    (0x0B80, 0x0BFF, "Tamil", "Tamil"),
    (0x0C00, 0x0C7F, "Telugu", "Telugu"),
    (0x0C80, 0x0CFF, "Kannada", "Kannada"),
    (0x0D00, 0x0D7F, "Malayalam", "Malayalam"),
    (0x0600, 0x06FF, "Arabic", "Urdu"),
    (0x0750, 0x077F, "Arabic", "Urdu"),
    (0xFB50, 0xFDFF, "Arabic", "Urdu"),
    (0xFE70, 0xFEFF, "Arabic", "Urdu"),
]


def detect_language(text: str) -> dict:
    """Return {"language": ..., "script": ..., "share": fraction of counted characters in that script}.
    Indic vowel signs are combining marks (not "letters" for Python), so marks inside a script block are
    counted too - otherwise Indic words would be undercounted next to English words."""
    counts: dict[tuple[str, str], int] = {}
    for ch in str(text):
        is_mark = unicodedata.category(ch).startswith("M")
        if not (ch.isalpha() or is_mark):
            continue
        cp = ord(ch)
        key = None
        if ch.isalpha() and (ch.isascii() or 0x00C0 <= cp <= 0x024F):
            key = ("Latin", "English")
        else:
            for lo, hi, script, lang in _BLOCKS:
                if lo <= cp <= hi:
                    key = (script, lang)
                    break
        if key is not None:
            counts[key] = counts.get(key, 0) + 1
    total = sum(counts.values())
    if total == 0:
        return {"language": "Unknown", "script": "Unknown", "share": 0.0}
    (script, lang), n = max(counts.items(), key=lambda kv: kv[1])
    return {"language": lang, "script": script, "share": round(n / total, 3)}
