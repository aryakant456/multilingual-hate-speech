import re

with open("CSI4004_Text_Mining_Case_Study_23MID0056.html", "r", encoding="utf-8") as f:
    text = f.read()

display_eqs = re.findall(r"\$\$(.*?)\$\$", text, re.DOTALL)
print(f"Total display equations found: {len(display_eqs)}")
for i, eq in enumerate(display_eqs):
    print(f"\n[Equation {i+1}]")
    print(eq.strip())

inline_eqs = re.findall(r"(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)", text)
print(f"\nTotal inline equations found: {len(inline_eqs)}")

# Check for any lingering 'ext' from broken '\text'
bad_matches = re.findall(r"(ext[a-zA-Z_]+|imes)", text)
print(f"Suspicious unescaped tokens count: {len(bad_matches)}")
if bad_matches:
    print("Tokens found:", bad_matches[:10])
else:
    print("NO corrupted math tokens! All LaTeX math syntax is clean.")

# Check for Appendices
has_appendix = "13.0 APPENDICES" in text or "13.0 Appendices" in text or "Appendix A" in text
print(f"Contains Appendices: {has_appendix}")
