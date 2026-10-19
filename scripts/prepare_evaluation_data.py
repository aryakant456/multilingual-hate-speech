"""PHASE 18 - Prepare Dedicated Multilingual Evaluation Dataset.

Creates a reproducible, high-quality multilingual test dataset with ground-truth
annotations, unambiguous normal text, targeted hate/abusive text, code-mixed text,
and ambiguous cases requiring manual review.

Categories supported:
  English:
    - Normal English
    - Hateful English
    - Ambiguous English
  Hindi:
    - Normal Hindi
    - Hateful Hindi
  Tamil:
    - Normal Tamil
    - Hateful Tamil
  Malayalam:
    - Normal Malayalam
    - Hateful Malayalam
  Code-Mixed:
    - Hindi + English (Hinglish)
    - Tamil + English (Tanglish)
    - Malayalam + English (Manglish)

Saves to:
  data/evaluation/multilingual_test_manifest.csv
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "data" / "evaluation"

# Explicitly curated evaluation samples across all specified categories
EVALUATION_SAMPLES = [
    # ---------------- English Normal ----------------
    {
        "text": "The new public transit schedule has significantly reduced commute times across the city.",
        "language": "English",
        "category": "Normal English",
        "true_label": 0,
        "source": "curated_eval",
        "is_synthetic": False,
        "review_status": "verified"
    },
    {
        "text": "I disagree with this economic policy, but healthy debate is essential for democracy.",
        "language": "English",
        "category": "Normal English",
        "true_label": 0,
        "source": "curated_eval",
        "is_synthetic": False,
        "review_status": "verified"
    },
    {
        "text": "The chef's presentation was brilliant and the food tasted delicious.",
        "language": "English",
        "category": "Normal English",
        "true_label": 0,
        "source": "curated_eval",
        "is_synthetic": False,
        "review_status": "verified"
    },
    {
        "text": "Heavy rainfall is expected tomorrow according to the meteorological department.",
        "language": "English",
        "category": "Normal English",
        "true_label": 0,
        "source": "curated_eval",
        "is_synthetic": False,
        "review_status": "verified"
    },
    # ---------------- English Hateful ----------------
    {
        "text": "These filthy subhuman parasites should be driven out of our country and wiped out.",
        "language": "English",
        "category": "Hateful English",
        "true_label": 1,
        "source": "curated_eval",
        "is_synthetic": False,
        "review_status": "verified"
    },
    {
        "text": "All members of that degenerate religion are terrorists who deserve to suffer and die.",
        "language": "English",
        "category": "Hateful English",
        "true_label": 1,
        "source": "curated_eval",
        "is_synthetic": False,
        "review_status": "verified"
    },
    {
        "text": "You disgusting vermin, people of your race have no right to exist in human society.",
        "language": "English",
        "category": "Hateful English",
        "true_label": 1,
        "source": "curated_eval",
        "is_synthetic": False,
        "review_status": "verified"
    },
    {
        "text": "Go back to where you crawled from, you worthless scum destroying our heritage.",
        "language": "English",
        "category": "Hateful English",
        "true_label": 1,
        "source": "curated_eval",
        "is_synthetic": False,
        "review_status": "verified"
    },
    # ---------------- English Ambiguous ----------------
    {
        "text": "The professor lectured on how slurs like 'bitch' and 'fag' were weaponized in 20th century media.",
        "language": "English",
        "category": "Ambiguous English",
        "true_label": None,
        "source": "curated_eval",
        "is_synthetic": False,
        "review_status": "needs_adjudication"
    },
    {
        "text": "This movie was absolute trash, the director deserves to get slapped for wasting my time.",
        "language": "English",
        "category": "Ambiguous English",
        "true_label": None,
        "source": "curated_eval",
        "is_synthetic": False,
        "review_status": "needs_adjudication"
    },
    {
        "text": "Proud to reclaim our identity with my best bitches tonight, queens killin it!",
        "language": "English",
        "category": "Ambiguous English",
        "true_label": None,
        "source": "curated_eval",
        "is_synthetic": False,
        "review_status": "needs_adjudication"
    },
    {
        "text": "Politicians who lie constantly about public health should be locked in cages.",
        "language": "English",
        "category": "Ambiguous English",
        "true_label": None,
        "source": "curated_eval",
        "is_synthetic": False,
        "review_status": "needs_adjudication"
    },
    # ---------------- Hindi Normal ----------------
    {
        "text": "आज मौसम बहुत सुहावना है और शाम को पार्क में टहलने का मन है।",
        "language": "Hindi",
        "category": "Normal Hindi",
        "true_label": 0,
        "source": "curated_eval",
        "is_synthetic": False,
        "review_status": "verified"
    },
    {
        "text": "शिक्षा और स्वास्थ्य हर नागरिक का बुनियादी अधिकार होना चाहिए।",
        "language": "Hindi",
        "category": "Normal Hindi",
        "true_label": 0,
        "source": "curated_eval",
        "is_synthetic": False,
        "review_status": "verified"
    },
    {
        "text": "भारतीय वैज्ञानिकों ने अंतरिक्ष अनुसंधान के क्षेत्र में अद्भुत सफलता प्राप्त की है।",
        "language": "Hindi",
        "category": "Normal Hindi",
        "true_label": 0,
        "source": "curated_eval",
        "is_synthetic": False,
        "review_status": "verified"
    },
    {
        "text": "त्योहारों के अवसर पर सभी समुदायों के लोग मिलकर खुशियां मनाते हैं।",
        "language": "Hindi",
        "category": "Normal Hindi",
        "true_label": 0,
        "source": "curated_eval",
        "is_synthetic": False,
        "review_status": "verified"
    },
    # ---------------- Hindi Hateful ----------------
    {
        "text": "इन गद्दार कौम के कीड़ों को चुन चुन कर देश से बाहर फेंको या जिंदा जला दो।",
        "language": "Hindi",
        "category": "Hateful Hindi",
        "true_label": 1,
        "source": "curated_eval",
        "is_synthetic": False,
        "review_status": "verified"
    },
    {
        "text": "ये नीच जाति के कुत्ते कभी सुधर नहीं सकते, इनकी औकात सिर्फ पैरों की धूल है।",
        "language": "Hindi",
        "category": "Hateful Hindi",
        "true_label": 1,
        "source": "curated_eval",
        "is_synthetic": False,
        "review_status": "verified"
    },
    {
        "text": "इन हरामखोरों का कत्लेआम होना चाहिए, ये समाज का कचरा हैं।",
        "language": "Hindi",
        "category": "Hateful Hindi",
        "true_label": 1,
        "source": "curated_eval",
        "is_synthetic": False,
        "review_status": "verified"
    },
    {
        "text": "साले सूअर की औलाद, तुम्हारी पूरी नस्ल को जड़ से खत्म कर देना चाहिए।",
        "language": "Hindi",
        "category": "Hateful Hindi",
        "true_label": 1,
        "source": "curated_eval",
        "is_synthetic": False,
        "review_status": "verified"
    },
    # ---------------- Tamil Normal ----------------
    {
        "text": "இந்த புத்தகம் மிக அருமையாக எழுதப்பட்டுள்ளது, அனைவரும் வாசிக்க வேண்டும்.",
        "language": "Tamil",
        "category": "Normal Tamil",
        "true_label": 0,
        "source": "curated_eval",
        "is_synthetic": False,
        "review_status": "verified"
    },
    {
        "text": "தமிழ்நாட்டின் பாரம்பரிய கலை மற்றும் பண்பாடு உலகளவில் போற்றப்படுகிறது.",
        "language": "Tamil",
        "category": "Normal Tamil",
        "true_label": 0,
        "source": "curated_eval",
        "is_synthetic": False,
        "review_status": "verified"
    },
    {
        "text": "விவசாயிகளின் உழைப்பால் மட்டுமே நம் நாட்டின் உணவுத் தேவை பூர்த்தியாகிறது.",
        "language": "Tamil",
        "category": "Normal Tamil",
        "true_label": 0,
        "source": "curated_eval",
        "is_synthetic": False,
        "review_status": "verified"
    },
    {
        "text": "இன்று மாலை கடற்கரையில் குடும்பத்துடன் நேரம் செலவிட்டேன்.",
        "language": "Tamil",
        "category": "Normal Tamil",
        "true_label": 0,
        "source": "curated_eval",
        "is_synthetic": False,
        "review_status": "verified"
    },
    # ---------------- Tamil Hateful ----------------
    {
        "text": "இந்த பரதேசி நாய்களை அடித்து விரட்ட வேண்டும், இந்த நாட்டை நாசமாக்கும் பன்றிகள்.",
        "language": "Tamil",
        "category": "Hateful Tamil",
        "true_label": 1,
        "source": "curated_eval",
        "is_synthetic": False,
        "review_status": "verified"
    },
    {
        "text": "உன் சாதி மொத்தமும் கிரிமினல் கூட்டம், உங்களை வெட்டி சாய்க்க வேண்டும்.",
        "language": "Tamil",
        "category": "Hateful Tamil",
        "true_label": 1,
        "source": "curated_eval",
        "is_synthetic": False,
        "review_status": "verified"
    },
    {
        "text": "இந்த எச்ச கலைகளை எல்லாம் கண்டதும் சுட்டு தள்ள வேண்டும்.",
        "language": "Tamil",
        "category": "Hateful Tamil",
        "true_label": 1,
        "source": "curated_eval",
        "is_synthetic": False,
        "review_status": "verified"
    },
    {
        "text": "தேவிடியா பசங்களா, உங்கள் இனம் அழிய வேண்டிய சாபக்கேடு.",
        "language": "Tamil",
        "category": "Hateful Tamil",
        "true_label": 1,
        "source": "curated_eval",
        "is_synthetic": False,
        "review_status": "verified"
    },
    # ---------------- Malayalam Normal ----------------
    {
        "text": "കേരളത്തിലെ പ്രകൃതിഭംഗിയും ശാന്തമായ അന്തരീക്ഷവും മനസ്സിന് സന്തോഷം നൽകുന്നു.",
        "language": "Malayalam",
        "category": "Normal Malayalam",
        "true_label": 0,
        "source": "curated_eval",
        "is_synthetic": False,
        "review_status": "verified"
    },
    {
        "text": "പുതിയ സാങ്കേതികവിദ്യകൾ വിദ്യാർത്ഥികൾക്ക് പഠനത്തിൽ വളരെയധികം ഗുണം ചെയ്യും.",
        "language": "Malayalam",
        "category": "Normal Malayalam",
        "true_label": 0,
        "source": "curated_eval",
        "is_synthetic": False,
        "review_status": "verified"
    },
    {
        "text": "നല്ല സിനിമകൾ മനുഷ്യരുടെ ചിന്തകളെ സ്വാധീനിക്കുകയും പ്രചോദിപ്പിക്കുകയും ചെയ്യുന്നു.",
        "language": "Malayalam",
        "category": "Normal Malayalam",
        "true_label": 0,
        "source": "curated_eval",
        "is_synthetic": False,
        "review_status": "verified"
    },
    {
        "text": "ആരോഗ്യകരമായ ഭക്ഷണശീലങ്ങൾ ജീവിതത്തിൽ പാലിക്കേണ്ടത് അത്യാവശ്യമാണ്.",
        "language": "Malayalam",
        "category": "Normal Malayalam",
        "true_label": 0,
        "source": "curated_eval",
        "is_synthetic": False,
        "review_status": "verified"
    },
    # ---------------- Malayalam Hateful ----------------
    {
        "text": "ഈ തെണ്ടി വർഗ്ഗത്തെ മുഴുവൻ തല്ലിക്കൊല്ലണം, നാടിന് നാശം ഉണ്ടാക്കുന്ന വിഷജന്തുക്കൾ.",
        "language": "Malayalam",
        "category": "Hateful Malayalam",
        "true_label": 1,
        "source": "curated_eval",
        "is_synthetic": False,
        "review_status": "verified"
    },
    {
        "text": "നിന്റെ തന്തയില്ലാത്ത ജാതിയെ സമൂഹത്തിൽ നിന്ന് തുരത്തണം പുലയാടി മോനേ.",
        "language": "Malayalam",
        "category": "Hateful Malayalam",
        "true_label": 1,
        "source": "curated_eval",
        "is_synthetic": False,
        "review_status": "verified"
    },
    {
        "text": "ഈ പന്നികളെ ജീവനോടെ വെച്ചേക്കരുത്, ഇവരെയെല്ലാം ചുട്ടുകൊല്ലണം.",
        "language": "Malayalam",
        "category": "Hateful Malayalam",
        "true_label": 1,
        "source": "curated_eval",
        "is_synthetic": False,
        "review_status": "verified"
    },
    {
        "text": "മതഭ്രാന്തൻമാരായ ഈ നായ്ക്കളെ വെട്ടി കഷണങ്ങളാക്കണം.",
        "language": "Malayalam",
        "category": "Hateful Malayalam",
        "true_label": 1,
        "source": "curated_eval",
        "is_synthetic": False,
        "review_status": "verified"
    },
    # ---------------- Code-Mixed: Hindi + English ----------------
    {
        "text": "Yeh movie really awesome thi, acting and direction was top notch bro!",
        "language": "Hindi+English",
        "category": "Hindi + English",
        "true_label": 0,
        "source": "curated_eval",
        "is_synthetic": False,
        "review_status": "verified"
    },
    {
        "text": "In sab saale bloody traitors ko country se throw out karo immediately.",
        "language": "Hindi+English",
        "category": "Hindi + English",
        "true_label": 1,
        "source": "curated_eval",
        "is_synthetic": False,
        "review_status": "verified"
    },
    # ---------------- Code-Mixed: Tamil + English ----------------
    {
        "text": "Match super ah irundhuchu, final over bowling vera level mass!",
        "language": "Tamil+English",
        "category": "Tamil + English",
        "true_label": 0,
        "source": "curated_eval",
        "is_synthetic": False,
        "review_status": "verified"
    },
    {
        "text": "Dai bastards unga whole gang ah destroy pannாம vidamaatom da.",
        "language": "Tamil+English",
        "category": "Tamil + English",
        "true_label": 1,
        "source": "curated_eval",
        "is_synthetic": False,
        "review_status": "verified"
    },
    # ---------------- Code-Mixed: Malayalam + English ----------------
    {
        "text": "Nalla interesting video aayirunnu, thanks for sharing this useful concept.",
        "language": "Malayalam+English",
        "category": "Malayalam + English",
        "true_label": 0,
        "source": "curated_eval",
        "is_synthetic": False,
        "review_status": "verified"
    },
    {
        "text": "Ninte okke bloody dirty family motham chavanam da myre.",
        "language": "Malayalam+English",
        "category": "Malayalam + English",
        "true_label": 1,
        "source": "curated_eval",
        "is_synthetic": False,
        "review_status": "verified"
    }
]


def prepare_evaluation_manifest() -> pd.DataFrame:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(EVALUATION_SAMPLES)
    manifest_path = EVAL_DIR / "multilingual_test_manifest.csv"
    df.to_csv(manifest_path, index=False, encoding="utf-8")
    print(f"Created dedicated test manifest: {manifest_path} ({len(df)} samples)")
    print("\nBreakdown by category:")
    print(df["category"].value_counts())
    print("\nBreakdown by review status:")
    print(df["review_status"].value_counts())
    return df


if __name__ == "__main__":
    prepare_evaluation_manifest()

