# step5_medicine_info/explainer.py
# ─────────────────────────────────────────────────────────────────
# Takes raw clinical data from OpenFDA and converts it into
# plain, simple language that any patient can understand.
#
# Why this matters:
#   OpenFDA data is written for healthcare professionals —
#   full of medical jargon. This module uses Gemini to
#   translate it into plain language for patients.
#
# HOW TO RUN:
#   python step5_medicine_info/explainer.py
# ─────────────────────────────────────────────────────────────────

import os
import sys
import json
import requests
from pathlib import Path
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).parent.parent))

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_URL     = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.5-flash:generateContent"
)


# ─────────────────────────────────────────────────────────────────
# CORE FUNCTION: Generate plain-language medicine explanation
# ─────────────────────────────────────────────────────────────────
def explain_medicine(medicine_info: dict) -> dict:
    """
    Takes OpenFDA drug data and generates a patient-friendly
    explanation using Gemini.

    Input : dict from openfda.get_medicine_info()
    Output: {
        "summary"       : "Paracetamol is a common painkiller...",
        "uses"          : "Used to relieve mild to moderate pain...",
        "how_to_take"   : "Take 1-2 tablets every 4-6 hours...",
        "side_effects"  : "May cause nausea or stomach upset...",
        "warnings"      : "Do not take more than 8 tablets per day...",
        "medicine_name" : "Paracetamol"
    }
    """
    medicine_name = medicine_info.get("brand_name",
                    medicine_info.get("queried_name", "Unknown Medicine"))

    # Build context from OpenFDA data
    context_parts = []
    if medicine_info.get("generic_name"):
        context_parts.append(
            f"Generic name: {medicine_info['generic_name']}"
        )
    if medicine_info.get("indications"):
        context_parts.append(
            f"Medical indications: {medicine_info['indications'][:400]}"
        )
    if medicine_info.get("dosage"):
        context_parts.append(
            f"Dosage information: {medicine_info['dosage'][:300]}"
        )
    if medicine_info.get("warnings"):
        context_parts.append(
            f"Warnings: {medicine_info['warnings'][:300]}"
        )
    if medicine_info.get("side_effects"):
        context_parts.append(
            f"Side effects: {medicine_info['side_effects'][:300]}"
        )

    # If no OpenFDA data found, ask Gemini from its own knowledge
    if not context_parts or not medicine_info.get("found"):
        context = (
            f"Medicine name: {medicine_name}\n"
            f"(Use your general pharmaceutical knowledge)"
        )
    else:
        context = "\n".join(context_parts)

    prompt = f"""You are a friendly pharmacist explaining a medicine to a patient.

Medicine: {medicine_name}

Clinical information:
{context}

Create a clear, simple patient-friendly explanation. Use plain English that 
anyone can understand. Avoid medical jargon. Be concise and helpful.

Respond ONLY with valid JSON in this exact format, nothing else:
{{
    "summary": "One sentence: what this medicine is in simple terms",
    "uses": "What this medicine is commonly used for (2-3 sentences, simple language)",
    "how_to_take": "General guidance on how to take it (2-3 sentences)",
    "side_effects": "Common side effects a patient should know about (2-3 sentences)",
    "warnings": "Key warnings or precautions (1-2 sentences)",
    "important_note": "Always consult your doctor or pharmacist for personalized advice."
}}

Rules:
- Write as if talking to a patient, not a doctor
- Keep each field to 2-3 short sentences maximum
- If information is not available, write a general helpful statement
- Return ONLY valid JSON, no markdown, no explanation"""

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature"    : 0.3,
            "maxOutputTokens": 600,
        }
    }

    try:
        response = requests.post(
            f"{GEMINI_URL}?key={GEMINI_API_KEY}",
            json=payload,
            timeout=20
        )
        response.raise_for_status()
        data     = response.json()
        raw_text = (
            data["candidates"][0]["content"]["parts"][0]["text"]
            .strip()
        )
        raw_text = raw_text.replace("```json", "").replace("```", "").strip()
        result   = json.loads(raw_text)
        result["medicine_name"] = medicine_name
        return result

    except json.JSONDecodeError:
        return _fallback_explanation(medicine_name)
    except Exception as e:
        print(f"  Gemini error for '{medicine_name}': {e}")
        return _fallback_explanation(medicine_name)


def _fallback_explanation(medicine_name: str) -> dict:
    """Returns a generic fallback if Gemini fails."""
    return {
        "summary"        : f"{medicine_name} is a medicine prescribed by your doctor.",
        "uses"           : "Please consult your doctor or pharmacist for details on usage.",
        "how_to_take"    : "Follow your doctor's instructions carefully.",
        "side_effects"   : "Every medicine may have side effects. Consult your pharmacist.",
        "warnings"       : "Do not stop taking this medicine without consulting your doctor.",
        "important_note" : "Always consult your doctor or pharmacist for personalized advice.",
        "medicine_name"  : medicine_name
    }


# ─────────────────────────────────────────────────────────────────
# FULL PIPELINE: OpenFDA + Explanation for one medicine
# ─────────────────────────────────────────────────────────────────
def get_full_medicine_card(medicine_name: str) -> dict:
    """
    Complete pipeline for one medicine:
    1. Query OpenFDA for clinical data
    2. Generate plain-language explanation via Gemini
    3. Return a complete medicine card

    This is what the Streamlit app calls for each detected medicine.
    """
    from step5_medicine_info.openfda import get_medicine_info

    print(f"  Getting info for: {medicine_name}")
    fda_info    = get_medicine_info(medicine_name)
    explanation = explain_medicine(fda_info)

    return {
        "medicine_name" : medicine_name,
        "fda_data"      : fda_info,
        "explanation"   : explanation
    }


# ─────────────────────────────────────────────────────────────────
# TEST
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  Step 5B — Medicine Explainer Test")
    print("=" * 55)

    if not GEMINI_API_KEY:
        print("ERROR: GEMINI_API_KEY not found in .env")
        sys.exit(1)

    # Simulate what we'd get from OpenFDA
    test_fda_data = {
        "brand_name"   : "Paracetamol",
        "generic_name" : "Acetaminophen",
        "indications"  : (
            "For the temporary relief of minor aches and pains "
            "due to headache, muscular aches, backache, minor pain "
            "of arthritis, toothache, and for reduction of fever."
        ),
        "dosage"       : (
            "Adults and children 12 years and over: take 2 tablets "
            "every 4 to 6 hours while symptoms last. Do not take more "
            "than 8 tablets in 24 hours."
        ),
        "warnings"     : (
            "Liver warning: This product contains acetaminophen. "
            "Severe liver damage may occur if you take more than 4,000 mg "
            "of acetaminophen in 24 hours."
        ),
        "side_effects" : (
            "Nausea, stomach pain, loss of appetite, itching, rash, "
            "headache, dark urine, clay-colored stools, jaundice."
        ),
        "found"        : True
    }

    print("\nGenerating patient-friendly explanation for Paracetamol...\n")
    result = explain_medicine(test_fda_data)

    print("MEDICINE CARD — Patient View")
    print("─" * 50)
    print(f"Medicine   : {result['medicine_name']}")
    print(f"Summary    : {result['summary']}")
    print(f"Uses       : {result['uses']}")
    print(f"How to take: {result['how_to_take']}")
    print(f"Side effects: {result['side_effects']}")
    print(f"Warnings   : {result['warnings']}")
    print(f"Note       : {result['important_note']}")

    print("\n" + "=" * 55)
    print("  Step 5B test complete!")
    print("  Next → run: streamlit run app/streamlit_app.py")
    print("=" * 55)