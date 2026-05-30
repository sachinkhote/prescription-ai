# step3_llm_postprocessor/corrector.py
# ─────────────────────────────────────────────────────────────────
# Replaces the minor project's difflib dictionary matcher with
# an LLM-based correction module using Gemini API.
#
# What it does:
#   Takes raw OCR output (possibly noisy/wrong) like "Ketoma"
#   and asks Gemini to correct it to a real medicine name "Ketoconazole"
#
# Why it's better than difflib:
#   - Works for ANY medicine, not just ones in a fixed list
#   - Understands medical context
#   - Handles completely new medicines automatically
#
# HOW TO RUN (test only):
#   python step3_llm_postprocessor/corrector.py
# ─────────────────────────────────────────────────────────────────

import os
import sys
import json
import requests
from pathlib import Path
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).parent.parent))

# Load API key from .env file
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_URL     = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.5-flash:generateContent"
)

import time # Import time for sleep function

# ─────────────────────────────────────────────────────────────────
# CORE FUNCTION: Correct a single OCR prediction
# ─────────────────────────────────────────────────────────────────
def correct_medicine_name(raw_ocr_text: str) -> dict:
    """
    Sends raw OCR output to Gemini and asks it to:
    1. Correct spelling to a real medicine name
    2. Return the generic name if known
    3. Indicate confidence level

    Returns a dict:
    {
        "corrected_name" : "Paracetamol",
        "generic_name"   : "Acetaminophen",
        "confidence"     : "high",
        "raw_input"      : "Paracetamol"
    }
    """
    if not raw_ocr_text or not raw_ocr_text.strip():
        return {
            "corrected_name" : "",
            "generic_name"   : "",
            "confidence"     : "none",
            "raw_input"      : raw_ocr_text
        }

    # Build the prompt
    # We ask for JSON output so it's easy to parse
    prompt = f"""You are a pharmaceutical expert assistant.

I have an OCR system that reads handwritten medicine names from prescriptions.
The OCR output may contain spelling errors due to messy handwriting.

Raw OCR output: "{raw_ocr_text}"

Your task:
1. Identify the most likely real medicine/drug name this refers to
2. Correct any spelling errors
3. Provide the generic name if known
4. Rate your confidence: high, medium, or low

Respond ONLY with valid JSON in this exact format, nothing else:
{{
    "corrected_name": "the corrected brand/medicine name",
    "generic_name": "the generic/chemical name or empty string if unknown",
    "confidence": "high or medium or low"
}}

Rules:
- If the input already looks correct, return it as-is
- If completely unrecognizable, return your best guess with low confidence
- Never return null values, use empty string instead
- Return ONLY the JSON, no explanation"""

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature"    : 0.1,   # low temp = more deterministic output
            "maxOutputTokens": 1024,
            "thinkingConfig": {"thinkingBudget": 0} # disable thinking to save tokens
        }
    }

    try:
        response = requests.post(
            f"{GEMINI_URL}?key={GEMINI_API_KEY}",
            json=payload,
            timeout=15
        )
        response.raise_for_status()
        data = response.json()

        # Extract text from Gemini response
        raw_text = (
            data["candidates"][0]["content"]["parts"][0]["text"]
            .strip()
        )
        
        # Clean up markdown code blocks if Gemini adds them
        raw_text = raw_text.replace("```json", "").replace("```", "").strip()

        # Parse JSON response
        result = json.loads(raw_text)
        result["raw_input"] = raw_ocr_text
        return result

    except json.JSONDecodeError:
        # Gemini returned something but not valid JSON
        return {
            "corrected_name" : raw_ocr_text,
            "generic_name"   : "",
            "confidence"     : "low",
            "raw_input"      : raw_ocr_text
        }
    except Exception as e:
        # API error — return original text unchanged
        print(f"  Gemini API error: {e}")
        return {
            "corrected_name" : raw_ocr_text,
            "generic_name"   : "",
            "confidence"     : "error",
            "raw_input"      : raw_ocr_text
        }


# ─────────────────────────────────────────────────────────────────
# BATCH FUNCTION: Correct a list of OCR predictions
# ─────────────────────────────────────────────────────────────────
def correct_batch(ocr_predictions: list) -> list:
    """
    Corrects a list of OCR predictions.
    Used when processing a full prescription with multiple medicines.

    Input  : ["Paracetamol", "Amoxcillin", "Ibuprof"]
    Output : [
        {"corrected_name": "Paracetamol", "confidence": "high", ...},
        {"corrected_name": "Amoxicillin", "confidence": "high", ...},
        {"corrected_name": "Ibuprofen",   "confidence": "high", ...}
    ]
    """
    results = []
    for pred in ocr_predictions:
        result = correct_medicine_name(pred)
        # Add a delay to avoid hitting API rate limits
        time.sleep(10)
        results.append(result)
        print(f"  '{pred}' → '{result['corrected_name']}' "
              f"[{result['confidence']}]")
    return results


# ─────────────────────────────────────────────────────────────────
# TEST: Run this file directly to test the corrector
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  Step 3 — LLM Post-Processor Test")
    print("=" * 55)

    if not GEMINI_API_KEY:
        print("ERROR: GEMINI_API_KEY not found in .env file")
        print("Create .env file with: GEMINI_API_KEY=your_key")
        sys.exit(1)

    print(f"Gemini API key loaded: {GEMINI_API_KEY[:8]}...")

    # Test cases — these simulate typical TrOCR errors
    # on handwritten medicine names
    test_cases = [
        "Paracetamol",    # correct — should stay the same
        "Amoxcillin",     # missing 'i' — common OCR error
    ]

    print(f"\nTesting {len(test_cases)} OCR predictions...\n")
    results = correct_batch(test_cases)

    print("\n" + "=" * 55)
    print("  RESULTS SUMMARY")
    print("=" * 55)
    print(f"{'Raw OCR':<20} {'Corrected':<20} {'Generic':<20} {'Confidence'}")
    print("-" * 75)
    for r in results:
        print(
            f"{r['raw_input']:<20} "
            f"{r['corrected_name']:<20} "
            f"{r['generic_name']:<20} "
            f"{r['confidence']}"
        )

    print("\nStep 3 test complete!")
    print("Next → run: python step4_ner_pipeline/ner.py")