# step4_ner_pipeline/ner.py
# ─────────────────────────────────────────────────────────────────
# Named Entity Recognition (NER) pipeline using scispaCy.
#
# What it does:
#   Takes corrected prescription text and extracts structured fields:
#     - Medicine name
#     - Dosage (e.g. 500mg, 10ml)
#     - Frequency (e.g. twice daily, TDS, BD)
#     - Duration (e.g. 5 days, 1 week)
#     - Route (e.g. oral, topical)
#
# Why it's new vs minor project:
#   Minor project output = plain text string "Paracetamol"
#   Major project output = structured dict with all fields above
#
# SETUP (run once before using):
#   pip install scispacy
#   pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_core_sci_sm-0.5.3.tar.gz
#
# HOW TO RUN:
#   python step4_ner_pipeline/ner.py
# ─────────────────────────────────────────────────────────────────

import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))


# ─────────────────────────────────────────────────────────────────
# LOAD scispaCy model
# ─────────────────────────────────────────────────────────────────
def load_ner_model():
    """
    Loads the scispaCy biomedical NER model.
    en_core_sci_sm is trained on biomedical text (PubMed papers)
    and understands medical terminology natively.
    """
    try:
        import spacy
        nlp = spacy.load("en_core_sci_sm")
        print("scispaCy model loaded: en_core_sci_sm")
        return nlp
    except OSError:
        print("ERROR: scispaCy model not installed.")
        print("Run this command to install:")
        print("pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/"
              "releases/v0.5.3/en_core_sci_sm-0.5.3.tar.gz")
        return None


# ─────────────────────────────────────────────────────────────────
# REGEX PATTERNS for dosage, frequency, duration, route
# scispaCy handles medicine names; regex handles the rest
# ─────────────────────────────────────────────────────────────────

# Dosage: 500mg, 10ml, 250 mg, 5mg/ml, 1g
DOSAGE_PATTERN = re.compile(
    r'\b(\d+(?:\.\d+)?)\s*(mg|ml|mcg|g|iu|units?|tabs?|caps?|mg/ml)\b',
    re.IGNORECASE
)

# Frequency: common prescription abbreviations + plain English
FREQUENCY_PATTERN = re.compile(
    r'\b('
    r'once\s+daily|twice\s+daily|three\s+times\s+daily|'
    r'four\s+times\s+daily|every\s+\d+\s+hours?|'
    r'od|bd|tds|qid|tid|bid|prn|sos|'        # Latin abbreviations
    r'in\s+the\s+morning|at\s+bedtime|at\s+night|'
    r'before\s+meals?|after\s+meals?|with\s+meals?|'
    r'\d+\s+times?\s+(?:a\s+)?day'
    r')\b',
    re.IGNORECASE
)

# Duration: 5 days, 1 week, 2 months, for 10 days
DURATION_PATTERN = re.compile(
    r'\b(?:for\s+)?(\d+)\s+(day|days|week|weeks|month|months)\b',
    re.IGNORECASE
)

# Route of administration
ROUTE_PATTERN = re.compile(
    r'\b(oral(?:ly)?|topical(?:ly)?|intravenous(?:ly)?|'
    r'iv\b|im\b|subcutaneous(?:ly)?|sublingual(?:ly)?|'
    r'inhaled?|inhal(?:ation)?|eye\s+drops?|ear\s+drops?|'
    r'nasal\s+spray|rectal(?:ly)?)\b',
    re.IGNORECASE
)


# ─────────────────────────────────────────────────────────────────
# CORE FUNCTION: Extract entities from prescription text
# ─────────────────────────────────────────────────────────────────
def extract_entities(text: str, nlp=None) -> dict:
    """
    Extracts structured medical entities from prescription text.

    Input : "Tab Paracetamol 500mg twice daily for 5 days"
    Output: {
        "medicines"   : ["Paracetamol"],
        "dosages"     : ["500mg"],
        "frequencies" : ["twice daily"],
        "durations"   : ["5 days"],
        "routes"      : [],
        "raw_text"    : "Tab Paracetamol 500mg twice daily for 5 days"
    }
    """
    entities = {
        "medicines"   : [],
        "dosages"     : [],
        "frequencies" : [],
        "durations"   : [],
        "routes"      : [],
        "raw_text"    : text
    }

    if not text or not text.strip():
        return entities

    # ── Medicine names via scispaCy ───────────────────────────
    if nlp is not None:
        doc = nlp(text)
        for ent in doc.ents:
            # scispaCy labels: CHEMICAL, DISEASE, etc.
            # We keep entities that look like drug names
            if ent.label_ in ["CHEMICAL", "SIMPLE_CHEMICAL"]:
                name = ent.text.strip()
                if len(name) > 2 and name not in entities["medicines"]:
                    entities["medicines"].append(name)

    # ── Dosages via regex ─────────────────────────────────────
    for match in DOSAGE_PATTERN.finditer(text):
        dosage = match.group(0).strip()
        if dosage not in entities["dosages"]:
            entities["dosages"].append(dosage)

    # ── Frequencies via regex ─────────────────────────────────
    for match in FREQUENCY_PATTERN.finditer(text):
        freq = match.group(0).strip().lower()
        # Normalize common abbreviations to plain English
        freq_map = {
            "od" : "once daily",
            "bd" : "twice daily",
            "bid": "twice daily",
            "tds": "three times daily",
            "tid": "three times daily",
            "qid": "four times daily",
            "prn": "as needed",
            "sos": "if necessary"
        }
        freq = freq_map.get(freq, freq)
        if freq not in entities["frequencies"]:
            entities["frequencies"].append(freq)

    # ── Durations via regex ───────────────────────────────────
    for match in DURATION_PATTERN.finditer(text):
        duration = match.group(0).strip()
        if duration not in entities["durations"]:
            entities["durations"].append(duration)

    # ── Routes via regex ──────────────────────────────────────
    for match in ROUTE_PATTERN.finditer(text):
        route = match.group(0).strip().lower()
        if route not in entities["routes"]:
            entities["routes"].append(route)

    return entities


# ─────────────────────────────────────────────────────────────────
# PROCESS FULL PRESCRIPTION
# ─────────────────────────────────────────────────────────────────
def process_prescription(
    corrected_medicines: list,
    full_text: str = "",
    nlp=None
) -> list:
    """
    Processes all medicines detected in a prescription.

    corrected_medicines : list of dicts from step3 corrector
    full_text           : full prescription text if available
    nlp                 : loaded scispaCy model

    Returns list of structured medicine records.
    """
    results = []

    for med in corrected_medicines:
        medicine_name = med.get("corrected_name", "")
        if not medicine_name:
            continue

        # Try to extract entities from context around this medicine
        # in the full prescription text
        context = full_text if full_text else medicine_name
        entities = extract_entities(context, nlp)

        # Build structured record
        record = {
            "medicine_name"  : medicine_name,
            "generic_name"   : med.get("generic_name", ""),
            "dosage"         : entities["dosages"][0] if entities["dosages"] else "Not specified",
            "frequency"      : entities["frequencies"][0] if entities["frequencies"] else "Not specified",
            "duration"       : entities["durations"][0] if entities["durations"] else "Not specified",
            "route"          : entities["routes"][0] if entities["routes"] else "Oral (default)",
            "ocr_confidence" : med.get("confidence", "unknown"),
            "raw_ocr"        : med.get("raw_input", "")
        }
        results.append(record)

    return results


# ─────────────────────────────────────────────────────────────────
# TEST
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  Step 4 — NER Pipeline Test")
    print("=" * 55)

    nlp = load_ner_model()

    # Test with realistic prescription sentences
    test_prescriptions = [
        "Tab Paracetamol 500mg twice daily for 5 days",
        "Cap Amoxicillin 250mg TDS after meals for 7 days",
        "Ibuprofen 400mg oral BD for 3 days",
        "Metformin 500mg once daily after breakfast",
        "Azithromycin 500mg OD for 3 days",
        "Apply Ketoconazole cream topically twice daily for 2 weeks",
    ]

    print("\nExtracting entities from prescription text:\n")

    for text in test_prescriptions:
        print(f"Input: {text}")
        entities = extract_entities(text, nlp)
        print(f"  Medicines  : {entities['medicines']}")
        print(f"  Dosages    : {entities['dosages']}")
        print(f"  Frequencies: {entities['frequencies']}")
        print(f"  Durations  : {entities['durations']}")
        print(f"  Routes     : {entities['routes']}")
        print()

    print("Step 4 test complete!")
    print("Next → run: python step5_medicine_info/openfda.py")