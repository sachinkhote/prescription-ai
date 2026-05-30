# step5_medicine_info/openfda.py
# ─────────────────────────────────────────────────────────────────
# Queries the OpenFDA API to get real clinical information
# about each medicine detected in the prescription.
#
# OpenFDA = Open Food and Drug Administration database
# It's a FREE official US government drug database.
# No API key needed for basic use (1000 requests/day free).
#
# What it returns per medicine:
#   - Brand names
#   - Generic name
#   - What it's used for (indications)
#   - How to use it (dosage & administration)
#   - Warnings
#   - Side effects (adverse reactions)
#
# HOW TO RUN:
#   python step5_medicine_info/openfda.py
# ─────────────────────────────────────────────────────────────────

import sys
import requests
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config import OPENFDA_BASE_URL, OPENFDA_LIMIT


# ─────────────────────────────────────────────────────────────────
# CORE FUNCTION: Query OpenFDA for one medicine
# ─────────────────────────────────────────────────────────────────
def get_medicine_info(medicine_name: str) -> dict:
    """
    Queries the OpenFDA drug label API for a given medicine name.

    Input : "Paracetamol"
    Output: {
        "brand_name"     : "Tylenol",
        "generic_name"   : "Acetaminophen",
        "indications"    : "For relief of mild to moderate pain...",
        "dosage"         : "Adults: 325-650mg every 4-6 hours...",
        "warnings"       : "Do not exceed 4g per day...",
        "side_effects"   : "Nausea, rash...",
        "manufacturer"   : "Johnson & Johnson",
        "found"          : True
    }
    """
    empty_result = {
        "brand_name"   : medicine_name,
        "generic_name" : "",
        "indications"  : "",
        "dosage"       : "",
        "warnings"     : "",
        "side_effects" : "",
        "manufacturer" : "",
        "found"        : False
    }

    if not medicine_name or not medicine_name.strip():
        return empty_result

    # Try searching by brand name first, then generic name
    search_queries = [
        f'openfda.brand_name:"{medicine_name}"',
        f'openfda.generic_name:"{medicine_name}"',
        f'openfda.substance_name:"{medicine_name}"',
    ]

    for query in search_queries:
        try:
            params = {
                "search" : query,
                "limit"  : OPENFDA_LIMIT
            }
            response = requests.get(
                OPENFDA_BASE_URL,
                params=params,
                timeout=10
            )

            if response.status_code != 200:
                continue

            data    = response.json()
            results = data.get("results", [])

            if not results:
                continue

            # Take the first result
            label = results[0]

            # Helper to safely extract first item from a list field
            def get_field(field_name, default="Not available"):
                value = label.get(field_name, [])
                if isinstance(value, list) and value:
                    # Truncate long text to first 500 chars
                    return value[0][:500].strip()
                return default

            # Extract OpenFDA metadata
            openfda = label.get("openfda", {})

            result = {
                "brand_name"   : (
                    openfda.get("brand_name", [medicine_name])[0]
                    if openfda.get("brand_name") else medicine_name
                ),
                "generic_name" : (
                    openfda.get("generic_name", [""])[0]
                    if openfda.get("generic_name") else ""
                ),
                "indications"  : get_field("indications_and_usage"),
                "dosage"       : get_field("dosage_and_administration"),
                "warnings"     : get_field("warnings"),
                "side_effects" : get_field("adverse_reactions"),
                "manufacturer" : (
                    openfda.get("manufacturer_name", [""])[0]
                    if openfda.get("manufacturer_name") else ""
                ),
                "found"        : True
            }
            return result

        except requests.exceptions.Timeout:
            print(f"  Timeout querying OpenFDA for '{medicine_name}'")
            continue
        except Exception as e:
            print(f"  OpenFDA error for '{medicine_name}': {e}")
            continue

    # Nothing found — return empty result with the name
    print(f"  '{medicine_name}' not found in OpenFDA database")
    empty_result["found"] = False
    return empty_result


# ─────────────────────────────────────────────────────────────────
# BATCH FUNCTION: Get info for multiple medicines
# ─────────────────────────────────────────────────────────────────
def get_batch_medicine_info(medicine_names: list) -> list:
    """
    Gets OpenFDA information for a list of medicine names.
    Used when processing a full prescription with multiple medicines.
    """
    results = []
    for name in medicine_names:
        print(f"  Querying OpenFDA: {name}...")
        info = get_medicine_info(name)
        info["queried_name"] = name
        results.append(info)
    return results


# ─────────────────────────────────────────────────────────────────
# TEST
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  Step 5A — OpenFDA API Test")
    print("=" * 55)
    print("Testing with common medicines from the BD dataset...\n")

    test_medicines = [
        "Paracetamol",
        "Amoxicillin",
        "Ibuprofen",
        "Metformin",
        "Azithromycin",
    ]

    for medicine in test_medicines:
        print(f"\n{'─'*50}")
        print(f"Medicine: {medicine}")
        print(f"{'─'*50}")

        info = get_medicine_info(medicine)

        if info["found"]:
            print(f"  Brand name   : {info['brand_name']}")
            print(f"  Generic name : {info['generic_name']}")
            print(f"  Manufacturer : {info['manufacturer']}")
            print(f"  Indications  : {info['indications'][:150]}...")
            print(f"  Warnings     : {info['warnings'][:150]}...")
        else:
            print(f"  Not found in OpenFDA database")

    print("\n" + "=" * 55)
    print("  OpenFDA test complete!")
    print("  Next → run: python step5_medicine_info/explainer.py")
    print("=" * 55)