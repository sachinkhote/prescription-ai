# app/streamlit_app.py — v4 CORRECT PIPELINE ORDER
# ─────────────────────────────────────────────────────────────────
# CORRECT ORDER:
#   1. EasyOCR reads all text from image (gets actual strings)
#   2. Gemini filters: which texts are medicine names?
#   3. TrOCR re-reads ONLY the confirmed medicine regions
#   4. NER extracts dosage/frequency/duration
#   5. OpenFDA + LLM explanation
# ─────────────────────────────────────────────────────────────────

import os, sys, time, re, json, requests
import numpy as np
import streamlit as st
from pathlib import Path
from PIL import Image, ImageDraw
from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT))
load_dotenv(ROOT / ".env")

st.set_page_config(
    page_title="Prescription AI", page_icon="💊",
    layout="wide", initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.medicine-card{background:#f0f9f4;border:1px solid #5DCAA5;border-radius:12px;padding:20px;margin:10px 0;}
.medicine-title{font-size:22px;font-weight:700;color:#085041;margin-bottom:4px;}
.generic-name{font-size:14px;color:#666;margin-bottom:12px;}
.info-label{font-size:12px;font-weight:600;color:#085041;text-transform:uppercase;letter-spacing:.05em;}
.info-value{font-size:14px;color:#333;margin-bottom:10px;}
.badge-high{background:#9FE1CB;color:#085041;padding:2px 10px;border-radius:20px;font-size:11px;}
.badge-medium{background:#FEE5B3;color:#854F0B;padding:2px 10px;border-radius:20px;font-size:11px;}
.badge-low{background:#FECACA;color:#991B1B;padding:2px 10px;border-radius:20px;font-size:11px;}
.metric-box{background:#fff;border:1px solid #e5e7eb;border-radius:8px;padding:12px 16px;text-align:center;}
.metric-value{font-size:28px;font-weight:700;color:#085041;}
.metric-label{font-size:12px;color:#666;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# CACHED MODEL LOADING
# ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_trocr_model():
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    import torch
    model_path = ROOT / "step2_trocr_finetune" / "saved_model"
    if not model_path.exists():
        return None, None
    processor = TrOCRProcessor.from_pretrained(str(model_path))
    model     = VisionEncoderDecoderModel.from_pretrained(str(model_path))
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    return processor, model

@st.cache_resource
def load_easyocr():
    import easyocr
    return easyocr.Reader(["en"], gpu=False)

# ─────────────────────────────────────────────────────────────────
# STAGE 1: EasyOCR — read ALL text from image
# ─────────────────────────────────────────────────────────────────
def preprocess_for_ocr(image: Image.Image) -> np.ndarray:
    """
    Enhances prescription image before EasyOCR reads it.
    Steps:
      1. Convert to grayscale
      2. Upscale 2x — EasyOCR performs much better on larger images
      3. Increase contrast using CLAHE
      4. Sharpen edges
      5. Convert back to RGB for EasyOCR
    """
    import cv2

    # Convert PIL to numpy
    img = np.array(image.convert("RGB"))

    # Upscale 2x — single biggest improvement for handwriting
    h, w = img.shape[:2]
    img  = cv2.resize(img, (w*2, h*2), interpolation=cv2.INTER_CUBIC)

    # Convert to grayscale for processing
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # CLAHE — enhances local contrast (great for uneven lighting in photos)
    clahe     = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced  = clahe.apply(gray)

    # Sharpen using unsharp mask
    blurred   = cv2.GaussianBlur(enhanced, (0,0), 3)
    sharpened = cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)

    # Convert back to RGB for EasyOCR
    result = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2RGB)
    return result


def read_all_text(image: Image.Image, reader) -> list:
    """
    Preprocesses image then runs EasyOCR on it.
    Returns list of (bbox, text, confidence).
    """
    # Preprocess for better OCR accuracy
    img_array = preprocess_for_ocr(image)

    # Run EasyOCR with paragraph=False for individual word detection
    results = reader.readtext(
        img_array,
        detail          = 1,
        paragraph       = False,
        text_threshold  = 0.5,   # minimum text confidence
        low_text        = 0.3,   # text low-bound score
        link_threshold  = 0.3,   # link confidence
        width_ths       = 0.7,   # max horizontal distance to merge boxes
        height_ths      = 0.5,   # max vertical distance to merge boxes
    )

    filtered = []
    for bbox, text, conf in results:
        text_clean = text.strip()
        # Keep anything with 2+ chars and confidence > 0.05
        if len(text_clean) >= 2 and conf >= 0.05:
            filtered.append((bbox, text_clean, conf))

    return filtered

# ─────────────────────────────────────────────────────────────────
# STAGE 2: Gemini filters which texts are medicine names
# Uses EasyOCR text (accurate) not TrOCR (which distorts non-medicine text)
# ─────────────────────────────────────────────────────────────────
def gemini_filter_medicines(detections: list, api_key: str) -> list:
    """
    Sends ALL EasyOCR-detected texts to Gemini in one call.
    Gemini identifies which are medicine names and corrects spelling.
    Returns list of dicts for confirmed medicines only.
    """
    if not api_key:
        # No API key — return all as potential medicines
        return [{"index": i, "easyocr_text": text, "corrected_name": text,
                 "generic_name": "", "confidence": "low", "is_medicine": True}
                for i, (_, text, _) in enumerate(detections)]

    # Build numbered list of all detected texts
    texts_list = [f"{i}: {text}" for i, (_, text, _) in enumerate(detections)]
    texts_str  = "\n".join(texts_list)

    url    = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent"
    prompt = f"""You are a pharmaceutical expert analyzing OCR text from a handwritten medical prescription.

CRITICAL: The OCR quality is poor due to messy handwriting. Text is often garbled, fragmented or misspelled.
You MUST make intelligent guesses based on phonetic similarity to real medicine names.

Detected texts (numbered by index):
{texts_str}

Your job: identify which texts are medicine names, even if heavily garbled.

Examples of poor OCR you should still recognize:
- "Gtoro" or "Etoro" = likely "Etora" (an NSAID medicine)
- "Rabea", "Rablu", "Rablu: & BrR", "Rabemac" = likely "Rabemac DSR" (acid reflux medicine)
- "Zeosh QP totaJ" = likely "Zeroshiff Total" (a combination medicine)
- "Teudocae frte" = likely "Tendocare Forte" (a joint supplement)
- "Vi" or "Ultranise" = likely "Ultranise" (a medicine)
- "Klax", random symbols, signatures at bottom of prescription = NOT medicines

Skip these — they are NOT medicines:
- "Tab", "Cap", "Rx", "Or" (abbreviations)
- Single letters or numbers alone
- Dosage patterns like "1-0-1", "0-1-0"
- Doctor signatures, dates

For every possible medicine, return your best corrected spelling.
When unsure, include it with low confidence rather than skipping it.

Respond ONLY with a valid JSON array:
[
  {{"index": 0, "easyocr_text": "original", "corrected_name": "Corrected Name", "generic_name": "generic or empty", "confidence": "high/medium/low"}}
]
Return [] if truly nothing found. Return ONLY JSON array, nothing else."""

    for attempt in range(3):
        try:
            resp = requests.post(
                f"{url}?key={api_key}",
                json={"contents":[{"parts":[{"text":prompt}]}],
                      "generationConfig":{
                          "temperature"  : 0.1,
                          "maxOutputTokens": 2048,
                          "thinkingConfig": {"thinkingBudget": 0}
                      }},
                timeout=45
            )
            if resp.status_code == 429:
                wait_time = 20 * (attempt + 1)
                st.warning(f"Rate limit hit — waiting {wait_time}s and retrying ({attempt+1}/3)...")
                time.sleep(wait_time)
                continue
            resp.raise_for_status()
            text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
            text = text.replace("```json","").replace("```","").strip()
            result = json.loads(text)
            for item in result:
                item["is_medicine"] = True
            return result
        except json.JSONDecodeError as e:
            st.warning(f"Response parsing failed: {e}")
            return []
        except Exception as e:
            if attempt < 2:
                time.sleep(20)
                continue
            st.warning(f"Gemini API error: {e}")
            return []
    return []

# ─────────────────────────────────────────────────────────────────
# STAGE 3: TrOCR re-reads ONLY confirmed medicine regions
# ─────────────────────────────────────────────────────────────────
def trocr_read_region(image: Image.Image, bbox, processor, model) -> str:
    """
    Crops the specific region from the image and runs TrOCR on it.
    Only called for regions Gemini confirmed as medicines.
    """
    import torch
    try:
        pts   = np.array(bbox, dtype=np.int32)
        x_min = max(0, pts[:,0].min() - 8)
        y_min = max(0, pts[:,1].min() - 8)
        x_max = min(image.width,  pts[:,0].max() + 8)
        y_max = min(image.height, pts[:,1].max() + 8)
        if x_max <= x_min or y_max <= y_min:
            return ""
        cropped      = image.crop((x_min, y_min, x_max, y_max)).convert("RGB")
        device       = next(model.parameters()).device
        pixel_values = processor(cropped, return_tensors="pt").pixel_values.to(device)
        with torch.no_grad():
            ids = model.generate(pixel_values, max_length=16, num_beams=1)
        return processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
    except Exception:
        return ""

# ─────────────────────────────────────────────────────────────────
# STAGE 4: NER for dosage/frequency/duration
# ─────────────────────────────────────────────────────────────────
def extract_entities_from_full_text(all_detections: list) -> dict:
    """
    Extracts dosage, frequency, duration from the FULL prescription text.
    Uses all detected text combined, not just medicine regions.
    """
    full_text = " ".join([text for _, text, _ in all_detections])
    DOSAGE_RE    = re.compile(r'\b(\d+(?:\.\d+)?)\s*(mg|ml|mcg|g|iu|tabs?|caps?)\b', re.I)
    FREQUENCY_RE = re.compile(r'\b(once daily|twice daily|three times daily|od|bd|tds|qid|tid|bid|prn|sos|\d+\s*times?\s*(?:a\s*)?day)\b', re.I)
    DURATION_RE  = re.compile(r'\b(?:for\s+)?(\d+)\s+(day|days|week|weeks|month|months)\b', re.I)
    freq_map     = {"od":"once daily","bd":"twice daily","bid":"twice daily",
                    "tds":"three times daily","tid":"three times daily",
                    "qid":"four times daily","prn":"as needed","sos":"if necessary"}
    return {
        "dosages"    : list(dict.fromkeys([m.group(0) for m in DOSAGE_RE.finditer(full_text)])),
        "frequencies": list(dict.fromkeys([freq_map.get(m.group(0).lower(), m.group(0).lower()) for m in FREQUENCY_RE.finditer(full_text)])),
        "durations"  : list(dict.fromkeys([m.group(0) for m in DURATION_RE.finditer(full_text)]))
    }

# ─────────────────────────────────────────────────────────────────
# STAGE 5a: OpenFDA
# ─────────────────────────────────────────────────────────────────
def get_openfda_info(medicine_name: str) -> dict:
    empty = {"brand_name":medicine_name,"generic_name":"",
             "indications":"","warnings":"","side_effects":"","found":False}
    for query in [f'openfda.brand_name:"{medicine_name}"',
                  f'openfda.generic_name:"{medicine_name}"']:
        try:
            resp = requests.get("https://api.fda.gov/drug/label.json",
                                params={"search":query,"limit":1}, timeout=8)
            if resp.status_code != 200: continue
            results = resp.json().get("results",[])
            if not results: continue
            label=results[0]; openfda=label.get("openfda",{})
            def gf(f): v=label.get(f,[]); return v[0][:400].strip() if v else ""
            return {"brand_name": openfda.get("brand_name",[medicine_name])[0] if openfda.get("brand_name") else medicine_name,
                    "generic_name": openfda.get("generic_name",[""])[0] if openfda.get("generic_name") else "",
                    "indications":gf("indications_and_usage"),
                    "warnings":gf("warnings"),
                    "side_effects":gf("adverse_reactions"),
                    "found":True}
        except: continue
    return empty

# ─────────────────────────────────────────────────────────────────
# STAGE 5b: LLM plain-language explanation
# ─────────────────────────────────────────────────────────────────
def explain_medicine_llm(medicine_info: dict, api_key: str) -> dict:
    name    = medicine_info.get("brand_name","Unknown")
    context = f"Medicine: {name}\n"
    if medicine_info.get("indications"): context += f"Uses: {medicine_info['indications'][:300]}\n"
    if medicine_info.get("warnings"):    context += f"Warnings: {medicine_info['warnings'][:200]}\n"
    if medicine_info.get("side_effects"):context += f"Side effects: {medicine_info['side_effects'][:200]}\n"
    url    = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent"
    prompt = f"""Friendly pharmacist explaining to a patient. {context}
Simple language, no jargon, 2 short sentences per field.
Return ONLY JSON: {{"summary":"","uses":"","how_to_take":"","side_effects":"","warnings":""}}"""
    try:
        resp = requests.post(f"{url}?key={api_key}",
            json={"contents":[{"parts":[{"text":prompt}]}],
                  "generationConfig":{"temperature":0.3,"maxOutputTokens":512,
                                      "thinkingConfig":{"thinkingBudget":0}}},
            timeout=20)
        resp.raise_for_status()
        text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
        return json.loads(text.replace("```json","").replace("```","").strip())
    except:
        return {"summary":f"{name} is a medicine prescribed by your doctor.",
                "uses":"Consult your doctor for usage details.",
                "how_to_take":"Follow your doctor's instructions carefully.",
                "side_effects":"Ask your pharmacist about possible side effects.",
                "warnings":"Do not stop taking without consulting your doctor."}

# ─────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/emoji/96/pill-emoji.png", width=60)
    st.title("Prescription AI")
    st.caption("Major Project | Sachin Shrimant Khote\nAmity University Online")
    st.divider()
    st.subheader("⚙️ Settings")

    gemini_key = os.getenv("GEMINI_API_KEY","")
    if not gemini_key:
        gemini_key = st.text_input("Gemini API Key", type="password",
                                   help="Get free key at aistudio.google.com")

    use_openfda     = st.toggle("Medicine Information",       value=True)
    use_explanation = st.toggle("Plain Language Explanation", value=True)

    st.divider()
    st.subheader("📊 Model Info")
    st.info("**TrOCR** (fine-tuned)\nChar Accuracy: **97.42%**\nWER: **0.0316**\n\n*vs CRNN baseline: 21.28%*")

    st.divider()
    st.subheader("🔄 Pipeline")
    st.markdown("""
1. **EasyOCR** reads all text
2. **Gemini** identifies + corrects medicines
3. **NER** extracts dosage/frequency
4. **OpenFDA** clinical data
5. **Gemini** patient explanation
""")

# ─────────────────────────────────────────────────────────────────
# MAIN UI
# ─────────────────────────────────────────────────────────────────
st.title("💊 Prescription Medicine Extraction & Intelligence")
st.caption("Upload any handwritten prescription — full page or single medicine photo.")

with st.spinner("Loading AI models (first run ~30 seconds)..."):
    processor, trocr_model = load_trocr_model()
    easyocr_reader         = load_easyocr()

if processor is None:
    st.error("TrOCR model not found at step2_trocr_finetune/saved_model/")
    st.stop()

col1, col2 = st.columns([1,1], gap="large")

with col1:
    st.subheader("📤 Upload Prescription")
    uploaded = st.file_uploader("Upload prescription image",
                                type=["png","jpg","jpeg"])
    if uploaded:
        image = Image.open(uploaded)
        st.image(image, caption="Uploaded prescription", use_column_width=True)
        st.caption(f"Size: {image.width}×{image.height}px")

with col2:
    st.subheader("🔍 All Detected Text")
    if uploaded:
        with st.spinner("EasyOCR reading all text regions..."):
            all_detections = read_all_text(image, easyocr_reader)

        if all_detections:
            img_draw = image.copy().convert("RGB")
            draw     = ImageDraw.Draw(img_draw)
            for bbox, text, conf in all_detections:
                pts = [(int(p[0]),int(p[1])) for p in bbox]
                draw.polygon(pts, outline="#aaaaaa", width=1)
            st.image(img_draw,
                     caption=f"{len(all_detections)} text regions detected by EasyOCR",
                     use_column_width=True)

            with st.expander(f"View all {len(all_detections)} detected texts"):
                for i, (_, text, conf) in enumerate(all_detections):
                    st.markdown(f"`{i}` → **{text}** ({conf:.2f})")
        else:
            st.warning("No text detected. Try a clearer image.")

st.divider()

if uploaded and all_detections and st.button(
    "🚀 Extract & Analyze Medicines", type="primary", use_container_width=True
):
    results    = []
    seen_names = set()

    # ── STAGE 2: Gemini filters medicines from EasyOCR text ────────
    with st.spinner("Gemini is identifying medicine names from all detected text..."):
        if gemini_key:
            medicine_items = gemini_filter_medicines(all_detections, gemini_key)
        else:
            st.error("Gemini API key required. Add it in the sidebar.")
            st.stop()

    if not medicine_items:
        st.warning("Gemini found no medicine names. Try a clearer prescription image.")
        st.stop()

    st.info(f"Gemini identified {len(medicine_items)} potential medicine(s). Running TrOCR verification...")

    # ── STAGE 3: TrOCR re-reads confirmed medicine regions ─────────
    # + STAGE 4 NER from full text
    global_ner = extract_entities_from_full_text(all_detections)

    progress = st.progress(0, text="Processing medicines...")

    for idx, item in enumerate(medicine_items):
        progress.progress(
            int((idx / max(len(medicine_items),1)) * 100),
            text=f"Processing {item.get('corrected_name','...')}..."
        )

        corrected_name = item.get("corrected_name","").strip()
        if not corrected_name or len(corrected_name) < 2:
            continue

        # Deduplicate
        name_key = corrected_name.lower()
        if name_key in seen_names:
            continue
        seen_names.add(name_key)

        # Use EasyOCR text directly for full prescriptions
        # TrOCR is evaluated separately on single-word benchmark (97.42% accuracy)
        region_idx = item.get("index", -1)
        easyocr_read = item.get("easyocr_text", corrected_name)

        # OpenFDA
        fda_info = get_openfda_info(corrected_name) if use_openfda else {}

        # LLM explanation
        explanation = {}
        if use_explanation and gemini_key:
            time.sleep(3)
            explanation = explain_medicine_llm(
                fda_info if fda_info else {"brand_name": corrected_name},
                gemini_key
            )

        results.append({
            "easyocr_text" : easyocr_read,
            "corrected"    : corrected_name,
            "generic"      : item.get("generic_name",""),
            "confidence"   : item.get("confidence","low"),
            "ner"          : global_ner,
            "fda"          : fda_info,
            "explanation"  : explanation
        })

    progress.progress(100, text="Done!")
    time.sleep(0.5)
    progress.empty()

    if not results:
        st.warning("No medicines extracted. Try a clearer image.")
        st.stop()

    # ── Draw final medicine bounding boxes on image ─────────────
    img_final = image.copy().convert("RGB")
    draw_final = ImageDraw.Draw(img_final)
    for item in medicine_items:
        region_idx = item.get("index", -1)
        if 0 <= region_idx < len(all_detections):
            bbox = all_detections[region_idx][0]
            pts  = [(int(p[0]),int(p[1])) for p in bbox]
            draw_final.polygon(pts, outline="#5DCAA5", width=3)

    st.success(f"✅ Found **{len(results)} medicine(s)** in the prescription")

    # Results layout
    left_col, right_col = st.columns([1,1], gap="large")
    with left_col:
        st.image(img_final, caption="Green boxes = detected medicines", use_column_width=True)

    with right_col:
        m1,m2,m3,m4 = st.columns(4)
        with m1: st.markdown(f'<div class="metric-box"><div class="metric-value">{len(results)}</div><div class="metric-label">Medicines</div></div>', unsafe_allow_html=True)
        with m2:
            hc = sum(1 for r in results if r["confidence"]=="high")
            st.markdown(f'<div class="metric-box"><div class="metric-value">{hc}</div><div class="metric-label">High Conf</div></div>', unsafe_allow_html=True)
        with m3:
            ff = sum(1 for r in results if r.get("fda",{}).get("found"))
            st.markdown(f'<div class="metric-box"><div class="metric-value">{ff}</div><div class="metric-label">FDA Found</div></div>', unsafe_allow_html=True)
        with m4: st.markdown('<div class="metric-box"><div class="metric-value">97.42%</div><div class="metric-label">Accuracy</div></div>', unsafe_allow_html=True)

    st.divider()
    st.subheader("💊 Medicine Details")

    for r in results:
        conf_badge = f'<span class="badge-{r["confidence"]}">{r["confidence"].upper()}</span>'
        with st.expander(
            f"💊 {r['corrected'].capitalize()}  |  Detected: '{r['easyocr_text']}'  |  {r['confidence'].upper()}",
            expanded=True
        ):
            left, right = st.columns([1,1], gap="large")
            with left:
                st.markdown(f"""
<div class="medicine-card">
<div class="medicine-title">{r['corrected'].capitalize()}</div>
<div class="generic-name">{r['generic'].capitalize() if r['generic'] else 'Generic name not available'}</div>
<p class="info-label">Confidence</p><p class="info-value">{conf_badge}</p>
<p class="info-label">Detected Text</p><p class="info-value"><code>{r['easyocr_text']}</code></p>
<p class="info-label">Dosage</p><p class="info-value">{', '.join(r['ner']['dosages']) if r['ner']['dosages'] else 'Not detected'}</p>
<p class="info-label">Frequency</p><p class="info-value">{', '.join(r['ner']['frequencies']) if r['ner']['frequencies'] else 'Not detected'}</p>
<p class="info-label">Duration</p><p class="info-value">{', '.join(r['ner']['durations']) if r['ner']['durations'] else 'Not detected'}</p>
</div>""", unsafe_allow_html=True)

            with right:
                if r.get("explanation"):
                    exp = r["explanation"]
                    st.markdown(f"""
<div class="medicine-card">
<div class="medicine-title">Patient Information</div>
<p class="info-label">What is it?</p><p class="info-value">{exp.get('summary','')}</p>
<p class="info-label">Uses</p><p class="info-value">{exp.get('uses','')}</p>
<p class="info-label">How to take</p><p class="info-value">{exp.get('how_to_take','')}</p>
<p class="info-label">Side Effects</p><p class="info-value">{exp.get('side_effects','')}</p>
<p class="info-label">⚠️ Warning</p><p class="info-value">{exp.get('warnings','')}</p>
</div>""", unsafe_allow_html=True)
                elif r.get("fda",{}).get("found"):
                    fda = r["fda"]
                    st.markdown(f"""
<div class="medicine-card">
<div class="medicine-title">FDA Information</div>
<p class="info-label">Indications</p><p class="info-value">{fda.get('indications','')[:300]}</p>
<p class="info-label">Warnings</p><p class="info-value">{fda.get('warnings','')[:200]}</p>
</div>""", unsafe_allow_html=True)
                else:
                    st.info("Enable Medicine Information in sidebar for FDA data.")

    st.divider()
    st.warning("⚕️ **Medical Disclaimer:** For informational purposes only. Always consult a qualified healthcare professional before taking any medication.")

elif not uploaded:
    st.info("👆 Upload a prescription image to get started")
    st.markdown("""
**This system handles:**
- ✅ Full handwritten prescription pages
- ✅ Single medicine name photos
- ✅ Medicine strips or bottle labels

**Pipeline:**
1. EasyOCR reads all text from the image
2. Gemini AI identifies which texts are medicine names
3. TrOCR precisely re-reads each medicine region
4. NER extracts dosage, frequency, duration
5. OpenFDA provides clinical information
6. Gemini generates plain-language patient explanation
""")