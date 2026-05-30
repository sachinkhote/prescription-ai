# Prescription Medicine Extraction and Intelligence Using TrOCR and OpenFDA

**MCA Major Project — Amity University Online (2024–2026)**
**Student:** Sachin Shrimant Khote | **Enrollment:** A9929724001448

---

## What this system does

Upload a handwritten medical prescription — full page or single medicine photo — and the system will:
1. Detect all handwritten text regions automatically
2. Transcribe each region using a fine-tuned TrOCR model
3. Correct OCR noise using an LLM post-processor
4. Extract structured entities: medicine name, dosage, frequency
5. Return medicine information: usage, side effects, properties

---

## Project structure

```
prescription-ai/
├── config.py                        # All settings in one place
├── requirements.txt
│
├── step1_data_setup/
│   ├── download_datasets.py         # Downloads BD + HF + Illegible datasets
│   ├── preprocess.py                # Clean, resize, augment all images
│   └── verify_data.py               # Visual verification + report screenshots
│
├── step2_trocr_finetune/
│   ├── dataset.py                   # PyTorch Dataset class for TrOCR
│   ├── train.py                     # Fine-tuning loop
│   └── evaluate.py                  # WER + Character Accuracy metrics
│
├── step3_llm_postprocessor/
│   └── corrector.py                 # LLM-based OCR spelling correction
│
├── step4_ner_pipeline/
│   └── ner.py                       # scispaCy entity extraction
│
├── step5_medicine_info/
│   ├── openfda.py                   # OpenFDA API lookup
│   └── explainer.py                 # LLM medicine explanation
│
├── app/
│   └── streamlit_app.py             # End-to-end Streamlit UI
│
├── data/
│   ├── raw/                         # Downloaded datasets
│   ├── processed/                   # Cleaned images + split CSVs
│   └── augmented/                   # Augmented training images
│
└── outputs/                         # Model checkpoints, plots, results
```

---

## How to run (step by step)

```bash
# 1. Clone and install
git clone https://github.com/sachinkhote/prescription-ai
cd prescription-ai
pip install -r requirements.txt

# 2. Set up Kaggle API (for dataset download)
# Place kaggle.json in ~/.kaggle/

# 3. Download datasets
python step1_data_setup/download_datasets.py

# 4. Preprocess and augment
python step1_data_setup/preprocess.py

# 5. Verify data looks correct
python step1_data_setup/verify_data.py

# 6. Fine-tune TrOCR
python step2_trocr_finetune/train.py

# 7. Run evaluation
python step2_trocr_finetune/evaluate.py

# 8. Launch the full app
streamlit run app/streamlit_app.py
```

---

## Dataset used

| Dataset | Size | Purpose |
|---|---|---|
| BD Prescription (Kaggle) | ~4,680 images | TrOCR fine-tuning (base) |
| HuggingFace Medical Words | ~2,000+ images | Vocabulary diversity |
| After augmentation (×3) | ~20,000+ images | Full training pool |
| Illegible Prescriptions (Kaggle) | Full scans | End-to-end testing |

---

## Model architecture

```
Input image (full prescription OR single medicine photo)
    ↓
EasyOCR text detector  →  bounding boxes of text regions
    ↓
TrOCR (fine-tuned)     →  raw transcribed text per region
    ↓
LLM post-processor     →  corrected medicine names
    ↓
scispaCy NER           →  structured: {name, dosage, frequency}
    ↓
OpenFDA API + LLM      →  medicine usage, side effects, properties
    ↓
Streamlit UI           →  structured output card
```

---

## Results (to be filled after training)

| Metric | CRNN (Minor Project) | TrOCR (Major Project) |
|---|---|---|
| Raw Character Accuracy | 21.28% | — |
| Corrected Character Accuracy | 59.27% | — |
| Word Error Rate | — | — |
| NER F1 Score | N/A | — |