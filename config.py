 
import os
from pathlib import Path
 
# ── Project root ──────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent                   # prescription-ai/
 
# ── Data directories ─────────────────────────────────────────────
DATA_DIR        = ROOT_DIR / "data"
RAW_DIR         = DATA_DIR / "raw"                 # downloaded datasets go here
PROCESSED_DIR   = DATA_DIR / "processed"           # cleaned & resized images
AUGMENTED_DIR   = DATA_DIR / "augmented"           # augmented training images
OUTPUTS_DIR     = ROOT_DIR / "outputs"             # model outputs, plots, reports
 
# ── Dataset names (subfolders inside RAW_DIR) ────────────────────
BD_DATASET_DIR          = RAW_DIR / "bd_prescription"
HF_DATASET_DIR          = RAW_DIR / "hf_medical_words"
ILLEGIBLE_DATASET_DIR   = RAW_DIR / "illegible_prescription"
 
# ── Image preprocessing settings ─────────────────────────────────
IMG_HEIGHT  = 32          # TrOCR standard input height
IMG_WIDTH   = 128         # TrOCR standard input width
IMG_CHANNELS = 1          # Grayscale — color adds no value for HTR
 
# ── Train / Val / Test split ratios ──────────────────────────────
TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
TEST_RATIO  = 0.10        # Must sum to 1.0
 
# ── TrOCR model settings ─────────────────────────────────────────
# We start from Microsoft's handwriting model (already IAM pre-trained)
TROCR_BASE_MODEL    = "microsoft/trocr-base-handwritten"
TROCR_SAVE_PATH     = ROOT_DIR / "step2_trocr_finetune" / "saved_model"
TROCR_BATCH_SIZE    = 8       # Reduce to 4 if GPU memory is limited
TROCR_EPOCHS        = 30
TROCR_LR            = 5e-5    # Standard fine-tuning learning rate
TROCR_MAX_LENGTH    = 32      # Max characters in a medicine name
 
# ── Augmentation settings ─────────────────────────────────────────
AUG_MULTIPLIER = 3            # Each image → 3 augmented copies
                              # 8000 real → ~24,000 augmented samples
 
# ── EasyOCR settings (text detection for full prescriptions) ─────
EASYOCR_LANGUAGES   = ["en"]
EASYOCR_GPU         = True    # Set False if no GPU available
EASYOCR_CONF_THRESH = 0.3    # Min confidence to keep a detection
 
# ── OpenFDA API ───────────────────────────────────────────────────
OPENFDA_BASE_URL = "https://api.fda.gov/drug/label.json"
OPENFDA_LIMIT    = 1          # We only need top result per medicine
 
# ── LLM settings ─────────────────────────────────────────────────
# Uses Gemini free tier (same as your Week 4 GenAI learning path)
# Store your key in a .env file: GEMINI_API_KEY=your_key_here
LLM_MODEL = "gemini-1.5-flash"
 
# ── Evaluation metrics output path ───────────────────────────────
METRICS_OUTPUT = OUTPUTS_DIR / "evaluation_results.csv"
 
# ── Create all directories if they don't exist ───────────────────
for directory in [
    RAW_DIR, PROCESSED_DIR, AUGMENTED_DIR, OUTPUTS_DIR,
    BD_DATASET_DIR, HF_DATASET_DIR, ILLEGIBLE_DATASET_DIR,
    TROCR_SAVE_PATH
]:
    directory.mkdir(parents=True, exist_ok=True)
 
print("Config loaded. Project root:", ROOT_DIR)
 