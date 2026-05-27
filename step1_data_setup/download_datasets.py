import os
import sys
import zipfile
import shutil
from pathlib import Path
 
# Add project root to path so we can import config
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    BD_DATASET_DIR,
    HF_DATASET_DIR,
    ILLEGIBLE_DATASET_DIR,
    RAW_DIR
)
 
 
# ── Helper: print progress clearly ───────────────────────────────
def log(msg):
    print(f"\n{'─'*55}\n  {msg}\n{'─'*55}")
 
 
# ─────────────────────────────────────────────────────────────────
# DATASET 1: BD Prescription Dataset from Kaggle
# URL: https://www.kaggle.com/datasets/mehaksingal/illegible-medical-prescription-images-dataset
# Contains: 4,680 segmented pharmaceutical word images
# ─────────────────────────────────────────────────────────────────
def download_bd_dataset():
    log("Downloading BD Prescription Dataset from Kaggle...")
 
    try:
        import kaggle
        # kaggle.api.authenticate() reads from ~/.kaggle/kaggle.json automatically
        kaggle.api.authenticate()
 
        # Download the dataset zip file into our raw directory
        kaggle.api.dataset_download_files(
            dataset="shawon10/doctors-handwritten-prescription-bd-dataset",
            path=str(BD_DATASET_DIR),
            unzip=True,          # Auto-unzip after download
            quiet=False          # Show progress bar
        )
        print("BD dataset downloaded successfully.")
        print(f"Location: {BD_DATASET_DIR}")
 
    except Exception as e:
        print(f"Kaggle download failed: {e}")
        print("\nManual download option:")
        print("1. Visit: https://www.kaggle.com/datasets/shawon10/doctors-handwritten-prescription-bd-dataset")
        print(f"2. Download and extract to: {BD_DATASET_DIR}")
 
 
# ─────────────────────────────────────────────────────────────────
# DATASET 2: HuggingFace Medical Prescription Words
# Source: avi-kai/medicine-words-handwritten on HuggingFace
# Contains: Additional handwritten medicine word images
# ─────────────────────────────────────────────────────────────────
def download_hf_dataset():
    log("Downloading HuggingFace Medical Words Dataset...")
 
    try:
        from datasets import load_dataset
 
        # Load the dataset from HuggingFace hub
        # This downloads and caches it automatically
        dataset = load_dataset(
            "avi-kai/medicine-words-handwritten",
            split="train",
            trust_remote_code=True
        )
 
        print(f"HuggingFace dataset loaded: {len(dataset)} samples")
 
        # Save images and labels locally in our standard format
        # This makes it compatible with our unified data loader later
        labels_file = HF_DATASET_DIR / "labels.csv"
        images_dir  = HF_DATASET_DIR / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
 
        import pandas as pd
        records = []
 
        for idx, sample in enumerate(dataset):
            # Each sample has 'image' (PIL Image) and 'label' (text)
            img_filename = f"hf_{idx:05d}.png"
            img_path     = images_dir / img_filename
 
            # Save the PIL image to disk
            sample["image"].save(img_path)
 
            records.append({
                "filename" : img_filename,
                "label"    : sample["label"],
                "source"   : "huggingface"
            })
 
        # Save all labels to a CSV for easy loading later
        df = pd.DataFrame(records)
        df.to_csv(labels_file, index=False)
 
        print(f"Saved {len(records)} HuggingFace images to: {images_dir}")
        print(f"Labels saved to: {labels_file}")
 
    except Exception as e:
        print(f"HuggingFace download failed: {e}")
        print("Try: pip install datasets")
 
 
# ─────────────────────────────────────────────────────────────────
# DATASET 3: Illegible Medical Prescription Images (Kaggle)
# URL: https://www.kaggle.com/datasets/mehaksingal/
#      illegible-medical-prescription-images-dataset
# Contains: Full prescription scan images (for EasyOCR testing)
# NOTE: We use these for END-TO-END testing only, not TrOCR training
# ─────────────────────────────────────────────────────────────────
def download_illegible_dataset():
    log("Downloading Illegible Prescription Images (for full-prescription testing)...")
 
    try:
        import kaggle
        kaggle.api.authenticate()
 
        kaggle.api.dataset_download_files(
            dataset="mehaksingal/illegible-medical-prescription-images-dataset",
            path=str(ILLEGIBLE_DATASET_DIR),
            unzip=True,
            quiet=False
        )
        print("Illegible prescription dataset downloaded successfully.")
        print(f"Location: {ILLEGIBLE_DATASET_DIR}")
 
    except Exception as e:
        print(f"Download failed: {e}")
        print("\nManual download:")
        print("1. Visit: https://www.kaggle.com/datasets/mehaksingal/illegible-medical-prescription-images-dataset")
        print(f"2. Extract to: {ILLEGIBLE_DATASET_DIR}")
 
 
# ─────────────────────────────────────────────────────────────────
# VERIFY: Check what was downloaded
# ─────────────────────────────────────────────────────────────────
def verify_downloads():
    log("Verifying downloaded datasets...")
 
    datasets_info = [
        ("BD Dataset",             BD_DATASET_DIR),
        ("HuggingFace Dataset",    HF_DATASET_DIR),
        ("Illegible Prescriptions",ILLEGIBLE_DATASET_DIR),
    ]
 
    for name, path in datasets_info:
        if path.exists():
            # Count all image files recursively
            images = list(path.rglob("*.png")) + \
                     list(path.rglob("*.jpg")) + \
                     list(path.rglob("*.jpeg"))
            print(f"  {name}: {len(images)} images found at {path}")
        else:
            print(f"  {name}: NOT FOUND at {path}")
 
 
# ── Main ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Starting dataset downloads...")
    print("This may take a few minutes depending on your connection.\n")
 
    download_bd_dataset()
    download_hf_dataset()
    download_illegible_dataset()
    verify_downloads()
 
    print("\nAll datasets ready. Next step: run preprocess.py")
 