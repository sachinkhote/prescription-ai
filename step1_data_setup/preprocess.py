# step1_data_setup/preprocess.py
# ─────────────────────────────────────────────────────────────────
# Reads the BD dataset using its actual CSV label files,
# cleans all images, applies augmentation, and saves
# ready-to-use train/val/test CSVs for TrOCR training.
#
# BD Dataset structure (what we actually have):
#   Training/training_words/0.png ...
#   Training/training_labels.csv  → IMAGE, MEDICINE_NAME, GENERIC_NAME
#   Testing/testing_words/...
#   Validation/validation_words/...
#
# HOW TO RUN:
#   python step1_data_setup/preprocess.py
# ─────────────────────────────────────────────────────────────────

import sys
import cv2
import numpy as np
import pandas as pd
import albumentations as A
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))
from config import (
    BD_DATASET_DIR,
    PROCESSED_DIR,
    AUGMENTED_DIR,
    IMG_HEIGHT,
    IMG_WIDTH,
    AUG_MULTIPLIER
)


# ─────────────────────────────────────────────────────────────────
# STEP 1: READ all three splits from BD dataset CSV files
# ─────────────────────────────────────────────────────────────────
def load_bd_dataset():
    """
    Reads Training/Validation/Testing CSVs and builds
    a unified DataFrame with full image paths and labels.
    """
    print("\n[1/4] Loading BD dataset from CSV files...")

    # Auto-detect the dataset root folder
    # (avoids hardcoding the apostrophe in "Doctor's")
    bd_root = list(BD_DATASET_DIR.iterdir())[0]
    print(f"  Dataset root: {bd_root.name}")

    split_map = {
        "train" : ("Training",   "training_words",   "training_labels.csv"),
        "val"   : ("Validation", "validation_words", "validation_labels.csv"),
        "test"  : ("Testing",    "testing_words",    "testing_labels.csv"),
    }

    all_records = []

    for split_name, (folder, img_subfolder, csv_file) in split_map.items():
        csv_path  = bd_root / folder / csv_file
        imgs_path = bd_root / folder / img_subfolder

        if not csv_path.exists():
            print(f"  WARNING: {csv_path} not found — skipping")
            continue

        df = pd.read_csv(csv_path)

        # Columns: IMAGE, MEDICINE_NAME, GENERIC_NAME
        for _, row in df.iterrows():
            img_full_path = imgs_path / row["IMAGE"]
            if img_full_path.exists():
                all_records.append({
                    "filepath"     : str(img_full_path),
                    "label"        : str(row["MEDICINE_NAME"]).strip().lower(),
                    "generic_name" : str(row["GENERIC_NAME"]).strip().lower(),
                    "split"        : split_name,
                    "source"       : "bd_dataset"
                })

        print(f"  {split_name:5s} → {len(df)} rows in CSV, "
              f"{sum(1 for r in all_records if r['split']==split_name)} images found")

    df_all = pd.DataFrame(all_records)
    print(f"\n  Total loaded : {len(df_all)} images")
    print(f"  Unique medicines (brand)  : {df_all['label'].nunique()}")
    print(f"  Unique medicines (generic): {df_all['generic_name'].nunique()}")

    return df_all


# ─────────────────────────────────────────────────────────────────
# STEP 2: CLEAN — resize, grayscale, binarize each image
# ─────────────────────────────────────────────────────────────────
def clean_image(img_path: str):
    """
    Applies the standard cleaning pipeline to one image:
      1. Read as grayscale
      2. Resize to IMG_HEIGHT x IMG_WIDTH (32x128)
      3. OTSU binarization — sharpens handwriting strokes
    Returns numpy array or None if image can't be loaded.
    """
    # Use np.fromfile + imdecode to avoid Windows Unicode path issues
    try:
        img_array = np.fromfile(img_path, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
    except Exception:
        img = None
    if img is None:
        return None

    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT),
                     interpolation=cv2.INTER_AREA)

    # OTSU threshold — automatically finds best binarization level
    _, img = cv2.threshold(img, 0, 255,
                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return img


def preprocess_all_images(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans all images and saves them to PROCESSED_DIR/images/.
    Adds 'processed_path' column to the DataFrame.
    """
    print("\n[2/4] Cleaning and resizing all images...")

    out_dir = PROCESSED_DIR / "images"
    out_dir.mkdir(parents=True, exist_ok=True)

    processed_paths = []
    failed = 0

    for idx, row in tqdm(df.iterrows(), total=len(df),
                         desc="  Preprocessing"):
        img = clean_image(row["filepath"])

        if img is None:
            processed_paths.append(None)
            failed += 1
            continue

        out_path = out_dir / f"img_{idx:06d}.png"
        cv2.imwrite(str(out_path), img)
        processed_paths.append(str(out_path))

    df = df.copy()
    df["processed_path"] = processed_paths
    df_clean = df[df["processed_path"].notna()].copy()

    print(f"  Processed : {len(df_clean)} images")
    print(f"  Failed    : {failed} images")

    return df_clean


# ─────────────────────────────────────────────────────────────────
# STEP 3: AUGMENT — only the training split
# ─────────────────────────────────────────────────────────────────
augmentation_pipeline = A.Compose([
    A.Rotate(limit=8, p=0.7),
    A.Affine(shear=(-10, 10), p=0.5),
    A.RandomBrightnessContrast(
        brightness_limit=0.2,
        contrast_limit=0.2,
        p=0.6
    ),
    A.GaussNoise(var_limit=(10, 30), p=0.4),
    A.GaussianBlur(blur_limit=(1, 3), p=0.3),
    A.ElasticTransform(alpha=0.5, sigma=10, p=0.4),
])


def augment_training_data(df_train: pd.DataFrame) -> pd.DataFrame:
    """
    Generates AUG_MULTIPLIER augmented copies of each
    training image. Only training data is augmented —
    val and test stay clean for fair evaluation.
    """
    print(f"\n[3/4] Augmenting training data (×{AUG_MULTIPLIER})...")

    aug_dir = AUGMENTED_DIR / "images"
    aug_dir.mkdir(parents=True, exist_ok=True)

    aug_records = []

    for idx, row in tqdm(df_train.iterrows(), total=len(df_train),
                         desc="  Augmenting"):
        try:
            img_array = np.fromfile(row["processed_path"], dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
        except Exception:
            img = None
        if img is None:
            continue

        for aug_idx in range(AUG_MULTIPLIER):
            img_rgb   = np.stack([img] * 3, axis=-1)
            augmented = augmentation_pipeline(image=img_rgb)
            aug_img   = augmented["image"][:, :, 0]

            aug_filename = f"aug_{idx:06d}_{aug_idx}.png"
            aug_path     = aug_dir / aug_filename
            cv2.imwrite(str(aug_path), aug_img)

            aug_records.append({
                "filepath"       : str(aug_path),
                "label"          : row["label"],
                "generic_name"   : row["generic_name"],
                "split"          : "train",
                "source"         : "bd_dataset_aug",
                "processed_path" : str(aug_path)
            })

    df_aug = pd.DataFrame(aug_records)
    print(f"  Original train samples  : {len(df_train)}")
    print(f"  Augmented samples added : {len(df_aug)}")
    print(f"  Total training pool     : {len(df_train) + len(df_aug)}")

    return df_aug


# ─────────────────────────────────────────────────────────────────
# STEP 4: SAVE final split CSVs
# ─────────────────────────────────────────────────────────────────
def save_splits(df_clean: pd.DataFrame, df_aug: pd.DataFrame):
    """
    Saves three CSV files that TrOCR training will read directly:
      processed/train.csv  — original train + augmented
      processed/val.csv    — original validation only
      processed/test.csv   — original test only
    """
    print("\n[4/4] Saving final split CSVs...")

    df_train_orig = df_clean[df_clean["split"] == "train"].copy()
    df_val        = df_clean[df_clean["split"] == "val"].copy()
    df_test       = df_clean[df_clean["split"] == "test"].copy()

    # Combine original training with augmented copies
    df_train_full = pd.concat(
        [df_train_orig, df_aug], ignore_index=True
    ).sample(frac=1, random_state=42)   # shuffle

    # Save all three splits
    df_train_full.to_csv(PROCESSED_DIR / "train.csv", index=False)
    df_val.to_csv        (PROCESSED_DIR / "val.csv",   index=False)
    df_test.to_csv       (PROCESSED_DIR / "test.csv",  index=False)

    print(f"  train.csv : {len(df_train_full):,} samples "
          f"({len(df_train_orig):,} original + "
          f"{len(df_aug):,} augmented)")
    print(f"  val.csv   : {len(df_val):,} samples")
    print(f"  test.csv  : {len(df_test):,} samples")
    print(f"\n  Saved to: {PROCESSED_DIR}")


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  STEP 1 — Data Preprocessing Pipeline")
    print("=" * 55)

    # 1. Load dataset from CSVs
    df_all = load_bd_dataset()

    # 2. Clean all images
    df_clean = preprocess_all_images(df_all)

    # 3. Augment training split only
    df_train = df_clean[df_clean["split"] == "train"].copy()
    df_aug   = augment_training_data(df_train)

    # 4. Save final CSVs
    save_splits(df_clean, df_aug)

    print("\n" + "=" * 55)
    print("  Step 1 COMPLETE")
    print("  Next → run: python step1_data_setup/verify_data.py")
    print("=" * 55)