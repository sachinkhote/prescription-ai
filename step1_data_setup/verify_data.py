import sys
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
 
sys.path.append(str(Path(__file__).parent.parent))
from config import PROCESSED_DIR, AUGMENTED_DIR, OUTPUTS_DIR
 
 
def verify_dataset():
    print("=" * 55)
    print("  Dataset Verification Report")
    print("=" * 55)
 
    # ── Load split CSVs ───────────────────────────────────────────
    splits = {}
    for split_name in ["train", "val", "test"]:
        csv_path = PROCESSED_DIR / f"{split_name}.csv"
        if csv_path.exists():
            splits[split_name] = pd.read_csv(csv_path)
            print(f"\n  {split_name.upper()} SET: {len(splits[split_name]):,} samples")
        else:
            print(f"\n  {split_name}.csv not found. Run preprocess.py first.")
            return
 
    total = sum(len(v) for v in splits.values())
    print(f"\n  TOTAL SAMPLES : {total:,}")
 
    # ── Label statistics ──────────────────────────────────────────
    train_df = splits["train"]
    print(f"\n  Unique medicine names in train: {train_df['label'].nunique()}")
    print(f"\n  Top 10 most common labels:")
    print(train_df['label'].value_counts().head(10).to_string())
 
    # ── Source distribution ───────────────────────────────────────
    print(f"\n  Samples by source:")
    print(train_df['source'].value_counts().to_string())
 
    # ── Visual verification ───────────────────────────────────────
    # Show a grid of sample images with their labels
    print("\n  Generating sample visualization...")
 
    test_df = splits["test"]
    samples = test_df.sample(min(12, len(test_df)), random_state=42)
 
    fig = plt.figure(figsize=(14, 8))
    fig.suptitle(
        "Dataset Sample Verification — Processed Images",
        fontsize=14, fontweight="bold", y=1.01
    )
 
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.5, wspace=0.3)
 
    for i, (_, row) in enumerate(samples.iterrows()):
        if i >= 12:
            break
 
        ax = fig.add_subplot(gs[i // 4, i % 4])
 
        # Load and display the processed image
        img_path = row.get("processed_path") or row.get("filepath")
        img      = cv2.imread(str(img_path), 0)
 
        if img is not None:
            ax.imshow(img, cmap="gray", aspect="auto")
 
        ax.set_title(
            row["label"].capitalize(),
            fontsize=9,
            fontweight="bold",
            color="#333333",
            pad=3
        )
        ax.axis("off")
 
    plt.tight_layout()
    out_path = OUTPUTS_DIR / "step1_sample_verification.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved to: {out_path}")
 
    # ── Augmentation comparison ───────────────────────────────────
    # Show original vs augmented versions of the same image
    print("\n  Generating augmentation comparison...")
 
    orig_sample = splits["test"].iloc[0]
    orig_img    = cv2.imread(orig_sample.get("processed_path") or
                             orig_sample["filepath"], 0)
 
    if orig_img is not None:
        # Load up to 5 augmented versions of any image for comparison
        aug_csv = AUGMENTED_DIR / "augmented_labels.csv"
        if aug_csv.exists():
            aug_df  = pd.read_csv(aug_csv)
            samples_aug = aug_df[
                aug_df["label"] == orig_sample["label"]
            ].head(5)
 
            cols    = 1 + len(samples_aug)
            fig, axes = plt.subplots(1, cols, figsize=(cols * 2.5, 2.5))
 
            axes[0].imshow(orig_img, cmap="gray")
            axes[0].set_title("Original", fontsize=9, fontweight="bold")
            axes[0].axis("off")
 
            for j, (_, aug_row) in enumerate(samples_aug.iterrows()):
                aug_img = cv2.imread(aug_row["filepath"], 0)
                if aug_img is not None:
                    axes[j + 1].imshow(aug_img, cmap="gray")
                    axes[j + 1].set_title(f"Aug {j+1}", fontsize=9)
                    axes[j + 1].axis("off")
 
            plt.suptitle(
                f"Original vs Augmented — '{orig_sample['label'].capitalize()}'",
                fontsize=11, fontweight="bold"
            )
            plt.tight_layout()
            aug_out = OUTPUTS_DIR / "step1_augmentation_comparison.png"
            plt.savefig(aug_out, dpi=150, bbox_inches="tight")
            plt.show()
            print(f"  Augmentation comparison saved to: {aug_out}")
 
    # ── Label distribution chart ──────────────────────────────────
    print("\n  Generating label distribution chart...")
 
    fig, ax = plt.subplots(figsize=(12, 4))
    top_labels = train_df["label"].value_counts().head(20)
    ax.bar(
        range(len(top_labels)),
        top_labels.values,
        color="#5A7DD4",
        edgecolor="white",
        linewidth=0.5
    )
    ax.set_xticks(range(len(top_labels)))
    ax.set_xticklabels(
        [l.capitalize() for l in top_labels.index],
        rotation=45, ha="right", fontsize=8
    )
    ax.set_title(
        "Top 20 Medicine Labels — Training Set Distribution",
        fontsize=12, fontweight="bold"
    )
    ax.set_ylabel("Sample Count")
    ax.grid(axis="y", alpha=0.3)
 
    plt.tight_layout()
    dist_out = OUTPUTS_DIR / "step1_label_distribution.png"
    plt.savefig(dist_out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Distribution chart saved to: {dist_out}")
 
    print("\n" + "=" * 55)
    print("  Verification COMPLETE")
    print("  All output images are in:", OUTPUTS_DIR)
    print("  Use these screenshots in your project report.")
    print("=" * 55)
 
 
if __name__ == "__main__":
    verify_dataset()