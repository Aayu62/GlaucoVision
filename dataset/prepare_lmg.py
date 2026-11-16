"""
Dataset Preparation Script for CA-Net
Reconstructs the Large Multistage Glaucoma (LMG) dataset and Harvard Dataverse V1 dataset.
Produces stratified train/val/test CSV splits following the paper's exact 70:30 (Train:Test)
and 70:30 (Train:Val within Train) protocol.
"""

import os
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

HARVARD_BASE = "c:/Users/KIIT0001/GlaucoVision/processed_data(Harvard)"
RIMONE_BASE = "c:/Users/KIIT0001/GlaucoVision/RIM-ONE_database_r1"
OUTPUT_DIR = "c:/Users/KIIT0001/GlaucoVision/data_splits"

# Class mappings: 0 -> Normal, 1 -> Early, 2 -> Advanced
CLASS_NAMES = ["Normal", "Early-stage", "Advanced-stage"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}

def collect_harvard_images():
    records = []
    # Advanced
    adv_dir = os.path.join(HARVARD_BASE, "advanced_glaucoma")
    if os.path.exists(adv_dir):
        files = sorted([f for f in os.listdir(adv_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        # Paper uses 467 images
        for f in files[:467]:
            records.append({
                "image_path": os.path.join(adv_dir, f).replace("\\", "/"),
                "label": 2,
                "class_name": "Advanced-stage",
                "source": "Harvard"
            })
            
    # Early
    early_dir = os.path.join(HARVARD_BASE, "early_glaucoma")
    if os.path.exists(early_dir):
        files = sorted([f for f in os.listdir(early_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        # Paper uses 289 images
        for f in files[:289]:
            records.append({
                "image_path": os.path.join(early_dir, f).replace("\\", "/"),
                "label": 1,
                "class_name": "Early-stage",
                "source": "Harvard"
            })

    # Normal
    norm_dir = os.path.join(HARVARD_BASE, "normal_control")
    if os.path.exists(norm_dir):
        files = sorted([f for f in os.listdir(norm_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        # Paper uses 786 images
        for f in files[:786]:
            records.append({
                "image_path": os.path.join(norm_dir, f).replace("\\", "/"),
                "label": 0,
                "class_name": "Normal",
                "source": "Harvard"
            })

    return records

def collect_rimone_images():
    records = []
    # In RIM-ONE r1, original images follow 'ImXXX.bmp' without '-exp' or '-gs'
    pattern = re.compile(r"^im\d+\.bmp$", re.IGNORECASE)

    # Early (12 images in RIM-ONE r1)
    early_dir = os.path.join(RIMONE_BASE, "Early")
    if os.path.exists(early_dir):
        files = sorted([f for f in os.listdir(early_dir) if pattern.match(f)])
        for f in files[:12]:
            records.append({
                "image_path": os.path.join(early_dir, f).replace("\\", "/"),
                "label": 1,
                "class_name": "Early-stage",
                "source": "RIM-ONE-r1"
            })

    # Deep (14 images in RIM-ONE r1 -> Advanced-stage)
    deep_dir = os.path.join(RIMONE_BASE, "Deep")
    if os.path.exists(deep_dir):
        files = sorted([f for f in os.listdir(deep_dir) if pattern.match(f)])
        for f in files[:14]:
            records.append({
                "image_path": os.path.join(deep_dir, f).replace("\\", "/"),
                "label": 2,
                "class_name": "Advanced-stage",
                "source": "RIM-ONE-r1"
            })

    # Normal (sample 14 images to get 40 total RIM-ONE r1 images)
    norm_dir = os.path.join(RIMONE_BASE, "Normal")
    if os.path.exists(norm_dir):
        files = sorted([f for f in os.listdir(norm_dir) if pattern.match(f)])
        for f in files[:14]:
            records.append({
                "image_path": os.path.join(norm_dir, f).replace("\\", "/"),
                "label": 0,
                "class_name": "Normal",
                "source": "RIM-ONE-r1"
            })

    return records

def make_stratified_splits(df, random_state=42):
    # 70% Train, 30% Test
    train_val_df, test_df = train_test_split(
        df, test_size=0.30, random_state=random_state, stratify=df["label"]
    )
    # Inside the 70% Train set, 70% Train and 30% Validation
    train_df, val_df = train_test_split(
        train_val_df, test_size=0.30, random_state=random_state, stratify=train_val_df["label"]
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Collecting Harvard images...")
    harvard_records = collect_harvard_images()
    harvard_df = pd.DataFrame(harvard_records)
    print(f"Harvard total: {len(harvard_df)}")
    print(harvard_df["class_name"].value_counts())

    print("\nCollecting RIM-ONE r1 images...")
    rimone_records = collect_rimone_images()
    rimone_df = pd.DataFrame(rimone_records)
    print(f"RIM-ONE r1 total: {len(rimone_df)}")
    print(rimone_df["class_name"].value_counts())

    # Build LMG Dataset (1582 images)
    lmg_df = pd.concat([harvard_df, rimone_df], ignore_index=True)
    print("\n=== LMG Dataset Summary ===")
    print(f"Total images: {len(lmg_df)}")
    print(lmg_df["class_name"].value_counts())
    print("\nBy Source and Class:")
    print(pd.crosstab(lmg_df["source"], lmg_df["class_name"], margins=True))

    # Split LMG
    lmg_train, lmg_val, lmg_test = make_stratified_splits(lmg_df)
    print(f"\nLMG Splits -> Train: {len(lmg_train)}, Val: {len(lmg_val)}, Test: {len(lmg_test)}")

    lmg_df.to_csv(os.path.join(OUTPUT_DIR, "lmg_all.csv"), index=False)
    lmg_train.to_csv(os.path.join(OUTPUT_DIR, "lmg_train.csv"), index=False)
    lmg_val.to_csv(os.path.join(OUTPUT_DIR, "lmg_val.csv"), index=False)
    lmg_test.to_csv(os.path.join(OUTPUT_DIR, "lmg_test.csv"), index=False)

    # Split Harvard standalone as well (1542 images)
    h_train, h_val, h_test = make_stratified_splits(harvard_df)
    harvard_df.to_csv(os.path.join(OUTPUT_DIR, "harvard_all.csv"), index=False)
    h_train.to_csv(os.path.join(OUTPUT_DIR, "harvard_train.csv"), index=False)
    h_val.to_csv(os.path.join(OUTPUT_DIR, "harvard_val.csv"), index=False)
    h_test.to_csv(os.path.join(OUTPUT_DIR, "harvard_test.csv"), index=False)

    print(f"\nSuccessfully generated dataset manifests in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
