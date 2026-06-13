"""
Breast Ultrasound Dataset - V2 Preparation Pipeline
====================================================
1. Stratified Split (70/15/15) with fixed seed
2. Multiple mask consolidation (Bitwise OR)
3. Resize all images + masks to 256x256
4. Advanced offline augmentation (only on TRAIN set)
   - Horizontal flip
   - Rotation with REFLECT padding (no black corners)
   - Random brightness / contrast adjustment
   - Gaussian noise injection
5. Class balancing via oversampling minority classes
"""

import os
import re
import shutil
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.model_selection import train_test_split
from collections import Counter

# ─── Configuration ───────────────────────────────────────────────
RANDOM_SEED = 42
TARGET_SIZE = (256, 256)  # (width, height)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

SOURCE_DIR = "/home/sefasys/Desktop/Breast_Ultrasound_Dataset/Dataset_BUSI_with_GT"
DEST_DIR = "/home/sefasys/Desktop/Breast_Ultrasound_Dataset/Dataset_BUSI_with_GT_Split"

CATEGORIES = ["benign", "malignant", "normal"]

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ─── Helper Functions ────────────────────────────────────────────

def combine_masks(mask_paths):
    """Combine multiple masks into one using bitwise OR."""
    if not mask_paths:
        return None
    combined = np.array(Image.open(mask_paths[0]).convert("L"))
    for path in mask_paths[1:]:
        mask = np.array(Image.open(path).convert("L"))
        combined = np.bitwise_or(combined, mask)
    return Image.fromarray(combined)


def resize_pair(img, mask, target_size):
    """Resize image (BICUBIC) and mask (NEAREST) to target size."""
    img_resized = img.resize(target_size, Image.BICUBIC)
    mask_resized = mask.resize(target_size, Image.NEAREST)
    return img_resized, mask_resized


def rotate_with_reflect(img, angle, resample):
    """
    Rotate an image by filling empty areas with reflected content
    instead of black pixels.
    """
    # Pad the image with reflected pixels before rotation,
    # then crop back to original size after rotation.
    w, h = img.size
    pad = max(w, h) // 4  # generous padding

    img_np = np.array(img)
    if img_np.ndim == 2:
        padded = np.pad(img_np, pad, mode="reflect")
    else:
        padded = np.pad(img_np, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")

    padded_img = Image.fromarray(padded)
    rotated = padded_img.rotate(angle, resample=resample, expand=False)

    # Crop back to center
    rw, rh = rotated.size
    cx, cy = rw // 2, rh // 2
    left = cx - w // 2
    top = cy - h // 2
    cropped = rotated.crop((left, top, left + w, top + h))
    return cropped


def augment_pair(img, mask):
    """
    Apply the EXACT same geometric transforms to both image and mask,
    plus photometric transforms to image only.
    Returns (augmented_img, augmented_mask).
    """
    # 1. Random Horizontal Flip (50% chance)
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

    # 2. Random Rotation with REFLECT padding (-15 to +15 degrees)
    angle = random.uniform(-15, 15)
    if abs(angle) > 1:  # skip trivial rotations
        img = rotate_with_reflect(img, angle, Image.BICUBIC)
        mask = rotate_with_reflect(mask, angle, Image.NEAREST)

    # 3. Random Brightness adjustment (image only, 0.8 - 1.2)
    brightness_factor = random.uniform(0.8, 1.2)
    img = ImageEnhance.Brightness(img).enhance(brightness_factor)

    # 4. Random Contrast adjustment (image only, 0.8 - 1.2)
    contrast_factor = random.uniform(0.8, 1.2)
    img = ImageEnhance.Contrast(img).enhance(contrast_factor)

    # 5. Gaussian Noise (image only, subtle)
    if random.random() < 0.5:
        img_np = np.array(img).astype(np.float32)
        noise = np.random.normal(0, 5, img_np.shape)  # std=5, subtle
        img_np = np.clip(img_np + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_np)

    return img, mask


# ─── Step 1: Collect all items from source ───────────────────────

def collect_items(source_dir):
    """Scan the reorganized dataset and collect all items."""
    dataset = {cat: [] for cat in CATEGORIES}

    for category in CATEGORIES:
        cat_path = os.path.join(source_dir, category)
        if not os.path.exists(cat_path):
            continue

        for folder_id in sorted(os.listdir(cat_path)):
            folder_path = os.path.join(cat_path, folder_id)
            if not os.path.isdir(folder_path):
                continue

            files = os.listdir(folder_path)
            masks = sorted([f for f in files if "_mask" in f])
            base_images = sorted(
                [f for f in files if "_mask" not in f and f.endswith(".png")]
            )

            if base_images:
                dataset[category].append(
                    {
                        "id": folder_id,
                        "category": category,
                        "base_img_path": os.path.join(folder_path, base_images[0]),
                        "mask_paths": [os.path.join(folder_path, m) for m in masks],
                    }
                )

    return dataset


# ─── Step 2: Stratified Split ────────────────────────────────────

def stratified_split(dataset):
    """Split each category into train/val/test with stratification."""
    splits = {"train": [], "val": [], "test": []}

    for category, items in dataset.items():
        if not items:
            continue

        # 70% train, 30% temp
        train_items, temp_items = train_test_split(
            items, test_size=(1 - TRAIN_RATIO), random_state=RANDOM_SEED
        )
        # 50% of 30% = 15% each
        val_items, test_items = train_test_split(
            temp_items, test_size=0.5, random_state=RANDOM_SEED
        )

        splits["train"].extend(train_items)
        splits["val"].extend(val_items)
        splits["test"].extend(test_items)

    return splits


# ─── Step 3: Process & Save ──────────────────────────────────────

def process_and_save(item, dest_folder):
    """
    Load image + masks, combine masks, resize to TARGET_SIZE, save.
    Returns the destination folder path.
    """
    category = item["category"]
    item_id = item["id"]
    target_dir = os.path.join(dest_folder, category, item_id)
    os.makedirs(target_dir, exist_ok=True)

    # Load base image
    img = Image.open(item["base_img_path"]).convert("RGB")

    # Combine masks
    if item["mask_paths"]:
        mask = combine_masks(item["mask_paths"])
    else:
        # Create an empty mask if none exists
        mask = Image.new("L", img.size, 0)

    # Resize
    img, mask = resize_pair(img, mask, TARGET_SIZE)

    # Save
    img.save(os.path.join(target_dir, "image.png"))
    mask.save(os.path.join(target_dir, "mask.png"))

    return target_dir


# ─── Step 4: Class Balancing via Augmentation ────────────────────

def balance_train_set(train_items, dest_dir):
    """
    Oversample minority classes in the train set using augmentation.
    """
    # Count per category
    cat_counts = Counter(item["category"] for item in train_items)
    max_count = max(cat_counts.values())

    print(f"\n  Train set class distribution BEFORE balancing:")
    for cat in CATEGORIES:
        print(f"    {cat}: {cat_counts.get(cat, 0)}")

    aug_counter = 0

    for category in CATEGORIES:
        current_count = cat_counts.get(category, 0)
        if current_count >= max_count:
            continue

        needed = max_count - current_count
        cat_items = [item for item in train_items if item["category"] == category]

        print(f"\n  Augmenting '{category}': generating {needed} new samples...")

        for i in range(needed):
            # Pick a random original item
            source_item = random.choice(cat_items)
            source_dir = os.path.join(
                dest_dir, "train", category, source_item["id"]
            )

            # Load the already-resized image and mask
            img = Image.open(os.path.join(source_dir, "image.png")).convert("RGB")
            mask = Image.open(os.path.join(source_dir, "mask.png")).convert("L")

            # Apply augmentation
            aug_img, aug_mask = augment_pair(img, mask)

            # Save to new folder
            aug_id = f"aug_{source_item['id']}_{i}"
            aug_dir = os.path.join(dest_dir, "train", category, aug_id)
            os.makedirs(aug_dir, exist_ok=True)

            aug_img.save(os.path.join(aug_dir, "image.png"))
            aug_mask.save(os.path.join(aug_dir, "mask.png"))
            aug_counter += 1

    print(f"\n  Total augmented samples created: {aug_counter}")


# ─── Main Pipeline ───────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Breast Ultrasound Dataset - V2 Pipeline")
    print("=" * 60)

    # Step 1: Collect
    print("\n[1/4] Scanning source dataset...")
    dataset = collect_items(SOURCE_DIR)
    for cat in CATEGORIES:
        print(f"  {cat}: {len(dataset[cat])} cases")

    # Step 2: Split
    print("\n[2/4] Performing stratified split (70/15/15)...")
    splits = stratified_split(dataset)
    for split_name, items in splits.items():
        cat_breakdown = Counter(item["category"] for item in items)
        print(f"  {split_name}: {len(items)} total -> {dict(cat_breakdown)}")

    # Step 3: Process (combine masks, resize, save)
    print("\n[3/4] Processing images (mask consolidation + resize to 256x256)...")
    for split_name, items in splits.items():
        dest_folder = os.path.join(DEST_DIR, split_name)
        for item in items:
            process_and_save(item, dest_folder)
        print(f"  {split_name}: {len(items)} items processed")

    # Step 4: Balance train set
    print("\n[4/4] Balancing train set via offline augmentation...")
    train_items = splits["train"]
    balance_train_set(train_items, DEST_DIR)

    # Final Summary
    print("\n" + "=" * 60)
    print("  FINAL DATASET SUMMARY")
    print("=" * 60)
    for split_name in ["train", "val", "test"]:
        split_path = os.path.join(DEST_DIR, split_name)
        print(f"\n  [{split_name.upper()}]")
        for cat in CATEGORIES:
            cat_path = os.path.join(split_path, cat)
            if os.path.exists(cat_path):
                total = len(os.listdir(cat_path))
                originals = len(
                    [f for f in os.listdir(cat_path) if not f.startswith("aug_")]
                )
                augmented = total - originals
                print(
                    f"    {cat}: {total} total "
                    f"({originals} original + {augmented} augmented)"
                )

    print(f"\n  Output: {DEST_DIR}")
    print("  All images: 256x256 PNG")
    print("  Naming: image.png + mask.png per folder")
    print("=" * 60)


if __name__ == "__main__":
    main()
