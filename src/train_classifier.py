"""
Breast Ultrasound Classification — Google Colab Training Script
================================================================
Model: EfficientNet-B0 (ImageNet pretrained) + Transfer Learning
Strategy: 2-Stage Fine-Tuning with aggressive anti-overfitting

Usage (Colab):
  1. Upload Dataset_BUSI_with_GT_Split to Google Drive
  2. Mount Drive
  3. Run this script

Author: AI Assistant
"""

# ─── 1. IMPORTS & SETUP ──────────────────────────────────────────

import os
import copy
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# ─── 2. CONFIGURATION ────────────────────────────────────────────

class Config:
    """All hyperparameters in one place for easy tuning."""

    # ── Paths ──
    # Assuming script is run from src/ directory or similar
    DATASET_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Dataset_BUSI_with_GT_Split"))
    
    # ── Output ──
    OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "classification"))

    # ── Model ──
    MODEL_NAME = "efficientnet_b0"
    NUM_CLASSES = 3
    CLASS_NAMES = ["benign", "malignant", "normal"]

    # ── Image ──
    IMG_SIZE = 256
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    # ── Training — Stage 1: Only classifier head ──
    STAGE1_EPOCHS = 15
    STAGE1_LR = 1e-3
    STAGE1_BATCH_SIZE = 32

    # ── Training — Stage 2: Fine-tune backbone ──
    STAGE2_EPOCHS = 30
    STAGE2_LR = 1e-4           # 10x lower for fine-tuning
    STAGE2_BATCH_SIZE = 32
    STAGE2_UNFREEZE_FROM = 6   # Unfreeze from this block onward (0-7 for EfficientNet-B0)

    # ── Regularization ──
    WEIGHT_DECAY = 1e-4
    DROPOUT = 0.4
    LABEL_SMOOTHING = 0.1

    # ── Early Stopping ──
    PATIENCE = 8               # Stop if val loss doesn't improve for N epochs

    # ── Reproducibility ──
    SEED = 42

    # ── Device ──
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─── 3. DATASET ──────────────────────────────────────────────────

class BreastUltrasoundDataset(Dataset):
    """
    Loads images from the folder structure:
      split/class_name/sample_id/image.png
    """

    def __init__(self, root_dir, split, transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.class_names = Config.CLASS_NAMES
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}

        self.samples = []  # list of (image_path, label_idx)

        split_dir = os.path.join(root_dir, split)
        for class_name in self.class_names:
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.exists(class_dir):
                continue
            for sample_id in os.listdir(class_dir):
                img_path = os.path.join(class_dir, sample_id, "image.png")
                if os.path.exists(img_path):
                    self.samples.append((img_path, self.class_to_idx[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_class_distribution(self):
        labels = [label for _, label in self.samples]
        return Counter(labels)


# ─── 4. DATA TRANSFORMS (Online Augmentation) ────────────────────

def get_transforms():
    """
    Online augmentation for TRAIN set.
    Val/Test only get resize + normalize.
    """
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(
            degrees=15,
            interpolation=transforms.InterpolationMode.BILINEAR,
            fill=128,  # gray fill instead of black
        ),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.1,
            hue=0.05,
        ),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.05, 0.05),  # slight shift
            scale=(0.95, 1.05),      # slight zoom
        ),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(Config.IMAGENET_MEAN, Config.IMAGENET_STD),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),  # cutout-like
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(Config.IMAGENET_MEAN, Config.IMAGENET_STD),
    ])

    return train_transform, val_transform


# ─── 5. MODEL ────────────────────────────────────────────────────

def build_model():
    """
    EfficientNet-B0 with pretrained ImageNet weights.
    Replace the classifier head for 3-class output.
    """
    # Load pretrained model
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
    model = models.efficientnet_b0(weights=weights)

    # Freeze all layers initially
    for param in model.parameters():
        param.requires_grad = False

    # Replace classifier head with custom head
    in_features = model.classifier[1].in_features  # 1280 for B0
    model.classifier = nn.Sequential(
        nn.Dropout(p=Config.DROPOUT),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.BatchNorm1d(256),
        nn.Dropout(p=Config.DROPOUT * 0.5),
        nn.Linear(256, Config.NUM_CLASSES),
    )

    return model.to(Config.DEVICE)


def unfreeze_backbone(model, from_block=6):
    """
    Unfreeze the last N blocks of EfficientNet for fine-tuning.
    EfficientNet-B0 has 8 blocks (0-7) in model.features.
    """
    # Unfreeze from specified block onward
    for i, block in enumerate(model.features):
        if i >= from_block:
            for param in block.parameters():
                param.requires_grad = True

    # Also unfreeze batch norm in unfrozen layers (important!)
    return model


# ─── 6. TRAINING LOOP ────────────────────────────────────────────

class EarlyStopping:
    """Stop training when validation loss stops improving."""

    def __init__(self, patience=8, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch and return (loss, accuracy)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """Evaluate on val/test set and return (loss, accuracy)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def train_stage(
    model, train_loader, val_loader, criterion, optimizer, scheduler,
    num_epochs, device, stage_name, early_stopping
):
    """
    Train a stage (head-only or fine-tune) and return history.
    """
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
    }
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0

    print(f"\n{'=' * 50}")
    print(f"  {stage_name}")
    print(f"{'=' * 50}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    print(f"  Param/Sample ratio: {trainable/len(train_loader.dataset):.1f}x\n")

    for epoch in range(num_epochs):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        if scheduler:
            scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - t0

        # Track best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            marker = " ★ best"
        else:
            marker = ""

        print(
            f"  Epoch {epoch+1:02d}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
            f"{elapsed:.1f}s{marker}"
        )

        # Early stopping check
        if early_stopping(val_loss):
            print(f"\n  ⚠ Early stopping triggered at epoch {epoch+1}")
            break

    # Restore best weights
    model.load_state_dict(best_model_wts)
    print(f"\n  Best Val Accuracy: {best_val_acc:.4f}")
    return model, history


# ─── 7. EVALUATION & VISUALIZATION ───────────────────────────────

@torch.no_grad()
def get_predictions(model, dataloader, device):
    """Get all predictions and true labels."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    for images, labels in dataloader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def plot_training_history(history_stage1, history_stage2, save_path):
    """Plot training curves for both stages."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Training History", fontsize=16, fontweight="bold")

    stages = [
        (history_stage1, "Stage 1: Head Only"),
        (history_stage2, "Stage 2: Fine-Tune"),
    ]

    for col, (history, title) in enumerate(stages):
        epochs = range(1, len(history["train_loss"]) + 1)

        # Loss
        axes[0, col].plot(epochs, history["train_loss"], "b-o", label="Train", markersize=3)
        axes[0, col].plot(epochs, history["val_loss"], "r-o", label="Val", markersize=3)
        axes[0, col].set_title(f"{title} — Loss")
        axes[0, col].set_xlabel("Epoch")
        axes[0, col].set_ylabel("Loss")
        axes[0, col].legend()
        axes[0, col].grid(True, alpha=0.3)

        # Accuracy
        axes[1, col].plot(epochs, history["train_acc"], "b-o", label="Train", markersize=3)
        axes[1, col].plot(epochs, history["val_acc"], "r-o", label="Val", markersize=3)
        axes[1, col].set_title(f"{title} — Accuracy")
        axes[1, col].set_xlabel("Epoch")
        axes[1, col].set_ylabel("Accuracy")
        axes[1, col].legend()
        axes[1, col].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "training_history.png"), dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {save_path}/training_history.png")


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Raw counts
    disp1 = ConfusionMatrixDisplay(cm, display_labels=class_names)
    disp1.plot(ax=axes[0], cmap="Blues", values_format="d")
    axes[0].set_title("Confusion Matrix (Counts)")

    # Normalized
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    disp2 = ConfusionMatrixDisplay(cm_norm, display_labels=class_names)
    disp2.plot(ax=axes[1], cmap="Blues", values_format=".2f")
    axes[1].set_title("Confusion Matrix (Normalized)")

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {save_path}/confusion_matrix.png")


def plot_per_class_metrics(y_true, y_pred, class_names, save_path):
    """Plot per-class precision, recall, f1."""
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

    metrics = ["precision", "recall", "f1-score"]
    data = {m: [report[c][m] for c in class_names] for m in metrics}

    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, metric in enumerate(metrics):
        bars = ax.bar(x + i * width, data[metric], width, label=metric.capitalize())
        for bar, val in zip(bars, data[metric]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Score")
    ax.set_title("Per-Class Metrics")
    ax.set_xticks(x + width)
    ax.set_xticklabels(class_names)
    ax.legend()
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "per_class_metrics.png"), dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {save_path}/per_class_metrics.png")


# ─── 8. MAIN PIPELINE ────────────────────────────────────────────

def main():
    set_seed(Config.SEED)
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("  BREAST ULTRASOUND CLASSIFICATION")
    print("  EfficientNet-B0 + Transfer Learning")
    print("=" * 60)
    print(f"\n  Device: {Config.DEVICE}")
    print(f"  Dataset: {Config.DATASET_ROOT}")

    # ── Data ──
    print("\n[1/5] Loading dataset...")
    train_transform, val_transform = get_transforms()

    train_dataset = BreastUltrasoundDataset(Config.DATASET_ROOT, "train", train_transform)
    val_dataset = BreastUltrasoundDataset(Config.DATASET_ROOT, "val", val_transform)
    test_dataset = BreastUltrasoundDataset(Config.DATASET_ROOT, "test", val_transform)

    print(f"  Train: {len(train_dataset)} samples -> {dict(train_dataset.get_class_distribution())}")
    print(f"  Val:   {len(val_dataset)} samples -> {dict(val_dataset.get_class_distribution())}")
    print(f"  Test:  {len(test_dataset)} samples -> {dict(test_dataset.get_class_distribution())}")

    train_loader = DataLoader(
        train_dataset, batch_size=Config.STAGE1_BATCH_SIZE,
        shuffle=True, num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=Config.STAGE1_BATCH_SIZE,
        shuffle=False, num_workers=2, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=Config.STAGE1_BATCH_SIZE,
        shuffle=False, num_workers=2, pin_memory=True,
    )

    # ── Model ──
    print("\n[2/5] Building model...")
    model = build_model()

    # Loss with label smoothing (anti-overfitting)
    criterion = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)

    # ── Stage 1: Train classifier head only ──
    print("\n[3/5] Stage 1: Training classifier head...")
    optimizer_s1 = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=Config.STAGE1_LR,
        weight_decay=Config.WEIGHT_DECAY,
    )
    scheduler_s1 = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_s1, mode="min", factor=0.5, patience=3,
    )
    early_stop_s1 = EarlyStopping(patience=Config.PATIENCE)

    model, history_s1 = train_stage(
        model, train_loader, val_loader, criterion, optimizer_s1, scheduler_s1,
        Config.STAGE1_EPOCHS, Config.DEVICE,
        "STAGE 1 — Classifier Head Only", early_stop_s1,
    )

    # ── Stage 2: Fine-tune backbone ──
    print("\n[4/5] Stage 2: Fine-tuning backbone...")
    model = unfreeze_backbone(model, from_block=Config.STAGE2_UNFREEZE_FROM)

    optimizer_s2 = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=Config.STAGE2_LR,
        weight_decay=Config.WEIGHT_DECAY,
    )
    scheduler_s2 = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_s2, mode="min", factor=0.5, patience=3,
    )
    early_stop_s2 = EarlyStopping(patience=Config.PATIENCE)

    model, history_s2 = train_stage(
        model, train_loader, val_loader, criterion, optimizer_s2, scheduler_s2,
        Config.STAGE2_EPOCHS, Config.DEVICE,
        "STAGE 2 — Fine-Tune (Backbone Blocks 6-7)", early_stop_s2,
    )

    # ── Evaluation on TEST set ──
    print("\n[5/5] Evaluating on TEST set...")
    test_loss, test_acc = evaluate(model, test_loader, criterion, Config.DEVICE)
    print(f"\n  ╔══════════════════════════════╗")
    print(f"  ║  TEST ACCURACY: {test_acc:.4f}       ║")
    print(f"  ║  TEST LOSS:     {test_loss:.4f}       ║")
    print(f"  ╚══════════════════════════════╝")

    # Detailed metrics
    y_pred, y_true, y_probs = get_predictions(model, test_loader, Config.DEVICE)

    print("\n  Classification Report:")
    print(classification_report(y_true, y_pred, target_names=Config.CLASS_NAMES))

    # ── Plots ──
    print("\n  Generating visualizations...")
    plot_training_history(history_s1, history_s2, Config.OUTPUT_DIR)
    plot_confusion_matrix(y_true, y_pred, Config.CLASS_NAMES, Config.OUTPUT_DIR)
    plot_per_class_metrics(y_true, y_pred, Config.CLASS_NAMES, Config.OUTPUT_DIR)

    # ── Save Model ──
    model_path = os.path.join(Config.OUTPUT_DIR, "best_model.pth")
    torch.save({
        "model_state_dict": model.state_dict(),
        "class_names": Config.CLASS_NAMES,
        "test_accuracy": test_acc,
        "config": {
            "model": Config.MODEL_NAME,
            "img_size": Config.IMG_SIZE,
            "dropout": Config.DROPOUT,
        },
    }, model_path)
    print(f"\n  Model saved: {model_path}")


    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
