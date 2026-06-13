"""
Breast Ultrasound Segmentation — U-Net Training Script
=======================================================
Model: U-Net (from scratch, lightweight)
Task: Binary segmentation (tumor vs background)
Input: 256x256 RGB ultrasound image
Output: 256x256 binary mask
"""

import os
import copy
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ═══════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

class Config:
    # ── Paths ──
    # Assuming script is run from src/ directory or similar
    DATASET_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Dataset_BUSI_with_GT_Split"))
    OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "segmentation"))

    IMG_SIZE = 256
    NUM_CLASSES = 1  # Binary: tumor or not
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    # Training
    EPOCHS = 50
    BATCH_SIZE = 8
    LR = 1e-4
    WEIGHT_DECAY = 1e-4
    PATIENCE = 10

    # Only train on benign + malignant (normal has empty masks)
    CATEGORIES = ["benign", "malignant"]

    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ═══════════════════════════════════════════════════════════════════
#  DATASET
# ═══════════════════════════════════════════════════════════════════

class SegmentationDataset(Dataset):
    """Loads image + mask pairs for segmentation."""

    def __init__(self, root_dir, split, augment=False):
        self.samples = []
        self.augment = augment

        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(Config.IMAGENET_MEAN, Config.IMAGENET_STD),
        ])

        split_dir = os.path.join(root_dir, split)
        for category in Config.CATEGORIES:
            cat_dir = os.path.join(split_dir, category)
            if not os.path.exists(cat_dir):
                continue
            for sample_id in os.listdir(cat_dir):
                img_path = os.path.join(cat_dir, sample_id, "image.png")
                mask_path = os.path.join(cat_dir, sample_id, "mask.png")
                if os.path.exists(img_path) and os.path.exists(mask_path):
                    self.samples.append((img_path, mask_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # Online augmentation (same transform for both)
        if self.augment:
            if random.random() > 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() > 0.5:
                angle = random.uniform(-15, 15)
                image = image.rotate(angle, resample=Image.BICUBIC, fillcolor=128)
                mask = mask.rotate(angle, resample=Image.NEAREST, fillcolor=0)

        # Convert image: normalize with ImageNet stats
        image = self.img_transform(image)

        # Convert mask: to tensor, binarize (0 or 1)
        mask = torch.from_numpy(np.array(mask)).float() / 255.0
        mask = (mask > 0.5).float().unsqueeze(0)  # shape: (1, 256, 256)

        return image, mask


# ═══════════════════════════════════════════════════════════════════
#  U-NET MODEL
# ═══════════════════════════════════════════════════════════════════

class ConvBlock(nn.Module):
    """Two convolutions + BatchNorm + ReLU."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """
    Lightweight U-Net for 256x256 binary segmentation.

    Encoder: 3 → 64 → 128 → 256 → 512 (with max pooling)
    Bottleneck: 512 → 1024
    Decoder: 1024 → 512 → 256 → 128 → 64 (with skip connections)
    Output: 64 → 1 (sigmoid)
    """
    def __init__(self):
        super().__init__()

        # Encoder (downsampling)
        self.enc1 = ConvBlock(3, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, 512)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(512, 1024)

        # Decoder (upsampling)
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(1024, 512)   # 512 (up) + 512 (skip) = 1024 input

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(512, 256)    # 256 + 256 = 512

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(256, 128)    # 128 + 128 = 256

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(128, 64)     # 64 + 64 = 128

        # Final 1x1 convolution → 1 channel output
        self.final = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)                # (B, 64, 256, 256)
        e2 = self.enc2(self.pool(e1))     # (B, 128, 128, 128)
        e3 = self.enc3(self.pool(e2))     # (B, 256, 64, 64)
        e4 = self.enc4(self.pool(e3))     # (B, 512, 32, 32)

        # Bottleneck
        b = self.bottleneck(self.pool(e4))  # (B, 1024, 16, 16)

        # Decoder + skip connections
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))   # (B, 512, 32, 32)
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))  # (B, 256, 64, 64)
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))  # (B, 128, 128, 128)
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))  # (B, 64, 256, 256)

        return self.final(d1)  # (B, 1, 256, 256) — raw logits


# ═══════════════════════════════════════════════════════════════════
#  LOSS FUNCTION: Dice + BCE Combined
# ═══════════════════════════════════════════════════════════════════

class DiceBCELoss(nn.Module):
    """
    Dice Loss: Measures overlap between prediction and ground truth.
    BCE Loss: Standard binary cross-entropy per pixel.
    Combined = better convergence + better boundary detection.
    """
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        # BCE part
        bce_loss = self.bce(logits, targets)

        # Dice part
        probs = torch.sigmoid(logits)
        smooth = 1e-6
        intersection = (probs * targets).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice = (2.0 * intersection + smooth) / (union + smooth)
        dice_loss = 1.0 - dice.mean()

        return bce_loss + dice_loss


# ═══════════════════════════════════════════════════════════════════
#  METRICS
# ═══════════════════════════════════════════════════════════════════

def compute_metrics(logits, targets, threshold=0.5):
    """Compute Dice Score and IoU."""
    preds = (torch.sigmoid(logits) > threshold).float()
    smooth = 1e-6

    intersection = (preds * targets).sum(dim=(2, 3))
    union_dice = preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
    union_iou = union_dice - intersection

    dice = (2.0 * intersection + smooth) / (union_dice + smooth)
    iou = (intersection + smooth) / (union_iou + smooth)

    return dice.mean().item(), iou.mean().item()


# ═══════════════════════════════════════════════════════════════════
#  TRAINING & EVALUATION
# ═══════════════════════════════════════════════════════════════════

class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, score):
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, total_dice, total_iou, count = 0, 0, 0, 0

    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        dice, iou = compute_metrics(logits.detach(), masks)
        total_loss += loss.item() * images.size(0)
        total_dice += dice * images.size(0)
        total_iou += iou * images.size(0)
        count += images.size(0)

    return total_loss / count, total_dice / count, total_iou / count


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_dice, total_iou, count = 0, 0, 0, 0

    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        logits = model(images)
        loss = criterion(logits, masks)

        dice, iou = compute_metrics(logits, masks)
        total_loss += loss.item() * images.size(0)
        total_dice += dice * images.size(0)
        total_iou += iou * images.size(0)
        count += images.size(0)

    return total_loss / count, total_dice / count, total_iou / count


# ═══════════════════════════════════════════════════════════════════
#  VISUALIZATION
# ═══════════════════════════════════════════════════════════════════

def plot_training_history(history, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Segmentation Training History", fontsize=16, fontweight="bold")

    for ax, metric, title in zip(
        axes, ["loss", "dice", "iou"], ["Loss", "Dice Score", "IoU"]
    ):
        ax.plot(history[f"train_{metric}"], "b-o", label="Train", markersize=3)
        ax.plot(history[f"val_{metric}"], "r-o", label="Val", markersize=3)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "training_history.png"), dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: training_history.png")


@torch.no_grad()
def plot_predictions(model, dataset, device, save_path, num_samples=8):
    """Show side-by-side: image | ground truth | prediction."""
    model.eval()
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    axes[0, 0].set_title("Ultrasound Image", fontsize=14)
    axes[0, 1].set_title("Ground Truth Mask", fontsize=14)
    axes[0, 2].set_title("Model Prediction", fontsize=14)

    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

    # Denormalization for display
    mean = torch.tensor(Config.IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(Config.IMAGENET_STD).view(3, 1, 1)

    for i, idx in enumerate(indices):
        image, mask = dataset[idx]

        # Predict
        logits = model(image.unsqueeze(0).to(device))
        pred = (torch.sigmoid(logits) > 0.5).float().cpu().squeeze()

        # Denormalize image for display
        img_display = image * std + mean
        img_display = img_display.permute(1, 2, 0).clamp(0, 1).numpy()

        axes[i, 0].imshow(img_display)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(mask.squeeze(), cmap="gray")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(pred, cmap="gray")
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "predictions.png"), dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: predictions.png")


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    set_seed(Config.SEED)
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("  BREAST ULTRASOUND SEGMENTATION — U-Net")
    print("=" * 60)
    print(f"  Device: {Config.DEVICE}")

    # ── Data ──
    print("\n[1/4] Loading dataset...")
    train_ds = SegmentationDataset(Config.DATASET_ROOT, "train", augment=True)
    val_ds = SegmentationDataset(Config.DATASET_ROOT, "val", augment=False)
    test_ds = SegmentationDataset(Config.DATASET_ROOT, "test", augment=False)
    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # ── Model ──
    print("\n[2/4] Building U-Net...")
    model = UNet().to(Config.DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    criterion = DiceBCELoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)
    early_stop = EarlyStopping(patience=Config.PATIENCE)

    # ── Training ──
    print("\n[3/4] Training...")
    history = {k: [] for k in ["train_loss", "train_dice", "train_iou", "val_loss", "val_dice", "val_iou"]}
    best_dice = 0.0
    best_model_wts = None

    for epoch in range(Config.EPOCHS):
        t0 = time.time()

        train_loss, train_dice, train_iou = train_one_epoch(model, train_loader, criterion, optimizer, Config.DEVICE)
        val_loss, val_dice, val_iou = evaluate(model, val_loader, criterion, Config.DEVICE)

        scheduler.step(val_dice)

        history["train_loss"].append(train_loss)
        history["train_dice"].append(train_dice)
        history["train_iou"].append(train_iou)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)
        history["val_iou"].append(val_iou)

        elapsed = time.time() - t0
        marker = ""
        if val_dice > best_dice:
            best_dice = val_dice
            best_model_wts = copy.deepcopy(model.state_dict())
            marker = " ★ best"

        print(
            f"  Epoch {epoch+1:02d}/{Config.EPOCHS} | "
            f"Train Dice: {train_dice:.4f} IoU: {train_iou:.4f} | "
            f"Val Dice: {val_dice:.4f} IoU: {val_iou:.4f} | "
            f"{elapsed:.1f}s{marker}"
        )

        if early_stop(val_dice):
            print(f"\n  ⚠ Early stopping at epoch {epoch+1}")
            break

    # Restore best weights
    model.load_state_dict(best_model_wts)

    # ── Evaluation ──
    print("\n[4/4] Evaluating on TEST set...")
    test_loss, test_dice, test_iou = evaluate(model, test_loader, criterion, Config.DEVICE)
    print(f"\n  ╔═══════════════════════════════════╗")
    print(f"  ║  TEST DICE SCORE: {test_dice:.4f}          ║")
    print(f"  ║  TEST IoU:        {test_iou:.4f}          ║")
    print(f"  ║  TEST LOSS:       {test_loss:.4f}          ║")
    print(f"  ╚═══════════════════════════════════╝")

    # ── Visualizations ──
    print("\n  Generating visualizations...")
    plot_training_history(history, Config.OUTPUT_DIR)
    plot_predictions(model, test_ds, Config.DEVICE, Config.OUTPUT_DIR)

    # ── Save ──
    model_path = os.path.join(Config.OUTPUT_DIR, "unet_best.pth")
    torch.save({
        "model_state_dict": model.state_dict(),
        "test_dice": test_dice,
        "test_iou": test_iou,
    }, model_path)
    print(f"\n  Model saved: {model_path}")
    print("\n" + "=" * 60)
    print("  SEGMENTATION TRAINING COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
