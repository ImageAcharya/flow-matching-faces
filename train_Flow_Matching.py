"""
train_Flow_Matching.py
--------
Trains the FlowUNet on a face dataset using Flow Matching.

Saves to  Outputs_Flow_Matching/
    noise_adding_process.png  — 10-panel grid: real image → pure noise (t=0..1),
                                 saved once at the very start of training.
    loss_curve.png            — train vs validation MSE loss over all epochs.
    model_checkpoint.pth      — final trained weights.

Usage:
    python train_Flow_Matching.py
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

from model_Flow_Matching import FlowUNet, forward_process

# ─────────────────────────────────────────────────────────────
# Output directory
# ─────────────────────────────────────────────────────────────
OUT_DIR = "Outputs_Flow_Matching"
os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────
IMAGE_SIZE   = 128
CHANNELS     = 3
BATCH_SIZE   = 32
EPOCHS       = 100
LR           = 1e-4
WEIGHT_DECAY = 1e-4
VAL_SPLIT    = 0.1

DATASET_PATH = "/home/image/mbust/Dataset/1/celeba_hq_256"

# ─────────────────────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# ─────────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────────
preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * CHANNELS, [0.5] * CHANNELS),
])

# ─────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────
class FaceDataset(Dataset):
    def __init__(self, folder: str):
        self.files = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".png"))
        ]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        img = Image.open(self.files[idx]).convert("RGB")
        return preprocess(img)


dataset    = FaceDataset(DATASET_PATH)
val_size   = int(len(dataset) * VAL_SPLIT)
train_size = len(dataset) - val_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

loader     = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
val_loader = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

print(f"Train: {train_size}  |  Val: {val_size}")


# ─────────────────────────────────────────────────────────────
# Helper: tensor → displayable uint8 numpy array
# ─────────────────────────────────────────────────────────────
def tensor_to_uint8(img_tensor: torch.Tensor) -> np.ndarray:
    """Convert a single (C,H,W) tensor in [-1,1] → (H,W,3) uint8."""
    img = (img_tensor + 1) / 2
    img = img.clamp(0, 1)
    return (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────
# Noise-adding visualisation  
# ─────────────────────────────────────────────────────────────
def save_noise_adding_process(sample_image: torch.Tensor, path: str):
    """
    Build a 10-panel grid showing clean → noisy at t = 0.0, 0.1, …, 0.9, 1.0.
    sample_image : (C, H, W) tensor in [-1, 1], on CPU.
    """
    steps = 10
    fig, axes = plt.subplots(1, steps + 1, figsize=(2 * (steps + 1), 2.5))

    for i, t_val in enumerate(np.linspace(0, 1, steps + 1)):
        t_tensor = torch.tensor([t_val], dtype=torch.float32)
        noisy, _ = forward_process(sample_image.unsqueeze(0), t_tensor)
        axes[i].imshow(tensor_to_uint8(noisy[0]))
        axes[i].set_title(f"t={t_val:.1f}", fontsize=8)
        axes[i].axis("off")

    fig.suptitle("Noise-adding process  (t=0 clean → t=1 noise)", y=1.02)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {path}")


# Grab one clean image and save the visualisation right away
_sample_img = next(iter(loader))[0].cpu()   # first image of first batch
save_noise_adding_process(_sample_img, os.path.join(OUT_DIR, "noise_adding_process.png"))


# ─────────────────────────────────────────────────────────────
# Model, Optimiser, Scheduler, Loss
# ─────────────────────────────────────────────────────────────
model     = FlowUNet().to(device)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
loss_fn   = nn.MSELoss()

print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")


# ─────────────────────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────────────────────
train_losses: list[float] = []
val_losses:   list[float] = []

for epoch in range(EPOCHS):
    # ── Train ─────────────────────────────────────────────────
    model.train()
    total_loss = 0.0
    for I in tqdm(loader, desc=f"Epoch {epoch + 1:03d} [train]"):
        I = I.to(device)
        t = torch.rand(I.size(0), device=device)
        I_t, v_true = forward_process(I, t)
        v_pred = model(I_t, t)
        loss   = loss_fn(v_pred, v_true)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_train = total_loss / len(loader)
    train_losses.append(avg_train)

    # ── Validation ────────────────────────────────────────────
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for I in val_loader:
            I = I.to(device)
            # Deterministic, evenly-spaced timesteps for reproducible val loss
            t = torch.linspace(0, 1, steps=I.size(0), device=device)
            I_t, v_true = forward_process(I, t)
            val_loss += loss_fn(model(I_t, t), v_true).item()

    avg_val = val_loss / len(val_loader)
    val_losses.append(avg_val)

    scheduler.step()
    print(f"Epoch {epoch + 1:03d} | Train: {avg_train:.4f} | Val: {avg_val:.4f}")


# ─────────────────────────────────────────────────────────────
# Save checkpoint
# ─────────────────────────────────────────────────────────────
ckpt_path = os.path.join(OUT_DIR, "model_checkpoint.pth")
torch.save(model.state_dict(), ckpt_path)
print(f"Checkpoint saved → {ckpt_path}")


# ─────────────────────────────────────────────────────────────
# Plot loss curves
# ─────────────────────────────────────────────────────────────
loss_path = os.path.join(OUT_DIR, "loss_curve.png")
plt.figure(figsize=(8, 4))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses,   label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Flow Matching UNet — Train vs Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(loss_path, dpi=150)
plt.close()
print(f"Loss curve saved → {loss_path}")