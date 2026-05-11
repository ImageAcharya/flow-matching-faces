"""
infer_Flow_Matching.py
--------
Generates a brand-new unique face on every run using a trained FlowUNet.

Saves to  Outputs_Flow_Matching/   (4 files per run, each tagged with a unique id):
    generation_steps_<id>.png       — 6-panel grid of key denoising frames
    noise_adding_process_<id>.png   — 10-panel grid: the generated face re-noised
                                       from t=0 (clean) → t=1 (pure noise)
    generated_face_<id>.png         — single high-res final face
    generation_<id>.gif             — smooth animation of the full denoising process

Usage:
    python infer_Flow_Matching.py


import os
import uuid
import argparse

import torch
import numpy as np
import matplotlib.pyplot as plt
import imageio

from model_Flow_Matching import FlowUNet, forward_process

# ─────────────────────────────────────────────────────────────
# Output directory
# ─────────────────────────────────────────────────────────────
OUT_DIR = "Outputs_Flow_Matching"
os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# CLI args
# ─────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Flow Matching face generator")
parser.add_argument(
    "--checkpoint",
    type=str,
    default=os.path.join(OUT_DIR, "model_checkpoint.pth"),
    help="Path to trained model weights (default: Outputs_Flow_Matching/model_checkpoint.pth)",
)
parser.add_argument(
    "--steps",
    type=int,
    default=100,
    help="Number of ODE integration steps (default: 100)",
)
parser.add_argument(
    "--image-size",
    type=int,
    default=128,
    help="Spatial resolution used during training (default: 128)",
)
args = parser.parse_args()

# ─────────────────────────────────────────────────────────────
# Unique run ID  (ensures no filename collisions across runs)
# ─────────────────────────────────────────────────────────────
RUN_ID = uuid.uuid4().hex[:8]
print(f"Run ID: {RUN_ID}")

# ─────────────────────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# ─────────────────────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────────────────────
model = FlowUNet().to(device)
model.load_state_dict(torch.load(args.checkpoint, map_location=device))
model.eval()
print(f"Loaded weights from: {args.checkpoint}")


# ─────────────────────────────────────────────────────────────
# Helper: tensor → uint8 numpy array
# ─────────────────────────────────────────────────────────────
def tensor_to_uint8(img_tensor: torch.Tensor, size: int = 256) -> np.ndarray:
    """
    Convert a single (C,H,W) tensor in [-1,1] to a (size,size,3) uint8 array.
    Upscales to `size` for crisp display.
    """
    img = (img_tensor + 1) / 2
    img = img.clamp(0, 1)
    img = torch.nn.functional.interpolate(
        img.unsqueeze(0), size=(size, size),
        mode="bilinear", align_corners=False,
    ).squeeze(0)
    return (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────
# Sampling  (Euler ODE  t: 1 → 0)
# ─────────────────────────────────────────────────────────────
@torch.no_grad()
def sample(steps: int = 100):
    """
    Start from pure noise and integrate backwards to a clean image.

    Returns:
        frames : list of (1, C, H, W) tensors — one per step
    """
    x  = torch.randn(1, 3, args.image_size, args.image_size, device=device)
    dt = 1.0 / steps
    frames = []
    for i in range(steps):
        t = torch.tensor([1.0 - i / steps], device=device)
        v = model(x, t)
        x = x - dt * v
        frames.append(x.clone())
    return frames


# ─────────────────────────────────────────────────────────────
# Generate
# ─────────────────────────────────────────────────────────────
print(f"Generating face ({args.steps} ODE steps) …")
frames = sample(steps=args.steps)
final_face = frames[-1][0]          # (C, H, W) clean face


# ─────────────────────────────────────────────────────────────
# 1.  generation_steps_<id>.png
#     6-panel grid of key denoising frames
# ─────────────────────────────────────────────────────────────
key_indices = [0, args.steps // 10, args.steps // 4,
               args.steps // 2, args.steps * 3 // 4, args.steps - 1]

fig, axes = plt.subplots(1, len(key_indices), figsize=(3 * len(key_indices), 3.5))
for ax, idx in zip(axes, key_indices):
    ax.imshow(tensor_to_uint8(frames[idx][0]))
    progress = round((idx + 1) / args.steps * 100)
    ax.set_title(f"Step {idx + 1}\n({progress}%)", fontsize=9)
    ax.axis("off")
fig.suptitle("Denoising Steps", fontsize=12)
plt.tight_layout()
steps_path = os.path.join(OUT_DIR, f"generation_steps_{RUN_ID}.png")
plt.savefig(steps_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved → {steps_path}")


# ─────────────────────────────────────────────────────────────
# 2.  noise_adding_process_<id>.png
#     10-panel grid: the final generated face re-noised t=0→1
# ─────────────────────────────────────────────────────────────
n_panels = 10
fig, axes = plt.subplots(1, n_panels + 1, figsize=(2.2 * (n_panels + 1), 2.8))
face_cpu  = final_face.unsqueeze(0).cpu()

for i, t_val in enumerate(np.linspace(0, 1, n_panels + 1)):
    t_tensor    = torch.tensor([t_val], dtype=torch.float32)
    noisy, _    = forward_process(face_cpu, t_tensor)
    axes[i].imshow(tensor_to_uint8(noisy[0]))
    axes[i].set_title(f"t={t_val:.1f}", fontsize=8)
    axes[i].axis("off")

fig.suptitle("Generated face re-noised  (t=0 clean → t=1 noise)", y=1.02)
plt.tight_layout()
noise_path = os.path.join(OUT_DIR, f"noise_adding_process_{RUN_ID}.png")
plt.savefig(noise_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved → {noise_path}")


# ─────────────────────────────────────────────────────────────
# 3.  generated_face_<id>.png
#     Single high-res final face
# ─────────────────────────────────────────────────────────────
face_path = os.path.join(OUT_DIR, f"generated_face_{RUN_ID}.png")
fig, ax = plt.subplots(figsize=(4, 4))
ax.imshow(tensor_to_uint8(final_face, size=512))
ax.set_title(f"Generated Face  [id: {RUN_ID}]", fontsize=10)
ax.axis("off")
plt.tight_layout()
plt.savefig(face_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved → {face_path}")


# ─────────────────────────────────────────────────────────────
# 4.  generation_<id>.gif
#     Smooth animation of the full denoising process (15 fps)
# ─────────────────────────────────────────────────────────────
gif_frames = [tensor_to_uint8(f[0]) for f in frames]
gif_path   = os.path.join(OUT_DIR, f"generation_{RUN_ID}.gif")
imageio.mimsave(gif_path, gif_frames, fps=15, loop=0)
print(f"Saved → {gif_path}")

print(f"\nAll outputs written to: {OUT_DIR}/  (run id: {RUN_ID})")
