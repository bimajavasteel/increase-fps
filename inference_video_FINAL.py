# ============================================================
# Practical-RIFE – FINAL STABLE VIDEO INFERENCE (NO skvideo)
# Compatible: RIFE v4.26 | Kaggle CUDA T4 | Python 3.10+
# ============================================================

import os
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
import warnings

warnings.filterwarnings("ignore")
torch.set_grad_enabled(False)

# ============================================================
# ARGUMENTS
# ============================================================
parser = argparse.ArgumentParser("RIFE FINAL STABLE")
parser.add_argument("--video", type=str, required=True)
parser.add_argument("--model", type=str, default="train_log")
parser.add_argument("--multi", type=int, default=2)
parser.add_argument("--scale", type=float, default=1.0)
parser.add_argument("--fp16", action="store_true")
parser.add_argument("--output", type=str, default=None)
args = parser.parse_args()

# ============================================================
# DEVICE
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

USE_FP16 = args.fp16 and torch.cuda.is_available()

# ============================================================
# LOAD MODEL (RIFE v4.x SAFE)
# ============================================================
def load_rife_model(model_dir):
    try:
        from train_log.RIFE_HDv3 import Model
    except Exception as e:
        raise RuntimeError("RIFE_HDv3 tidak ditemukan di train_log") from e

    model = Model()
    model.load_model(model_dir, -1)
    model.eval()
    model.device()

    # ✅ FP16 YANG BENAR UNTUK RIFE
    if USE_FP16 and hasattr(model, "net"):
        model.net.half()

    if not hasattr(model, "version"):
        model.version = 4.0

    return model

model = load_rife_model(args.model)

# ============================================================
# VIDEO INPUT / OUTPUT (OpenCV ONLY)
# ============================================================
cap = cv2.VideoCapture(args.video)
if not cap.isOpened():
    raise RuntimeError(f"Gagal membuka video: {args.video}")

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out_fps = fps * args.multi
output_path = args.output or f"{os.path.splitext(args.video)[0]}_{args.multi}X.mp4"

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(output_path, fourcc, out_fps, (w, h))

# ============================================================
# PADDING (KONSISTEN & AMAN)
# ============================================================
def pad_tensor(x):
    base = max(128, int(128 / args.scale))
    ph = ((x.shape[2] - 1) // base + 1) * base
    pw = ((x.shape[3] - 1) // base + 1) * base
    return F.pad(x, (0, pw - x.shape[3], 0, ph - x.shape[2]))

# ============================================================
# FRAME → TENSOR
# ============================================================
def frame_to_tensor(frame):
    t = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    t = t.to(DEVICE)
    if USE_FP16:
        t = t.half()
    return pad_tensor(t)

# ============================================================
# INTERPOLATION
# ============================================================
def interpolate(I0, I1, count):
    outs = []
    for i in range(count):
        t = (i + 1) / (count + 1)
        out = model.inference(I0, I1, t, args.scale)
        outs.append(out)
    return outs

# ============================================================
# MAIN LOOP
# ============================================================
ret, prev_frame = cap.read()
if not ret:
    raise RuntimeError("Tidak bisa membaca frame pertama")

prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2RGB)
I0 = frame_to_tensor(prev_frame)

pbar = tqdm(total=total_frames - 1, desc="Interpolating")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    I1 = frame_to_tensor(frame)

    mids = interpolate(I0, I1, args.multi - 1)

    # frame asli
    writer.write(prev_frame[:, :, ::-1])

    # frame interpolasi
    for mid in mids:
        img = (mid[0].clamp(0, 1) * 255).byte().cpu().numpy().transpose(1, 2, 0)
        writer.write(img[:, :, ::-1])

    prev_frame = frame
    I0 = I1
    pbar.update(1)

# frame terakhir
writer.write(prev_frame[:, :, ::-1])

pbar.close()
cap.release()
writer.release()

print("✅ SELESAI TANPA sk-video")
print("📁 Output:", output_path)
