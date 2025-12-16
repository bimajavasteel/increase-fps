# ============================================================
# Practical-RIFE – FINAL STABLE INFERENCE (Kaggle CUDA T4)
# Author: QA Refactor
# Target: Stable, Safe, Reproducible, Copy-Paste Ready
# ============================================================

import os
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
from queue import Queue
import threading
import skvideo.io
import warnings

warnings.filterwarnings("ignore")

# =========================
# CONFIG SAFE DEFAULTS
# =========================
torch.set_grad_enabled(False)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_FP16 = False   # fp16 AKTIF HANYA DI MODEL, BUKAN GLOBAL
MAX_WORKERS = 1    # CUDA SAFE (no race)

if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True


# =========================
# ARGUMENTS
# =========================
parser = argparse.ArgumentParser("RIFE FINAL STABLE")
parser.add_argument("--video", type=str, required=True)
parser.add_argument("--model", type=str, default="train_log")
parser.add_argument("--multi", type=int, default=2)
parser.add_argument("--scale", type=float, default=1.0)
parser.add_argument("--fp16", action="store_true")
parser.add_argument("--output", type=str, default=None)
args = parser.parse_args()

USE_FP16 = args.fp16 and torch.cuda.is_available()


# =========================
# MODEL LOADER (UNIFIED)
# =========================
def load_rife_model(model_dir):
    try:
        from train_log.RIFE_HDv3 import Model
    except:
        raise RuntimeError("Model RIFE_HDv3 tidak ditemukan di train_log")

    model = Model()
    model.load_model(model_dir, -1)
    model.eval()
    model.device()

    if USE_FP16:
        model.half()

    if not hasattr(model, "version"):
        model.version = 4.0

    return model


model = load_rife_model(args.model)


# =========================
# VIDEO IO
# =========================
video_cap = cv2.VideoCapture(args.video)
fps = video_cap.get(cv2.CAP_PROP_FPS)
total_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
video_cap.release()

reader = skvideo.io.vreader(args.video)
first_frame = next(reader)
h, w, _ = first_frame.shape

out_fps = fps * args.multi
out_name = args.output or f"{os.path.splitext(args.video)[0]}_{args.multi}X.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(out_name, fourcc, out_fps, (w, h))


# =========================
# PADDING (CONSISTENT)
# =========================
def pad_tensor(x):
    tmp = max(128, int(128 / args.scale))
    ph = ((x.shape[2] - 1) // tmp + 1) * tmp
    pw = ((x.shape[3] - 1) // tmp + 1) * tmp
    return F.pad(x, (0, pw - x.shape[3], 0, ph - x.shape[2]))


# =========================
# TENSOR CONVERTER
# =========================
def frame_to_tensor(frame):
    t = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    t = t.to(DEVICE)
    if USE_FP16:
        t = t.half()
    return pad_tensor(t)


# =========================
# INFERENCE (SAFE API)
# =========================
def interpolate(I0, I1, n):
    results = []
    for i in range(n):
        t = (i + 1) / (n + 1)
        if model.version >= 3.9:
            out = model.inference(I0, I1, t, args.scale)
        else:
            out = model.inference(I0, I1, args.scale)
        results.append(out)
    return results


# =========================
# MAIN LOOP
# =========================
pbar = tqdm(total=total_frames)

prev = first_frame.copy()
I0 = frame_to_tensor(prev)

for frame in reader:
    I1 = frame_to_tensor(frame)

    mids = interpolate(I0, I1, args.multi - 1)

    # write prev
    writer.write(prev[:, :, ::-1])

    # write mids
    for m in mids:
        img = (m[0].clamp(0, 1) * 255).byte().cpu().numpy().transpose(1, 2, 0)
        writer.write(img[:, :, ::-1])

    prev = frame.copy()
    I0 = I1
    pbar.update(1)

# write last frame
writer.write(prev[:, :, ::-1])
pbar.close()
writer.release()

print("✅ FINAL STABLE INTERPOLATION DONE")
print("📁 Output:", out_name)
