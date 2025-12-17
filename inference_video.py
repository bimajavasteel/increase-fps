import os
import cv2
import torch
import argparse
import numpy as np
import time
import subprocess
import warnings
import _thread
import skvideo.io
from queue import Queue
from torch.nn import functional as F
from model.pytorch_msssim import ssim_matlab

warnings.filterwarnings("ignore")

# ============================================================
# 🔥 EMOTE REAL-TIME PROGRESS MONITOR (PATCH ONLY)
# ============================================================

class UltraProgressMonitor:
    def __init__(self, total_frames, args):
        self.total = total_frames
        self.done = 0
        self.start = time.time()
        self.last = 0
        self.args = args

        try:
            self.gpu = torch.cuda.get_device_name()
            self.vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        except:
            self.gpu = "CPU"
            self.vram_total = 0

    def gpu_stat(self):
        try:
            out = subprocess.check_output(
                ["nvidia-smi",
                 "--query-gpu=memory.used,utilization.gpu",
                 "--format=csv,noheader,nounits"]
            ).decode().strip()
            mem, util = out.split(",")
            return float(mem)/1024, int(util)
        except:
            return 0.0, 0

    def update(self, step=1):
        self.done += step
        now = time.time()
        if now - self.last < 0.5:
            return
        self.last = now

        elapsed = now - self.start
        fps = self.done / elapsed if elapsed > 0 else 0
        eta = int((self.total - self.done) / fps) if fps > 0 else 0
        percent = self.done / self.total * 100
        vram, util = self.gpu_stat()

        bar_len = 20
        filled = int(bar_len * percent / 100)
        bar = "🟩" * filled + "⬛" * (bar_len - filled)

        status = (
            "💥 GPU NGEGAS BROOO 🔥🔥🔥" if util > 90 else
            "⚡ STABIL & KENCANG 😎" if util > 70 else
            "🧊 GPU SANTAI 😴"
        )

        print("\033c", end="")
        print("🚀🎮💥 RIFE OVERDRIVE FPS MODE 💥🎮🚀")
        print(f"🖥️  GPU   : {self.gpu} 💎")
        print(f"⚙️  Mode  : {'FP16 🔥' if self.args.fp16 else 'FP32 🧊'} | "
              f"x{self.args.multi} FPS 🚀")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"{bar}  {percent:6.2f}% 😎")
        print(f"🎞️  Frame  : {self.done} / {self.total} 🧩")
        print(f"🚀 FPS OUT : {fps:6.2f} ⚡")
        print(f"⏳ ETA    : {time.strftime('%M:%S', time.gmtime(eta))} ⌛")
        print(f"💾 VRAM   : {vram:.2f}/{self.vram_total:.2f} GB 🧠")
        print(f"🔥 GPU    : {util}% 🥵")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(status)

# ============================================================
# ⚙️ ARGUMENTS (ASLI — TIDAK DIUBAH)
# ============================================================

parser = argparse.ArgumentParser(description='Interpolation for a pair of images')
parser.add_argument('--video', type=str, default=None)
parser.add_argument('--output', type=str, default=None)
parser.add_argument('--img', type=str, default=None)
parser.add_argument('--montage', action='store_true')
parser.add_argument('--model', type=str, default='train_log')
parser.add_argument('--fp16', action='store_true')
parser.add_argument('--UHD', action='store_true')
parser.add_argument('--scale', type=float, default=1.0)
parser.add_argument('--skip', action='store_true')
parser.add_argument('--fps', type=int, default=None)
parser.add_argument('--png', action='store_true')
parser.add_argument('--ext', type=str, default='mp4')
parser.add_argument('--exp', type=int, default=1)
parser.add_argument('--multi', type=int, default=2)
args = parser.parse_args()

# ============================================================
# 🖥️ DEVICE (ASLI)
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)

if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    if args.fp16:
        torch.set_default_tensor_type(torch.cuda.HalfTensor)

# ============================================================
# 🧠 LOAD MODEL (ASLI)
# ============================================================

from train_log.RIFE_HDv3 import Model
model = Model()
model.load_model(args.model, -1)
model.eval()
model.device()

# ============================================================
# 🎥 VIDEO INPUT (ASLI)
# ============================================================

assert args.video is not None
cap = cv2.VideoCapture(args.video)
fps_in = cap.get(cv2.CAP_PROP_FPS)
total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()

if args.fps is None:
    args.fps = fps_in * args.multi

reader = skvideo.io.vreader(args.video)
first = next(reader)
h, w, _ = first.shape

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_name = args.output or f"{os.path.splitext(args.video)[0]}_{args.multi}x.{args.ext}"
writer = cv2.VideoWriter(out_name, fourcc, args.fps, (w, h))

# ============================================================
# 🧠 PROGRESS INIT (PATCH)
# ============================================================

progress = UltraProgressMonitor(total_frame * args.multi, args)

# ============================================================
# 🔄 PIPELINE (ASLI + HOOK PROGRESS)
# ============================================================

def pad(img):
    tmp = max(128, int(128 / args.scale))
    ph = ((img.shape[2] - 1) // tmp + 1) * tmp
    pw = ((img.shape[3] - 1) // tmp + 1) * tmp
    return F.pad(img, (0, pw - img.shape[3], 0, ph - img.shape[2]))

prev = first
I1 = pad(torch.from_numpy(prev.transpose(2,0,1)).unsqueeze(0).to(device).float()/255.)
writer.write(prev)
progress.update()

for frame in reader:
    I0 = I1
    I1 = pad(torch.from_numpy(frame.transpose(2,0,1)).unsqueeze(0).to(device).float()/255.)

    I0s = F.interpolate(I0, (32,32), mode='bilinear', align_corners=False)
    I1s = F.interpolate(I1, (32,32), mode='bilinear', align_corners=False)
    ssim = ssim_matlab(I0s[:,:3], I1s[:,:3])

    if ssim > 0.996:
        mids = [I1] * (args.multi - 1)
    else:
        mids = [model.inference(I0, I1, (i+1)/args.multi) for i in range(args.multi-1)]

    for mid in mids:
        out = (mid[0]*255).byte().cpu().numpy().transpose(1,2,0)[:h,:w]
        writer.write(out)
        progress.update()

    writer.write(frame)
    progress.update()

writer.release()

print("\n✅ SELESAI — VIDEO SIAP 😎🔥")
