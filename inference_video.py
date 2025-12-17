import os
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
import warnings
import _thread
import skvideo.io
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
from model.pytorch_msssim import ssim_matlab
import time

warnings.filterwarnings("ignore")

# =========================
# Audio Transfer
# =========================
def transferAudio(sourceVideo, targetVideo):
    import shutil
    tempAudioFileName = "./temp/audio.mkv"

    if os.path.isdir("temp"):
        shutil.rmtree("temp")
    os.makedirs("temp")

    os.system(f'ffmpeg -y -i "{sourceVideo}" -c:a copy -vn {tempAudioFileName}')

    targetNoAudio = os.path.splitext(targetVideo)[0] + "_noaudio" + os.path.splitext(targetVideo)[1]
    os.rename(targetVideo, targetNoAudio)

    os.system(f'ffmpeg -y -i "{targetNoAudio}" -i {tempAudioFileName} -c copy "{targetVideo}"')

    if os.path.getsize(targetVideo) == 0:
        tempAudioFileName = "./temp/audio.m4a"
        os.system(f'ffmpeg -y -i "{sourceVideo}" -c:a aac -b:a 160k -vn {tempAudioFileName}')
        os.system(f'ffmpeg -y -i "{targetNoAudio}" -i {tempAudioFileName} -c copy "{targetVideo}"')
        if os.path.getsize(targetVideo) == 0:
            os.rename(targetNoAudio, targetVideo)
        else:
            os.remove(targetNoAudio)
    else:
        os.remove(targetNoAudio)

    shutil.rmtree("temp")


# =========================
# CUDA Parallel Processor
# =========================
class CUDAParallelProcessor:
    def __init__(self, model, args):
        self.model = model
        self.args = args
        self.executor = ThreadPoolExecutor(max_workers=min(4, torch.cuda.device_count() * 2))

    def interpolate(self, I0, I1):
        if self.model.version >= 3.9:
            return [
                self.model.inference(I0, I1, (i + 1) / self.args.multi, self.args.scale)
                for i in range(self.args.multi - 1)
            ]
        else:
            return self.model.make_inference(I0, I1, self.args.multi - 1)

    def process_pair(self, I0, I1, lastframe):
        I0s = F.interpolate(I0, (32, 32), mode="bilinear", align_corners=False)
        I1s = F.interpolate(I1, (32, 32), mode="bilinear", align_corners=False)
        ssim = ssim_matlab(I0s[:, :3], I1s[:, :3])

        if ssim > 0.996:
            mids = [I1] * (self.args.multi - 1)
        elif ssim < 0.2:
            mids = [I0] * (self.args.multi - 1)
        else:
            mids = self.interpolate(I0, I1)

        frames = [lastframe]
        for m in mids:
            frames.append((m[0] * 255).byte().cpu().numpy().transpose(1, 2, 0))
        return frames


# =========================
# Argparse
# =========================
parser = argparse.ArgumentParser()
parser.add_argument("--video", type=str)
parser.add_argument("--output", type=str)
parser.add_argument("--model", default="train_log")
parser.add_argument("--multi", type=int, default=2)
parser.add_argument("--scale", type=float, default=1.0)
parser.add_argument("--fp16", action="store_true")
parser.add_argument("--ext", default="mp4")
args = parser.parse_args()

# =========================
# Device
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)
torch.backends.cudnn.benchmark = True

# =========================
# Model Load
# =========================
from train_log.RIFE_HDv3 import Model
model = Model()
model.load_model(args.model, -1)
model.eval()
model.device()

processor = CUDAParallelProcessor(model, args)

# =========================
# Video IO
# =========================
cap = cv2.VideoCapture(args.video)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()

reader = skvideo.io.vreader(args.video)
lastframe = next(reader)

h, w, _ = lastframe.shape
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out_name = args.output or f"out_{args.multi}x.{args.ext}"
writer = cv2.VideoWriter(out_name, fourcc, fps * args.multi, (w, h))

# =========================
# Padding
# =========================
ph = ((h - 1) // 64 + 1) * 64
pw = ((w - 1) // 64 + 1) * 64
padding = (0, pw - w, 0, ph - h)

def to_tensor(frame):
    t = torch.from_numpy(frame.transpose(2, 0, 1)).unsqueeze(0).float().to(device) / 255.
    t = F.pad(t, padding)
    return t.half() if args.fp16 else t

I1 = to_tensor(lastframe)

# =========================
# Progress Bar (PRODUCTION)
# =========================
start_time = time.time()
pbar = tqdm(
    total=total_frames,
    ncols=100,
    bar_format="🎞️ {l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
)

# =========================
# Main Loop
# =========================
for frame in reader:
    I0 = I1
    I1 = to_tensor(frame)

    frames = processor.process_pair(I0, I1, lastframe)
    for f in frames:
        writer.write(f[:, :, ::-1])

    lastframe = frame
    pbar.update(1)

pbar.close()
writer.release()

# =========================
# Audio Merge
# =========================
transferAudio(args.video, out_name)

print("✅ DONE | Interpolation selesai dengan progress bar real-time.")
