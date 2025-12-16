import os
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
import skvideo.io
from queue import Queue
import threading
from contextlib import nullcontext

# ===============================
# ARGPARSE
# ===============================
parser = argparse.ArgumentParser("RIFE Final Clean Kaggle")
parser.add_argument('--video', type=str, required=True)
parser.add_argument('--output', type=str, default=None)
parser.add_argument('--multi', type=int, default=2)
parser.add_argument('--scale', type=float, default=1.0)
parser.add_argument('--fp16', action='store_true')
parser.add_argument('--model', type=str, default='train_log')
args = parser.parse_args()

assert args.multi >= 2
assert args.scale in [0.25, 0.5, 1.0]

# ===============================
# DEVICE
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)
torch.backends.cudnn.benchmark = True

autocast_ctx = torch.cuda.amp.autocast if args.fp16 else nullcontext

# ===============================
# LOAD MODEL (SINGLE SOURCE OF TRUTH)
# ===============================
from train_log.RIFE_HDv3 import Model

model = Model()
model.load_model(args.model, -1)
model.eval()
model.device()

assert model.version >= 3.9, "Model must support timestep inference"

# ===============================
# VIDEO IO
# ===============================
cap = cv2.VideoCapture(args.video)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()

target_fps = fps * args.multi

reader = skvideo.io.vreader(args.video)
first_frame = next(reader)
h, w, _ = first_frame.shape

# ===============================
# OUTPUT
# ===============================
if args.output:
    out_path = args.output
else:
    name, ext = os.path.splitext(args.video)
    out_path = f"{name}_{args.multi}x.mp4"

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(out_path, fourcc, target_fps, (w, h))

# ===============================
# PADDING
# ===============================
def pad(img):
    tmp = max(128, int(128 / args.scale))
    ph = ((img.shape[2] - 1) // tmp + 1) * tmp
    pw = ((img.shape[3] - 1) // tmp + 1) * tmp
    return F.pad(img, (0, pw - img.shape[3], 0, ph - img.shape[2]), mode='reflect')

# ===============================
# QUEUES
# ===============================
read_q = Queue(maxsize=30)
write_q = Queue(maxsize=30)

# ===============================
# THREADS (I/O ONLY)
# ===============================
def reader_thread():
    for frame in reader:
        read_q.put(frame)
    read_q.put(None)

def writer_thread():
    while True:
        item = write_q.get()
        if item is None:
            break
        writer.write(item[:, :, ::-1])

threading.Thread(target=reader_thread, daemon=True).start()
threading.Thread(target=writer_thread, daemon=True).start()

# ===============================
# INFERENCE LOOP (CUDA SINGLE THREAD)
# ===============================
last = first_frame
pbar = tqdm(total=total_frames)

while True:
    frame = read_q.get()
    if frame is None:
        break

    I0 = torch.from_numpy(last).permute(2,0,1).unsqueeze(0).float().to(device) / 255.
    I1 = torch.from_numpy(frame).permute(2,0,1).unsqueeze(0).float().to(device) / 255.

    I0 = pad(I0)
    I1 = pad(I1)

    with autocast_ctx():
        mids = []
        for i in range(args.multi - 1):
            t = (i + 1) / args.multi
            mid = model.inference(I0, I1, timestep=t)
            mids.append(mid)

    # write frames
    write_q.put(last)
    for mid in mids:
        img = (mid[0] * 255).clamp(0,255).byte().cpu().numpy().transpose(1,2,0)
        write_q.put(img[:h, :w])

    last = frame
    pbar.update(1)

# last frame
write_q.put(last)
write_q.put(None)

pbar.close()
writer.release()

print("✅ INTERPOLATION SELESAI TANPA KONFLIK")
