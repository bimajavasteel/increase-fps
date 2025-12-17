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

warnings.filterwarnings("ignore")

# =========================
# AUDIO TRANSFER
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
            print("Audio merge failed")
        else:
            os.remove(targetNoAudio)
    else:
        os.remove(targetNoAudio)

    shutil.rmtree("temp")

# =========================
# ARGUMENTS
# =========================
parser = argparse.ArgumentParser()
parser.add_argument('--video', type=str)
parser.add_argument('--output', type=str)
parser.add_argument('--img', type=str)
parser.add_argument('--model', type=str, default='train_log')
parser.add_argument('--multi', type=int, default=2)
parser.add_argument('--scale', type=float, default=1.0)
parser.add_argument('--fp16', action='store_true')
parser.add_argument('--fps', type=int)
parser.add_argument('--png', action='store_true')
parser.add_argument('--ext', type=str, default='mp4')
parser.add_argument('--batch_size', type=int, default=4)
args = parser.parse_args()

assert args.video or args.img

# =========================
# CUDA SETUP (NO WARM-UP)
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)

if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    if args.fp16:
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
    print("GPU:", torch.cuda.get_device_name())

# =========================
# LOAD MODEL
# =========================
from train_log.RIFE_HDv3 import Model
model = Model()
model.load_model(args.model, -1)
model.eval()
model.device()

# =========================
# VIDEO IO
# =========================
videogen = skvideo.io.vreader(args.video)
lastframe = next(videogen)
h, w, _ = lastframe.shape

if args.output:
    out_name = args.output
else:
    base, _ = os.path.splitext(args.video)
    out_name = f"{base}_{args.multi}X.{args.ext}"

fps_src = cv2.VideoCapture(args.video).get(cv2.CAP_PROP_FPS)
fps_out = args.fps if args.fps else fps_src * args.multi

fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
vid_out = cv2.VideoWriter(out_name, fourcc, fps_out, (w, h))

# =========================
# PADDING
# =========================
tmp = max(128, int(128 / args.scale))
ph = ((h - 1) // tmp + 1) * tmp
pw = ((w - 1) // tmp + 1) * tmp
padding = (0, pw - w, 0, ph - h)

def pad(x):
    return F.pad(x, padding)

# =========================
# PROCESS LOOP
# =========================
pbar = tqdm()
I1 = pad(torch.from_numpy(lastframe.transpose(2,0,1)).to(device).unsqueeze(0).float() / 255.)

for frame in videogen:
    I0 = I1
    I1 = pad(torch.from_numpy(frame.transpose(2,0,1)).to(device).unsqueeze(0).float() / 255.)

    I0s = F.interpolate(I0, (32,32))
    I1s = F.interpolate(I1, (32,32))
    ssim = ssim_matlab(I0s[:,:3], I1s[:,:3])

    if ssim < 0.2:
        mids = [I0] * (args.multi - 1)
    else:
        mids = []
        for i in range(args.multi - 1):
            t = (i + 1) / args.multi
            mids.append(model.inference(I0, I1, t, args.scale))

    vid_out.write((I0[0]*255).byte().cpu().numpy().transpose(1,2,0)[:h,:w])
    for m in mids:
        vid_out.write((m[0]*255).byte().cpu().numpy().transpose(1,2,0)[:h,:w])

    pbar.update(1)

vid_out.write((I1[0]*255).byte().cpu().numpy().transpose(1,2,0)[:h,:w])
vid_out.release()
pbar.close()

# =========================
# AUDIO
# =========================
try:
    transferAudio(args.video, out_name)
except:
    print("Audio skipped")

print("DONE (NO WARM-UP)")
