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
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor
from model.pytorch_msssim import ssim_matlab

warnings.filterwarnings("ignore")

def transferAudio(sourceVideo, targetVideo):
    import shutil
    import moviepy.editor
    tempAudioFileName = "./temp/audio.mkv"

    if os.path.isdir("temp"):
        shutil.rmtree("temp")
    os.makedirs("temp")
    os.system('ffmpeg -y -i "{}" -c:a copy -vn {}'.format(sourceVideo, tempAudioFileName))

    targetNoAudio = os.path.splitext(targetVideo)[0] + "_noaudio" + os.path.splitext(targetVideo)[1]
    os.rename(targetVideo, targetNoAudio)
    os.system('ffmpeg -y -i "{}" -i {} -c copy "{}"'.format(targetNoAudio, tempAudioFileName, targetVideo))

    if os.path.getsize(targetVideo) == 0:
        tempAudioFileName = "./temp/audio.m4a"
        os.system('ffmpeg -y -i "{}" -c:a aac -b:a 160k -vn {}'.format(sourceVideo, tempAudioFileName))
        os.system('ffmpeg -y -i "{}" -i {} -c copy "{}"'.format(targetNoAudio, tempAudioFileName, targetVideo))
        if (os.path.getsize(targetVideo) == 0):
            os.rename(targetNoAudio, targetVideo)
            print("Audio transfer failed. Interpolated video will have no audio")
        else:
            print("Lossless audio transfer failed. Audio was transcoded to AAC (M4A) instead.")
            os.remove(targetNoAudio)
    else:
        os.remove(targetNoAudio)

    shutil.rmtree("temp")

class CUDAParallelProcessor:
    def __init__(self, model, device, args):
        self.model = model
        self.device = device
        self.args = args
        self.num_workers = min(4, torch.cuda.device_count() * 2)
        self.executor = ThreadPoolExecutor(max_workers=self.num_workers)
        
    def parallel_inference(self, I0, I1, n):
        """Parallel inference untuk multiple intermediate frames"""
        if self.model.version >= 3.9:
            # Untuk model versi baru yang support direct timestep
            futures = []
            for i in range(n):
                timestep = (i + 1) * 1.0 / (n + 1)
                future = self.executor.submit(self.model.inference, I0, I1, timestep, self.args.scale)
                futures.append(future)
            return [future.result() for future in futures]
        else:
            # Untuk model versi lama, gunakan recursive approach
            return self.make_inference(I0, I1, n)
    
    def make_inference(self, I0, I1, n):
        """Recursive inference untuk model versi lama"""
        if n == 1:
            return [self.model.inference(I0, I1, self.args.scale)]
        
        middle = self.model.inference(I0, I1, self.args.scale)
        
        # Process kedua bagian secara parallel
        with ThreadPoolExecutor(max_workers=2) as executor:
            future1 = executor.submit(self.make_inference, I0, middle, n // 2)
            future2 = executor.submit(self.make_inference, middle, I1, n // 2)
            
            first_half = future1.result()
            second_half = future2.result()
        
        if n % 2:
            return [*first_half, middle, *second_half]
        else:
            return [*first_half, *second_half]

    def process_frame_batch(self, frame_batch):
        """Process batch of frames secara parallel"""
        results = []
        for frame_data in frame_batch:
            I0, I1, lastframe = frame_data
            output_frames = self.process_single_pair(I0, I1, lastframe)
            results.extend(output_frames)
        return results
    
    def process_single_pair(self, I0, I1, lastframe):
        """Process single frame pair"""
        # Calculate SSIM untuk detect static frames
        I0_small = F.interpolate(I0, (32, 32), mode='bilinear', align_corners=False)
        I1_small = F.interpolate(I1, (32, 32), mode='bilinear', align_corners=False)
        ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])
        
        output_frames = []
        
        if ssim > 0.996:
            # Static frame - skip interpolation
            output = [I1] * (self.args.multi - 1)
        elif ssim < 0.2:
            # Low similarity - use original frames
            output = [I0] * (self.args.multi - 1)
        else:
            # Normal interpolation
            output = self.parallel_inference(I0, I1, self.args.multi - 1)
        
        # Convert tensors ke numpy frames
        if self.args.montage:
            output_frames.append(np.concatenate((lastframe, lastframe), 1))
            for mid in output:
                mid_np = ((mid[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0))
                output_frames.append(np.concatenate((lastframe, mid_np[:h, :w]), 1))
        else:
            output_frames.append(lastframe)
            for mid in output:
                mid_np = ((mid[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0))
                output_frames.append(mid_np[:h, :w])
        
        return output_frames

parser = argparse.ArgumentParser(description='Interpolation for a pair of images')
parser.add_argument('--video', dest='video', type=str, default=None)
parser.add_argument('--output', dest='output', type=str, default=None)
parser.add_argument('--img', dest='img', type=str, default=None)
parser.add_argument('--montage', dest='montage', action='store_true', help='montage origin video')
parser.add_argument('--model', dest='modelDir', type=str, default='train_log', help='directory with trained model files')
parser.add_argument('--fp16', dest='fp16', action='store_true', help='fp16 mode for faster and more lightweight inference on cards with Tensor Cores')
parser.add_argument('--UHD', dest='UHD', action='store_true', help='support 4k video')
parser.add_argument('--scale', dest='scale', type=float, default=1.0, help='Try scale=0.5 for 4k video')
parser.add_argument('--skip', dest='skip', action='store_true', help='whether to remove static frames before processing')
parser.add_argument('--fps', dest='fps', type=int, default=None)
parser.add_argument('--png', dest='png', action='store_true', help='whether to vid_out png format vid_outs')
parser.add_argument('--ext', dest='ext', type=str, default='mp4', help='vid_out video extension')
parser.add_argument('--exp', dest='exp', type=int, default=1)
parser.add_argument('--multi', dest='multi', type=int, default=2)
parser.add_argument('--batch_size', dest='batch_size', type=int, default=4, help='Batch size for parallel processing')

args = parser.parse_args()

if args.exp != 1:
    args.multi = (2 ** args.exp)
assert (not args.video is None or not args.img is None)
if args.skip:
    print("skip flag is abandoned, please refer to issue #207.")
if args.UHD and args.scale==1.0:
    args.scale = 0.5
assert args.scale in [0.25, 0.5, 1.0, 2.0, 4.0]
if not args.img is None:
    args.png = True

# Setup device dengan optimasi CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)

if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    if args.fp16:
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
    
    print(f"Using GPU: {torch.cuda.get_device_name()}")
    print(f"Batch size: {args.batch_size}")

# Load model
try:
    from train_log.RIFE_HDv3 import Model
except:
    print("Please download our model from model list")

model = Model()
if not hasattr(model, 'version'):
    model.version = 0
model.load_model(args.modelDir, -1)
print("Loaded 3.x/4.x HD model.")
model.eval()
model.device()

# Setup parallel processor
parallel_processor = CUDAParallelProcessor(model, device, args)

if not args.video is None:
    videoCapture = cv2.VideoCapture(args.video)
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    tot_frame = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
    videoCapture.release()
    
    if args.fps is None:
        fpsNotAssigned = True
        args.fps = fps * args.multi
    else:
        fpsNotAssigned = False
        
    videogen = skvideo.io.vreader(args.video)
    lastframe = next(videogen)
    
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video_path_wo_ext, ext = os.path.splitext(args.video)
    print('{}.{}, {} frames in total, {}FPS to {}FPS'.format(video_path_wo_ext, args.ext, tot_frame, fps, args.fps))
    
    if args.png == False and fpsNotAssigned == True:
        print("The audio will be merged after interpolation process")
    else:
        print("Will not merge audio because using png or fps flag!")
else:
    videogen = []
    for f in os.listdir(args.img):
        if 'png' in f:
            videogen.append(f)
    tot_frame = len(videogen)
    videogen.sort(key= lambda x:int(x[:-4]))
    lastframe = cv2.imread(os.path.join(args.img, videogen[0]), cv2.IMREAD_UNCHANGED)[:, :, ::-1].copy()
    videogen = videogen[1:]

h, w, _ = lastframe.shape
vid_out_name = None
vid_out = None

if args.png:
    if not os.path.exists('vid_out'):
        os.mkdir('vid_out')
else:
    if args.output is not None:
        vid_out_name = args.output
    else:
        vid_out_name = '{}_{}X_{}fps.{}'.format(video_path_wo_ext, args.multi, int(np.round(args.fps)), args.ext)
    vid_out = cv2.VideoWriter(vid_out_name, fourcc, args.fps, (w, h))

def clear_write_buffer(user_args, write_buffer):
    cnt = 0
    while True:
        item = write_buffer.get()
        if item is None:
            break
        if user_args.png:
            cv2.imwrite('vid_out/{:0>7d}.png'.format(cnt), item[:, :, ::-1])
            cnt += 1
        else:
            vid_out.write(item[:, :, ::-1])

def build_read_buffer(user_args, read_buffer, videogen):
    try:
        for frame in videogen:
            if not user_args.img is None:
                frame = cv2.imread(os.path.join(user_args.img, frame), cv2.IMREAD_UNCHANGED)[:, :, ::-1].copy()
            if user_args.montage:
                frame = frame[:, left: left + w]
            read_buffer.put(frame)
    except:
        pass
    read_buffer.put(None)

def pad_image(img):
    if args.fp16:
        return F.pad(img, padding).half()
    else:
        return F.pad(img, padding)

# Setup padding
if args.montage:
    left = w // 4
    w = w // 2

tmp = max(128, int(128 / args.scale))
ph = ((h - 1) // tmp + 1) * tmp
pw = ((w - 1) // tmp + 1) * tmp
padding = (0, pw - w, 0, ph - h)

pbar = tqdm(total=tot_frame)
if args.montage:
    lastframe = lastframe[:, left: left + w]

write_buffer = Queue(maxsize=500)
read_buffer = Queue(maxsize=500)

_thread.start_new_thread(build_read_buffer, (args, read_buffer, videogen))
_thread.start_new_thread(clear_write_buffer, (args, write_buffer))

I1 = torch.from_numpy(np.transpose(lastframe, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
I1 = pad_image(I1)
temp = None

# Warm up GPU dengan input size yang benar
print("Warming up GPU...")
with torch.no_grad():
    warm_up_I0 = torch.randn(1, 3, h, w).to(device)
    warm_up_I1 = torch.randn(1, 3, h, w).to(device)
    
    # Apply padding yang sama
    warm_up_I0 = F.pad(warm_up_I0, padding)
    warm_up_I1 = F.pad(warm_up_I1, padding)
    
    if args.fp16:
        warm_up_I0 = warm_up_I0.half()
        warm_up_I1 = warm_up_I1.half()
    
    try:
        _ = model.inference(warm_up_I0, warm_up_I1, args.scale)
        torch.cuda.synchronize()
        print("GPU warm-up successful")
    except Exception as e:
        print(f"GPU warm-up failed: {e}")
        print("Continuing without warm-up...")

# Batch processing untuk meningkatkan throughput
frame_batch = []
batch_size = args.batch_size

while True:
    if temp is not None:
        frame = temp
        temp = None
    else:
        frame = read_buffer.get()
    
    if frame is None:
        break
        
    I0 = I1
    I1 = torch.from_numpy(np.transpose(frame, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
    I1 = pad_image(I1)
    
    # Add to batch
    frame_batch.append((I0.clone(), I1.clone(), lastframe.copy()))
    
    # Process batch when full atau di akhir
    if len(frame_batch) >= batch_size or (frame is None and len(frame_batch) > 0):
        # Process batch secara parallel
        try:
            batch_results = parallel_processor.process_frame_batch(frame_batch)
            
            # Write results
            for result_frame in batch_results:
                write_buffer.put(result_frame)
                pbar.update(1)
                
        except Exception as e:
            print(f"Error processing batch: {e}")
            # Fallback: process frames individually
            for I0_batch, I1_batch, lastframe_batch in frame_batch:
                try:
                    single_results = parallel_processor.process_single_pair(I0_batch, I1_batch, lastframe_batch)
                    for result_frame in single_results:
                        write_buffer.put(result_frame)
                        pbar.update(1)
                except Exception as e2:
                    print(f"Error processing single frame: {e2}")
                    # Skip problem frame
                    pbar.update(1)
        
        frame_batch.clear()
    
    lastframe = frame

# Process remaining frames in batch
if len(frame_batch) > 0:
    try:
        batch_results = parallel_processor.process_frame_batch(frame_batch)
        for result_frame in batch_results:
            write_buffer.put(result_frame)
            pbar.update(1)
    except Exception as e:
        print(f"Error processing final batch: {e}")

# Write last frame
if args.montage:
    write_buffer.put(np.concatenate((lastframe, lastframe), 1))
else:
    write_buffer.put(lastframe)
pbar.update(1)

# Cleanup
import time
while not write_buffer.empty():
    time.sleep(0.1)

write_buffer.put(None)
pbar.close()

if not vid_out is None:
    vid_out.release()

# Move audio to new video file if appropriate
if args.png == False and fpsNotAssigned == True and not args.video is None:
    try:
        transferAudio(args.video, vid_out_name)
    except:
        print("Audio transfer failed. Interpolated video will have no audio")
        targetNoAudio = os.path.splitext(vid_out_name)[0] + "_noaudio" + os.path.splitext(vid_out_name)[1]
        if os.path.exists(targetNoAudio):
            os.rename(targetNoAudio, vid_out_name)

print("Processing completed successfully!")
