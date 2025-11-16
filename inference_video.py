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
import torch.cuda.nvtx as nvtx
from model.pytorch_msssim import ssim_matlab

warnings.filterwarnings("ignore")

def transferAudio(sourceVideo, targetVideo):
    import shutil
    import moviepy.editor
    tempAudioFileName = "./temp/audio.mkv"

    # split audio from original video file and store in "temp" directory
    if True:

        # clear old "temp" directory if it exits
        if os.path.isdir("temp"):
            # remove temp directory
            shutil.rmtree("temp")
        # create new "temp" directory
        os.makedirs("temp")
        # extract audio from video
        os.system('ffmpeg -y -i "{}" -c:a copy -vn {}'.format(sourceVideo, tempAudioFileName))

    targetNoAudio = os.path.splitext(targetVideo)[0] + "_noaudio" + os.path.splitext(targetVideo)[1]
    os.rename(targetVideo, targetNoAudio)
    # combine audio file and new video file
    os.system('ffmpeg -y -i "{}" -i {} -c copy "{}"'.format(targetNoAudio, tempAudioFileName, targetVideo))

    if os.path.getsize(targetVideo) == 0: # if ffmpeg failed to merge the video and audio together try converting the audio to aac
        tempAudioFileName = "./temp/audio.m4a"
        os.system('ffmpeg -y -i "{}" -c:a aac -b:a 160k -vn {}'.format(sourceVideo, tempAudioFileName))
        os.system('ffmpeg -y -i "{}" -i {} -c copy "{}"'.format(targetNoAudio, tempAudioFileName, targetVideo))
        if (os.path.getsize(targetVideo) == 0): # if aac is not supported by selected format
            os.rename(targetNoAudio, targetVideo)
            print("Audio transfer failed. Interpolated video will have no audio")
        else:
            print("Lossless audio transfer failed. Audio was transcoded to AAC (M4A) instead.")

            # remove audio-less video
            os.remove(targetNoAudio)
    else:
        os.remove(targetNoAudio)

    # remove temp directory
    shutil.rmtree("temp")

class CUDAMemoryManager:
    def __init__(self, max_batch_size=4, num_streams=4):
        self.max_batch_size = max_batch_size
        self.num_streams = num_streams
        self.streams = [torch.cuda.Stream() for _ in range(num_streams)]
        self.current_stream = 0
        self.lock = _thread.allocate_lock()
        
    def get_stream(self):
        with self.lock:
            stream = self.streams[self.current_stream]
            self.current_stream = (self.current_stream + 1) % self.num_streams
            return stream

    def synchronize_all(self):
        for stream in self.streams:
            stream.synchronize()

class ParallelFrameProcessor:
    def __init__(self, model, device, args, memory_manager):
        self.model = model
        self.device = device
        self.args = args
        self.memory_manager = memory_manager
        self.executor = ThreadPoolExecutor(max_workers=min(8, torch.cuda.device_count() * 2))
        
    def parallel_inference(self, I0, I1, n):
        """Parallel inference menggunakan multi-threading CUDA"""
        nvtx.range_push("parallel_inference")
        
        if self.model.version >= 3.9:
            futures = []
            for i in range(n):
                stream = self.memory_manager.get_stream()
                future = self.executor.submit(
                    self.single_inference_with_stream, 
                    I0, I1, (i+1) * 1. / (n+1), stream
                )
                futures.append(future)
            
            results = [future.result() for future in futures]
            nvtx.range_pop()
            return results
        else:
            results = self.make_inference_parallel(I0, I1, n)
            nvtx.range_pop()
            return results
    
    def single_inference_with_stream(self, I0, I1, timestep, stream):
        """Single inference dengan stream tertentu"""
        with torch.cuda.stream(stream):
            result = self.model.inference(I0, I1, timestep, self.args.scale)
            torch.cuda.current_stream().synchronize()
            return result
    
    def make_inference_parallel(self, I0, I1, n):
        """Versi parallel dari make_inference untuk model versi lama"""
        if n == 1:
            return [self.model.inference(I0, I1, self.args.scale)]
        
        # Gunakan stream terpisah untuk middle frame
        middle_stream = self.memory_manager.get_stream()
        with torch.cuda.stream(middle_stream):
            middle = self.model.inference(I0, I1, self.args.scale)
        
        # Process kedua bagian secara parallel
        with ThreadPoolExecutor(max_workers=2) as executor:
            future1 = executor.submit(self.make_inference_parallel, I0, middle, n//2)
            future2 = executor.submit(self.make_inference_parallel, middle, I1, n//2)
            
            first_half = future1.result()
            second_half = future2.result()
        
        if n % 2:
            return [*first_half, middle, *second_half]
        else:
            return [*first_half, *second_half]
    
    def batch_process_frames(self, frame_batches):
        """Process multiple frame batches in parallel"""
        nvtx.range_push("batch_process_frames")
        
        futures = []
        for frame_batch in frame_batches:
            future = self.executor.submit(self.process_single_batch, frame_batch)
            futures.append(future)
        
        results = []
        for future in futures:
            batch_results = future.result()
            results.extend(batch_results)
        
        nvtx.range_pop()
        return results
    
    def process_single_batch(self, frame_batch):
        """Process a single batch of frames"""
        I0, I1, lastframe, original_frame = frame_batch
        
        # Calculate SSIM
        I0_small = F.interpolate(I0, (32, 32), mode='bilinear', align_corners=False)
        I1_small = F.interpolate(I1, (32, 32), mode='bilinear', align_corners=False)
        ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])
        
        output_frames = []
        
        if ssim > 0.996:
            # Static frame handling
            output = [I1] * (self.args.multi - 1)
        elif ssim < 0.2:
            # Low similarity - use original frames
            output = [I0] * (self.args.multi - 1)
        else:
            # Normal interpolation
            output = self.parallel_inference(I0, I1, self.args.multi - 1)
        
        # Prepare output frames
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

class OptimizedVideoReader:
    def __init__(self, video_path, batch_size=4):
        self.video_path = video_path
        self.batch_size = batch_size
        self.frames = []
        self.current_index = 0
        
        # Pre-load frames for better pipelining
        self.preload_frames()
    
    def preload_frames(self):
        """Pre-load frames untuk mengurangi I/O bottleneck"""
        print("Pre-loading frames...")
        cap = cv2.VideoCapture(self.video_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            self.frames.append(frame)
        cap.release()
        print(f"Pre-loaded {len(self.frames)} frames")
    
    def read_batch(self):
        """Read batch of frames"""
        if self.current_index >= len(self.frames):
            return None
        
        end_index = min(self.current_index + self.batch_size, len(self.frames))
        batch = self.frames[self.current_index:end_index]
        self.current_index = end_index
        
        return batch

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
parser.add_argument('--num_streams', dest='num_streams', type=int, default=4, help='Number of CUDA streams')

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
    # Optimasi CUDA settings
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    if args.fp16:
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
    
    print(f"Using GPU: {torch.cuda.get_device_name()}")
    print(f"CUDA streams: {args.num_streams}")
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

# Setup memory manager
memory_manager = CUDAMemoryManager(
    max_batch_size=args.batch_size, 
    num_streams=args.num_streams
)

# Setup parallel processor
parallel_processor = ParallelFrameProcessor(model, device, args, memory_manager)

if not args.video is None:
    # Gunakan optimized video reader
    video_reader = OptimizedVideoReader(args.video, batch_size=args.batch_size)
    tot_frame = len(video_reader.frames)
    
    videoCapture = cv2.VideoCapture(args.video)
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    videoCapture.release()
    
    if args.fps is None:
        fpsNotAssigned = True
        args.fps = fps * args.multi
    else:
        fpsNotAssigned = False
        
    print('{}.{}, {} frames in total, {}FPS to {}FPS'.format(
        os.path.splitext(args.video)[0], args.ext, tot_frame, fps, args.fps))
    
    if args.png == False and fpsNotAssigned == True:
        print("The audio will be merged after interpolation process")
    else:
        print("Will not merge audio because using png or fps flag!")
        
    # Get first frame
    lastframe = video_reader.frames[0] if video_reader.frames else None
    video_reader.current_index = 1  # Skip first frame since we already have it
else:
    # Image sequence processing
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
        video_path_wo_ext = os.path.splitext(args.video)[0]
        vid_out_name = '{}_{}X_{}fps.{}'.format(video_path_wo_ext, args.multi, int(np.round(args.fps)), args.ext)
    vid_out = cv2.VideoWriter(vid_out_name, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), args.fps, (w, h))

def clear_write_buffer(user_args, write_buffer):
    cnt = 0
    while True:
        try:
            item = write_buffer.get(timeout=1)
            if item is None:
                break
            if user_args.png:
                cv2.imwrite('vid_out/{:0>7d}.png'.format(cnt), item[:, :, ::-1])
                cnt += 1
            else:
                vid_out.write(item[:, :, ::-1])
        except Empty:
            continue

def build_frame_batches(reader, batch_size, h, w, padding, device, args):
    """Build batches of frames untuk processing"""
    batches = []
    current_batch = []
    
    while True:
        frame_batch = reader.read_batch()
        if frame_batch is None:
            break
            
        for frame in frame_batch:
            if args.montage:
                frame = frame[:, left: left + w]
            
            tensor_frame = torch.from_numpy(np.transpose(frame, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
            tensor_frame = F.pad(tensor_frame, padding)
            
            current_batch.append(tensor_frame)
            
            if len(current_batch) >= batch_size:
                batches.append(current_batch)
                current_batch = []
    
    if current_batch:
        batches.append(current_batch)
    
    return batches

def process_video_optimized():
    """Optimized main processing loop"""
    global lastframe, h, w
    
    # Setup padding
    if args.montage:
        left = w // 4
        w = w // 2
        
    tmp = max(128, int(128 / args.scale))
    ph = ((h - 1) // tmp + 1) * tmp
    pw = ((w - 1) // tmp + 1) * tmp
    padding = (0, pw - w, 0, ph - h)
    
    pbar = tqdm(total=tot_frame)
    write_buffer = Queue(maxsize=500)
    
    # Start writer thread
    _thread.start_new_thread(clear_write_buffer, (args, write_buffer))
    
    # Process frames in batches
    frame_batches = []
    I1 = torch.from_numpy(np.transpose(lastframe, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
    I1 = F.pad(I1, padding)
    
    if args.montage:
        lastframe = lastframe[:, left: left + w]
    
    # Build processing batches
    processing_batches = []
    temp_frame = None
    
    for i in range(0, tot_frame - 1, args.batch_size):
        batch_frames = []
        
        for j in range(args.batch_size):
            if i + j >= tot_frame - 1:
                break
                
            if temp_frame is not None:
                frame = temp_frame
                temp_frame = None
            else:
                if args.video is not None:
                    frame = video_reader.frames[i + j + 1]  # +1 karena kita mulai dari frame kedua
                else:
                    frame_path = os.path.join(args.img, videogen[i + j])
                    frame = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)[:, :, ::-1].copy()
            
            I0 = I1
            I1_tensor = torch.from_numpy(np.transpose(frame, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
            I1_tensor = F.pad(I1_tensor, padding)
            
            # Check for static frames
            I0_small = F.interpolate(I0, (32, 32), mode='bilinear', align_corners=False)
            I1_small = F.interpolate(I1_tensor, (32, 32), mode='bilinear', align_corners=False)
            ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])
            
            if ssim > 0.996:
                # Skip static frame
                if i + j + 2 < tot_frame:
                    next_frame = video_reader.frames[i + j + 2] if args.video is not None else \
                                cv2.imread(os.path.join(args.img, videogen[i + j + 1]), cv2.IMREAD_UNCHANGED)[:, :, ::-1].copy()
                    temp_frame = next_frame
                    
                    I1_tensor = torch.from_numpy(np.transpose(next_frame, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
                    I1_tensor = F.pad(I1_tensor, padding)
                    I1_tensor = model.inference(I0, I1_tensor, args.scale)
            
            batch_frames.append((I0.clone(), I1_tensor.clone(), lastframe.copy(), frame.copy()))
            lastframe = frame
            I1 = I1_tensor
        
        if batch_frames:
            processing_batches.append(batch_frames)
    
    # Process all batches in parallel
    print("Processing frames in parallel...")
    all_output_frames = parallel_processor.batch_process_frames(processing_batches)
    
    # Write all frames
    for frame in all_output_frames:
        write_buffer.put(frame)
        pbar.update(1)
    
    # Write last frame
    if args.montage:
        write_buffer.put(np.concatenate((lastframe, lastframe), 1))
    else:
        write_buffer.put(lastframe)
    pbar.update(1)
    
    # Cleanup
    write_buffer.put(None)
    pbar.close()
    
    # Wait for writer to finish
    import time
    while not write_buffer.empty():
        time.sleep(0.1)
    
    if not vid_out is None:
        vid_out.release()

    # Move audio to new video file if appropriate
    if args.png == False and 'fpsNotAssigned' in locals() and fpsNotAssigned == True and not args.video is None:
        try:
            transferAudio(args.video, vid_out_name)
        except:
            print("Audio transfer failed. Interpolated video will have no audio")
            targetNoAudio = os.path.splitext(vid_out_name)[0] + "_noaudio" + os.path.splitext(vid_out_name)[1]
            if os.path.exists(targetNoAudio):
                os.rename(targetNoAudio, vid_out_name)

if __name__ == "__main__":
    # Warm up GPU
    if torch.cuda.is_available():
        print("Warming up GPU...")
        warm_up_tensor = torch.randn(1, 3, 32, 32).to(device)
        if args.fp16:
            warm_up_tensor = warm_up_tensor.half()
        _ = model.inference(warm_up_tensor, warm_up_tensor, args.scale)
        torch.cuda.synchronize()
    
    # Run optimized processing
    process_video_optimized()
    
    print("Processing completed!")
