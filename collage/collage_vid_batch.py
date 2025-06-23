import os
import cv2 
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import argparse
import pickle
import ffmpeg
import sys
from multiprocessing import Pool, cpu_count
import functools
sys.path.insert(0, '../')
from procedure import *
import shutil

# Global variables for multiprocessing
dataset_global = None
rb_av_global = None
m_global = None
n_global = None

def init_worker(dataset, rb_av, m, n):
    """Initialize worker processes with shared data"""
    global dataset_global, rb_av_global, m_global, n_global
    dataset_global = dataset
    rb_av_global = rb_av
    m_global = m
    n_global = n

def process_frame_batch(frame_data):
    """Process a single frame - used by multiprocessing"""
    i, frame = frame_data
    blur_frame = blur(frame, 1) # Blur with a kernel size of 1
    final_frame = core_vid(blur_frame, m_global, n_global, dataset_global, rb_av_global)
    return i, final_frame

def process_frames_vectorized(frames, m, n, dataset, rb_av, batch_size=8, num_workers=None):
    """Process frames in batches using multiprocessing"""
    if num_workers is None:
        num_workers = min(cpu_count(), 8)  # Don't use too many cores
    
    FOLDER_NAME = "../op/videos/checkpoint"
    try:
        os.makedirs(FOLDER_NAME)
    except:
        pass
    
    # Prepare frame data for multiprocessing
    frame_data = [(i, frame) for i, frame in enumerate(frames)]
    
    # Use multiprocessing to process frames in parallel
    with Pool(processes=num_workers, 
              initializer=init_worker, 
              initargs=(dataset, rb_av, m, n)) as pool:
        
        # Process frames with progress bar
        results = []
        for result in tqdm(pool.imap(process_frame_batch, frame_data), 
                          total=len(frame_data), 
                          desc="Processing frames"):
            results.append(result)
    
    # Save processed frames
    for i, final_frame in results:
        cv2.imwrite(f"{FOLDER_NAME}/{i}_frame.png", final_frame)


parser = argparse.ArgumentParser()
parser.add_argument("--ip", help="Enter the video input path", required=True)
parser.add_argument("--op", help="Enter the collaged video output path", required=True)
parser.add_argument("--m", help="Enter the m of grid", required=True)
parser.add_argument("--n", help="Enter the n of grid", required=True)
parser.add_argument("--d", help="Enter c for cifar10, m for mist and s for svhn", required=True)

argparser = parser.parse_args()

dataset_choice = argparser.d
ip = argparser.ip
op = argparser.op
m = int(argparser.m)
n = int(argparser.n)

'''Loading the datasets'''

if dataset_choice == 'c':
    with open('../dataset/actual/cifar10', 'rb') as f:
        dataset = pickle.load(f)
    with open('../dataset/rb_av/cifar10_rb_av', 'rb') as f:
        rb_av = pickle.load(f)
elif dataset_choice == 's':
    with open('../dataset/actual/svhn', 'rb') as f:
        dataset = pickle.load(f)
    with open('../dataset/rb_av/svhn_rb_av', 'rb') as f:
        rb_av = pickle.load(f)
elif dataset_choice == 'i':
    with open('../dataset/actual/imnet10', 'rb') as f:
        dataset = pickle.load(f)
    with open('../dataset/rb_av/imnet10_rb_av', 'rb') as f:
        rb_av = pickle.load(f)
elif dataset_choice == 'a':
    with open('../dataset/actual/anime', 'rb') as f:
        dataset = pickle.load(f)
    with open('../dataset/rb_av/anime_rb_av', 'rb') as f:
        rb_av = pickle.load(f)

''' Loading the input video '''

cap = cv2.VideoCapture(ip)

if not cap.isOpened():
    print("Could not open video file.")
    sys.exit()
else:
    print("Video file opened successfully!")

print('Your video is under process. This may take some time.')


# Extracting sound

vid_input_file = ffmpeg.input(ip)
FOLDER_SOUND = "../ip/sounds"

try:
    os.makedirs(FOLDER_SOUND)
except:
    pass

sound_path = '../ip/sounds/sound.mp3'
vid_input_file.output(sound_path, acodec='mp3').run(overwrite_output=True, quiet=True)

ret, frame = cap.read()
height, width = frame.shape[0], frame.shape[1]

# Extracting the frames of the video

vid_frames = list()
while ret:
    vid_frames.append(frame)
    ret, frame = cap.read()

# Calculating FPS of the video

video_fps = cv2.VideoCapture(ip)
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
if int(major_ver) < 3:
    fps = video_fps.get(cv2.cv.CV_CAP_PROP_FPS)
else:
    fps = video_fps.get(cv2.CAP_PROP_FPS)
video_fps.release()

# Use batch processing 
process_frames_vectorized(vid_frames, m, n, dataset, rb_av)


print('Standby.... Your video is about to be ready')

'''Make the no sound video by merging the dumped'''

FOLDER_NAME_no_sound = "../op/no_sound_patched_video"

try:
    os.makedirs(FOLDER_NAME_no_sound)
except:
    pass

no_sound_patched_vid_path = '../op/no_sound_patched_video/no_sound_vid.mp4'

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(no_sound_patched_vid_path, fourcc, fps, (width, height))

FOLDER_NAME = "../op/videos/checkpoint"

for i in range(len(vid_frames)):
    loaded_frame = cv2.imread(FOLDER_NAME + '/' + str(i) + '_frame.png')
    video.write(loaded_frame)

video.release()
cv2.destroyAllWindows()

shutil.rmtree(FOLDER_NAME)


'''Merge the no sound video with the sound '''

input_video = ffmpeg.input(no_sound_patched_vid_path)
input_audio = ffmpeg.input(sound_path)
ffmpeg.concat(input_video, input_audio, v=1, a=1).output(op).run(overwrite_output=True, quiet=True)

shutil.rmtree(FOLDER_NAME_no_sound)
shutil.rmtree(FOLDER_SOUND)

print('Your collaged video is ready.')