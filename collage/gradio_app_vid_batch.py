import os
import cv2 
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import argparse
import pickle
import ffmpeg
import sys
sys.path.insert(0, '../')
from procedure import *
import shutil
import threading
import gradio as gr

# Global variable to track if image has been generated
image_generated = False

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
    blur_frame = blur(frame, 1) # Apply a kernel size of 1
    final_frame = core_vid(blur_frame, m_global, n_global, dataset_global, rb_av_global)
    return i, final_frame

def process_frames_vectorized(frames, m, n, dataset, rb_av, batch_size=8, num_workers=None):
    """Process frames in batches using multiprocessing"""
    if num_workers is None:
        num_workers = min(cpu_count(), 8)
    
    FOLDER_NAME = "../op/videos/checkpoint"
    try:
        os.makedirs(FOLDER_NAME)
    except:
        pass
    
    frame_data = [(i, frame) for i, frame in enumerate(frames)]
    
    with Pool(processes=num_workers, 
              initializer=init_worker, 
              initargs=(dataset, rb_av, m, n)) as pool, \
         tqdm(total=len(frame_data), desc="Processing & saving frames") as pbar:
        
        for result in pool.imap(process_frame_batch, frame_data):
            i, final_frame = result
            cv2.imwrite(f"{FOLDER_NAME}/{i}_frame.png", final_frame)
            pbar.update(1)


def collage_video(ip, d, m, n, coll_vid_file_name):
    m = int(m)
    n = int(n)

    global video_generated

    # Output path of video
    op = '../op/videos/' + coll_vid_file_name + '.mp4'

    '''Loading the datasets'''

    if d == 'CIFAR_10':
        # Load the CIFAR10 dataset
        with open('../dataset/actual/cifar10', 'rb') as f:
            dataset = pickle.load(f)
        # Load the cifar10 dataset by unpickling the pickled data
        with open('../dataset/rb_av/cifar10_rb_av', 'rb') as f:
            rb_av = pickle.load(f)
    elif d == 'SVHN':
        # Load the SVHN dataset
        with open('../dataset/actual/svhn', 'rb') as f:
            dataset = pickle.load(f)
        # Load the svhn dataset by unpickling the pickled data
        with open('../dataset/rb_av/svhn_rb_av', 'rb') as f:
            rb_av = pickle.load(f)
    elif d == 'ImageNet10':
        # Load the ImageNet10 dataset
        with open('../dataset/actual/imnet10', 'rb') as f:
            dataset = pickle.load(f)
        # Load the ImageNet10 dataset by unpickling the pickled data
        with open('../dataset/rb_av/imnet10_rb_av', 'rb') as f:
            rb_av = pickle.load(f)
    elif d == 'Anime':
        # Load the Anime dataset
        with open('../dataset/actual/anime', 'rb') as f:
            dataset = pickle.load(f)
        # Load the Anime dataset by unpickling the pickled data
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

    # Mark that image has been generated and start monitoring for close prompt
    video_generated = True
    threading.Thread(target=monitor_close_prompt, daemon=True).start()

    return op


def monitor_close_prompt():
    """Monitor for user input to close server"""
    print("\nPress 'y' to close the server: ", end='', flush=True)
    user_input = input().strip().lower()
    if user_input == 'y':
        print("Closing server...")
        os._exit(0)


with gr.Blocks() as demo:
    with gr.Row(equal_height = True):
        ip = gr.Video(label = 'input video',  height=400, width = 300)
        with gr.Column():
            d = gr.Radio(['Anime', 'ImageNet10', 'CIFAR_10', 'SVHN'], label="Dataset")
            m = gr.Textbox(label = 'm')
            n = gr.Textbox(label = 'n')
            coll_vid_file_name = gr.Textbox(label = 'Enter collaged video filename [To be saved in collage/dataset/op/videos]')
    generate_btn = gr.Button('Generate')

    output = gr.Video(label='Collaged video', height=400, width=600)

    generate_btn.click(fn = collage_video, inputs = [ip, d, m, n, coll_vid_file_name], outputs = output)

OUTPUT_FOLDER =  os.path.abspath('../op/videos')
demo.launch(allowed_paths=[OUTPUT_FOLDER])