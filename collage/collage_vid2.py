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
from multiprocessing import Pool, cpu_count
from functools import partial

import sys
sys.path.insert(0, '../')
from procedure import *

def process_frame(frame_data, m, n, dataset, rb_av, FOLDER_NAME):
    """Process a single frame"""
    i, frame = frame_data
    
    blur_frame = blur(frame, 151)  # Applying a kernel size of 151
    final_frame = core(blur_frame, m, n, dataset, rb_av)
    
    # Fixed the cv2.imwrite call
    cv2.imwrite('./' + FOLDER_NAME + '/' + str(i) + '_frame.jpg', final_frame)
    
    return final_frame

parser = argparse.ArgumentParser()

parser.add_argument("--ip", help="Enter the big image input path", required=True)
parser.add_argument("--m", help="Enter the m of grid", required=True)
parser.add_argument("--n", help="Enter the n of grid", required=True)
parser.add_argument("--d", help="Enter c for cifar10, m for mist and s for svhn", 
                                                                    required=True)

argparser = parser.parse_args()

dataset_choice = argparser.d
ip = argparser.ip
m = int(argparser.m)
n = int(argparser.n)

'''Loading the datasets'''

if dataset_choice == 'c':
    # Load the CIFAR10 dataset
    with open('cifar10', 'rb') as f:
        dataset = pickle.load(f)
    # Load the cifar10 dataset by unpickling the pickled data
    with open('cifar10_rb_av', 'rb') as f:
        rb_av = pickle.load(f)
elif dataset_choice == 'm':
    # Load the MNIST dataset
    with open('cifar10', 'rb') as f:
        dataset = pickle.load(f)
    # Load the mnist dataset by unpickling the pickled data
    with open('mnist_rb_av', 'rb') as f:
        rb_av = pickle.load(f)
elif dataset_choice == 's':
    # Load the SVHN dataset
    with open('svhn_rb', 'rb') as f:
        dataset = pickle.load(f)
    # Load the svhn dataset by unpickling the pickled data
    with open('svhn_rb_av', 'rb') as f:
        rb_av = pickle.load(f)

''' Loading the input video '''

cap = cv2.VideoCapture(ip)

# Check if the video was opened successfully
if not cap.isOpened():
    print("Could not open video file.")
else:
    print("Video file opened successfully!")

ret, frame = cap.read() # Read the first frame
height, width = frame.shape[0], frame.shape[1]

'''Extracting the frames and keeping them in a list'''

vid_frames = list()

while ret:
    vid_frames.append(frame.copy())  # Make a copy to avoid reference issues
    ret, frame = cap.read() #Read the next frame

cap.release()  # Release the video capture

'''Converting frames to dataset patches'''

fps = 34
output_file = '../op/output_changed.mp4'
# Create a VideoWriter object to save the video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Specify the codec for the output video file
video = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

# Creating a folder to dump the converted images
FOLDER_NAME = "checkpoint"

try:
    os.makedirs(FOLDER_NAME)
except:
    pass

# Parallel processing
if __name__ == '__main__':
    # Create frame data tuples (index, frame)
    frame_data = [(i, vid_frames[i]) for i in range(len(vid_frames))]
    
    # Create a partial function with fixed parameters
    process_func = partial(process_frame, m=m, n=n, dataset=dataset, rb_av=rb_av, FOLDER_NAME=FOLDER_NAME)
    
    # Use multiprocessing Pool for parallel processing
    num_processes = min(cpu_count(), len(vid_frames))  # Use available CPUs or frame count, whichever is smaller
    
    print(f"Processing {len(vid_frames)} frames using {num_processes} processes...")
    
    with Pool(processes=num_processes) as pool:
        processed_frames = list(tqdm(pool.imap(process_func, frame_data), total=len(frame_data)))
    
    # Write processed frames to video in order
    for final_frame in processed_frames:
        video.write(final_frame)
    
    video.release()
    print("Video processing completed!")