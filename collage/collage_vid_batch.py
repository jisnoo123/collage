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
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import time

import sys
sys.path.insert(0, '../')
from procedure import *

parser = argparse.ArgumentParser()

parser.add_argument("--ip", help="Enter the big image input path", required=True)
parser.add_argument("--m", help="Enter the m of grid", required=True)
parser.add_argument("--n", help="Enter the n of grid", required=True)
parser.add_argument("--d", help="Enter c for cifar10, m for mist and s for svhn", 
                                                                    required=True)
parser.add_argument("--workers", help="Number of worker processes (default: CPU count)", 
                    type=int, default=cpu_count())
parser.add_argument("--batch_size", help="Batch size for processing (default: 4)", 
                    type=int, default=4)

argparser = parser.parse_args()

dataset_choice = argparser.d
ip = argparser.ip
m = int(argparser.m)
n = int(argparser.n)
workers = argparser.workers
batch_size = argparser.batch_size

print(f"Using {workers} workers with batch size {batch_size}")

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

cap = cv2.VideoCapture(ip)

ret, frame = cap.read() # Read the first frame

height, width = frame.shape[0], frame.shape[1]

'''Extracting the frames and keeping them in a list'''

print("Extracting frames...")
vid_frames = list()

while ret:
    vid_frames.append(frame.copy())  # Use copy() to avoid memory issues
    ret, frame = cap.read() #Read the next frame

cap.release()  # Release video capture early

print(f"Extracted {len(vid_frames)} frames")

'''Converting frames to dataset patches'''

fps = 34
output_file = '../op/output_changed.mp4'
# Create a VideoWriter object to save the video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Specify the codec for the output video file

# Creating a folder to dump the converted images
FOLDER_NAME = "checkpoint"

try:
    os.makedirs(FOLDER_NAME)
except:
    pass

# Optimized function to process multiple frames in a batch
def process_frame_batch(batch_args):
    batch_indices, batch_frames, m, n, dataset, rb_av, folder_name = batch_args
    results = []
    
    for i, frame in zip(batch_indices, batch_frames):
        try:
            blur_frame = blur(frame, 151) #Applying a kernel size of 151
            final_frame = core(blur_frame, m, n, dataset, rb_av)
            
            # Save with higher compression for faster I/O
            cv2.imwrite(folder_name + '/' + str(i) + '_frame.png', final_frame, 
                       [cv2.IMWRITE_PNG_COMPRESSION, 1])  # Fast compression
            results.append(i)
        except Exception as e:
            print(f"Error processing frame {i}: {e}")
            results.append(None)
    
    return results

# Function to process a single frame (fallback)
def process_frame(args):
    i, frame, m, n, dataset, rb_av, folder_name = args
    
    try:
        blur_frame = blur(frame, 151) #Applying a kernel size of 151
        final_frame = core(blur_frame, m, n, dataset, rb_av)
        cv2.imwrite(folder_name + '/' + str(i) + '_frame.png', final_frame,
                   [cv2.IMWRITE_PNG_COMPRESSION, 1])  # Fast compression
        return i
    except Exception as e:
        print(f"Error processing frame {i}: {e}")
        return None

# Create batches for processing
def create_batches(vid_frames, batch_size):
    batches = []
    for i in range(0, len(vid_frames), batch_size):
        batch_indices = list(range(i, min(i + batch_size, len(vid_frames))))
        batch_frames = [vid_frames[j] for j in batch_indices]
        batches.append((batch_indices, batch_frames, m, n, dataset, rb_av, FOLDER_NAME))
    return batches

# Process frames in parallel with batching
print(f"Processing {len(vid_frames)} frames in parallel with batching...")
start_time = time.time()

if batch_size > 1:
    # Use batched processing for better efficiency
    batches = create_batches(vid_frames, batch_size)
    
    with ProcessPoolExecutor(max_workers=workers) as executor:
        # Submit all batches
        future_to_batch = {executor.submit(process_frame_batch, batch): batch for batch in batches}
        
        completed_frames = 0
        total_frames = len(vid_frames)
        
        with tqdm(total=total_frames, desc="Processing frames", unit="frame") as pbar:
            for future in as_completed(future_to_batch):
                try:
                    results = future.result()
                    completed_frames += len([r for r in results if r is not None])
                    pbar.update(len(results))
                except Exception as e:
                    print(f"Batch processing error: {e}")
else:
    # Use single frame processing
    frame_args = [(i, vid_frames[i], m, n, dataset, rb_av, FOLDER_NAME) for i in range(len(vid_frames))]
    
    with ProcessPoolExecutor(max_workers=workers) as executor:
        list(tqdm(executor.map(process_frame, frame_args), 
                  total=len(vid_frames),
                  desc="Processing frames",
                  unit="frame"))

processing_time = time.time() - start_time
print(f"Frame processing completed in {processing_time:.2f} seconds")

# Optimized video writing with threading
print("Creating output video...")
video = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

def load_frame(i):
    return i, cv2.imread('./'+ FOLDER_NAME + '/' + str(i) + '_frame.png')

# Use threading for I/O operations (reading processed frames)
start_time = time.time()
frames_dict = {}

with ThreadPoolExecutor(max_workers=min(8, len(vid_frames))) as executor:
    # Submit all frame loading tasks
    future_to_index = {executor.submit(load_frame, i): i for i in range(len(vid_frames))}
    
    with tqdm(total=len(vid_frames), desc="Loading frames", unit="frame") as pbar:
        for future in as_completed(future_to_index):
            try:
                index, loaded_frame = future.result()
                frames_dict[index] = loaded_frame
                pbar.update(1)
            except Exception as e:
                print(f"Error loading frame: {e}")

# Write frames in order
print("Writing video file...")
for i in tqdm(range(len(vid_frames)), desc="Writing video", unit="frame"):
    if i in frames_dict and frames_dict[i] is not None:
        video.write(frames_dict[i])
    else:
        print(f"Warning: Frame {i} is missing or corrupted")

video_time = time.time() - start_time
print(f"Video creation completed in {video_time:.2f} seconds")

total_time = processing_time + video_time
print(f"Total processing time: {total_time:.2f} seconds")
print(f"Average speed: {len(vid_frames)/total_time:.2f} frames/second")

# Release the video writer and close the video file
video.release()
cv2.destroyAllWindows()

# Optional: Clean up processed frames to save disk space
cleanup = input("Do you want to delete processed frame files? (y/n): ")
if cleanup.lower() == 'y':
    import shutil
    shutil.rmtree(FOLDER_NAME)
    print("Processed frames cleaned up.")