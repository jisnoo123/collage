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

import sys
sys.path.insert(0, '../')
from procedure import *

parser = argparse.ArgumentParser()

parser.add_argument("--ip", help="Enter the big image input path", required=True)
parser.add_argument("--m", help="Enter the m of grid", required=True)
parser.add_argument("--n", help="Enter the n of grid", required=True)
parser.add_argument("--d", help="Enter c for cifar10, m for mist and s for svhn", 
                                                                    required=True)


argparser = parser.parse_args()

dataset_choice = argparser.d
ip = argparser.ip
# op = argparser.op
m = int(argparser.m)
n = int(argparser.n)


'''Loading the datasets'''

if dataset_choice == 'c':
    # Load the CIFAR10 dataset
    with open('../dataset/actual/cifar10', 'rb') as f:
        dataset = pickle.load(f)
    # Load the cifar10 dataset by unpickling the pickled data
    with open('../dataset/rb_av/cifar10_rb_av', 'rb') as f:
        rb_av = pickle.load(f)
elif dataset_choice == 's':
    # Load the SVHN dataset
    with open('../dataset/actual/svhn', 'rb') as f:
        dataset = pickle.load(f)
    # Load the svhn dataset by unpickling the pickled data
    with open('../dataset/rb_av/svhn_rb_av', 'rb') as f:
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

vid_frames = list()

while ret:
    vid_frames.append(frame)
    ret, frame = cap.read() #Read the next frame



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



for i in range(len(vid_frames)):
    frame = vid_frames[i]

    blur_frame = blur(frame, 151) #Applying a kernel size of 151

    final_frame = core(blur_frame, m, n, dataset, rb_av)

    cv2.imwrite(FOLDER_NAME + '/' + str(i) + '_frame.png', final_frame)


# Merging the dumped frames to a video

for i in range(len(vid_frames)):
    loaded_frame = cv2.imread('./'+ FOLDER_NAME + '/' + str(i) + '_frame.png')
    video.write(frame)

# Release the video writer and close the video file
video.release()
cv2.destroyAllWindows()