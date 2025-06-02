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
sys.path.insert(0, '../')
from procedure import *
import shutil

parser = argparse.ArgumentParser()

parser.add_argument("--ip", help="Enter the video input path", required=True)
parser.add_argument("--op", help="Enter the collaged video output path", required=True)
parser.add_argument("--m", help="Enter the m of grid", required=True)
parser.add_argument("--n", help="Enter the n of grid", required=True)
parser.add_argument("--d", help="Enter c for cifar10, m for mist and s for svhn", 
                                                                    required=True)


argparser = parser.parse_args()

dataset_choice = argparser.d
ip = argparser.ip
op = argparser.op
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
    sys.exit()
else:
    print("Video file opened successfully!")

print('Your video is under process. This may take some time.')

# Extract sound of the video 

# Load the video file
vid_input_file = ffmpeg.input(ip)

FOLDER_SOUND = "../ip/sounds"

try:
    os.makedirs(FOLDER_SOUND)
except:
    pass

# Extract the audio and save it as an MP3 file
sound_path = '../ip/sounds/sound.mp3'

vid_input_file.output(sound_path, acodec='mp3').run(overwrite_output = True, quiet = True)


ret, frame = cap.read() # Read the first frame

height, width = frame.shape[0], frame.shape[1]

'''Extracting the frames and keeping them in a list'''

vid_frames = list()

while ret:
    vid_frames.append(frame)
    ret, frame = cap.read() #Read the next frame



'''Converting frames to dataset patches'''

# Find fps of the video
video_fps = cv2.VideoCapture(ip)
 
# Find OpenCV version
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

if int(major_ver)  < 3:
    fps = video_fps.get(cv2.cv.CV_CAP_PROP_FPS)
else:
    fps = video_fps.get(cv2.CAP_PROP_FPS)

video_fps.release()


# Creating a folder to dump the converted images

FOLDER_NAME = "../op/videos/checkpoint"

try:
    os.makedirs(FOLDER_NAME)
except:
    pass

for i in tqdm(range(len(vid_frames))):
    frame = vid_frames[i]

    blur_frame = blur(frame, 151) #Applying a kernel size of 151

    final_frame = core_vid(blur_frame, m, n, dataset, rb_av)

    cv2.imwrite(FOLDER_NAME + '/' + str(i) + '_frame.png', final_frame)


# Merging the dumped frames to a video

# Create a VideoWriter object to save the video

FOLDER_NAME_no_sound = "../op/no_sound_patched_video"

try:
    os.makedirs(FOLDER_NAME_no_sound)
except:
    pass

no_sound_patched_vid_path = '../op/no_sound_patched_video/no_sound_vid.mp4'

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Specify the codec for the output video file
video = cv2.VideoWriter(no_sound_patched_vid_path, fourcc, fps, (width, height))

for i in range(len(vid_frames)):
    loaded_frame = cv2.imread(FOLDER_NAME + '/' + str(i) + '_frame.png')
    video.write(loaded_frame)

# Release the video writer and close the video file
video.release()
cv2.destroyAllWindows()


# Remove the checkpoint folder

shutil.rmtree(FOLDER_NAME)

# Merge the video and the sound

input_video = ffmpeg.input(no_sound_patched_vid_path)

input_audio = ffmpeg.input(sound_path)

ffmpeg.concat(input_video, input_audio, v=1, a=1).output(op).run(overwrite_output = True, quiet = True)

shutil.rmtree(FOLDER_NAME_no_sound)
shutil.rmtree(FOLDER_SOUND)

print('Your collaged video is ready.')