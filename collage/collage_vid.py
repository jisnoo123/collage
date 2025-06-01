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
# parser.add_argument("--m", help="Enter the m of grid", required=True)
# parser.add_argument("--n", help="Enter the n of grid", required=True)
# parser.add_argument("--d", help="Enter c for cifar10, m for mist and s for svhn", 
#                                                                     required=True)


argparser = parser.parse_args()

# dataset_choice = argparser.d
ip = argparser.ip
# # op = argparser.op
# m = int(argparser.m)
# n = int(argparser.n)

# '''Loading the datasets'''

# if dataset_choice == 'c':
#     # Load the CIFAR10 dataset
#     with open('cifar10', 'rb') as f:
#         dataset = pickle.load(f)
#     # Load the cifar10 dataset by unpickling the pickled data
#     with open('cifar10_rb_av', 'rb') as f:
#         rb_av = pickle.load(f)
# elif dataset_choice == 'm':
#     # Load the MNIST dataset
#     with open('cifar10', 'rb') as f:
#         dataset = pickle.load(f)
#     # Load the mnist dataset by unpickling the pickled data
#     with open('mnist_rb_av', 'rb') as f:
#         rb_av = pickle.load(f)
# elif dataset_choice == 's':
#     # Load the SVHN dataset
#     with open('svhn_rb', 'rb') as f:
#         dataset = pickle.load(f)
#     # Load the svhn dataset by unpickling the pickled data
#     with open('svhn_rb_av', 'rb') as f:
#         rb_av = pickle.load(f)


''' Loading the input video '''

cap = cv2.VideoCapture(ip)

# Check if the video was opened successfully
if not cap.isOpened():
    print("Could not open video file.")
else:
    print("Video file opened successfully!")

cap = cv2.VideoCapture(ip)

ret, frame = cap.read()

'''Extracting the frames and keeping them in a list'''

vid_frames = list()

while ret:
    vid_frames.append(frame)
    ret, frame = cap.read() #Read the next frame

