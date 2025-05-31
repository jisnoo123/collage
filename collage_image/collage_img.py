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

# parser.add_argument("--ip", help="Enter the big image input path", required=True)
# parser.add_argument("--m", help="Enter the m of grid", required=True)
# parser.add_argument("--n", help="Enter the n of grid", required=True)
parser.add_argument("--d", help="Enter c for cifar10, m for mist and s for svhn", 
                                                                    required=True)

argparser = parser.parse_args()

dataset_choice = argparser.d

'''Loading the datasets'''

if dataset_choice == 'c':
    #Load the cifar10 dataset by unpickling the pickled data
    with open('cifar10', 'rb') as f:
        dataset = pickle.load(f)
elif dataset_choice == 'm':
    #Load the mnist dataset by unpickling the pickled data
    with open('mnist', 'rb') as f:
        dataset = pickle.load(f)
elif dataset_choice == 's':
    #Load the svhn dataset by unpickling the pickled data
    with open('svhn', 'rb') as f:
        dataset = pickle.load(f)

'''Store them in lists where each element is a numpy array representing the images'''

