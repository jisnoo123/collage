import os
import cv2 
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import argparse
import sys
sys.path.insert(0, '../')
from procedure import *

parser = argparse.ArgumentParser()

# parser.add_argument("--ip", help="Enter the big image input path", required=True)
# parser.add_argument("--m", help="Enter the m of grid", required=True)
# parser.add_argument("--n", help="Enter the n of grid", required=True)
parser.add_argument("--d", help="Enter c for cifar10, m for mist, s for svhn or i for imagenet10", 
                                                                                    required=True)

argparser = parser.parse_args()

dataset_choice = argparser.d

'''Loading the datasets'''

if dataset_choice == 'c':
    #Load the cifar10 dataset
    train_dataset = torchvision.datasets.CIFAR10(root='../dataset', train=True, download=True)
    test_dataset = torchvision.datasets.CIFAR10(root='../dataset', train=False, download=True)
elif dataset_choice == 'i':
    #Load the ImageNet1k dataset
    train_dataset = torchvision.datasets.ImageNet(root='../dataset/ImageNet1k', split='train')
    test_dataset = torchvision.datasets.ImageNet(root='../dataset/ImageNet1k', split='val')
elif dataset_choice == 'm':
    #Load MNIST
    train_dataset = torchvision.datasets.MNIST(root='../dataset', train=True, download=True)
    test_dataset = torchvision.datasets.MNIST(root='../dataset', train=False, download=True)
elif dataset_choice == 's':
    #Load SVHN
    train_dataset = torchvision.datasets.SVHN(root='../dataset', split='train', download=True)
    test_dataset = torchvision.datasets.SVHN(root='../dataset', split='test', download=True)

'''Store them in lists where each element is a numpy array representing the images'''

train_images = [np.array(train_dataset[i][0]) for i in range(len(train_dataset))]
test_images = [np.array(test_dataset[i][0]) for i in range(len(test_dataset))]

plt.imshow(train_images[0])
plt.show()