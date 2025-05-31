import os
import cv2 
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import pickle
import zipfile
from PIL import Image
from io import BytesIO
import sys
from tqdm import tqdm

sys.path.insert(1, '../procedure')

from manipulation import *

'''For calculating the average pixel of each blurred and resized image'''

def average_pixel(dataset):
    new_height = 10
    new_width = 10
    avg_px_rbdataset = list()

    for i in tqdm(range(len(dataset))):
        resized_img = resize_img(dataset[i], new_height, new_width)
        blurred_img = blur(dataset[i], 3)   #Using a kernel size of (3,3) on the image
        try:
            mean_rb = mean(dataset[i])
        except:
            print('This needs to be shown')
            print(dataset[i])
            print('continue?')
        avg_px_rbdataset.append(mean_rb)
    return avg_px_rbdataset


########## CIFAR10 ##########

#Load the unpickled CIFAR10 dataset

with open('cifar10', 'rb') as f:
    dataset = pickle.load(f)

avg_px_rbdataset = average_pixel(dataset)

print('........Pickling cifar10_rb_av........')
#Save the pixckled version in a file cifar10_rb
with open('cifar10_rb_av', 'wb') as f:
    pickle.dump(avg_px_rbdataset, f)
print('........Completed pickling cifar10_rb_av........')


########## MNIST ##########

#Load the unpickled MNIST RGB dataset

with open('mnist_rgb', 'rb') as f:
    dataset = pickle.load(f)

avg_px_rbdataset = average_pixel(dataset)

print('........Pickling mnist_rb_av........')
#Save the pickled version in a file mnist_rgb_av
with open('mnist_rb_av', 'wb') as f:
    pickle.dump(avg_px_rbdataset, f)
print('........Completed pickling mnist_rb_av........')

########## SVHN ##########

#Load the unpickled SVHN dataset

with open('svhn', 'rb') as f:
    dataset = pickle.load(f)

avg_px_rbdataset = average_pixel(dataset)

print('........Pickling svhn_rb_av........')
#Save the pixckled version in a file svhn_rb
with open('svhn_rb_av', 'wb') as f:
    pickle.dump(avg_px_rbdataset, f)

print('........Completed pickling svhn_rb_av........')