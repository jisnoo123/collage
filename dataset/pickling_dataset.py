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

sys.path.insert(1, '../procedure')

from manipulation import *

'''

Load CIFAR10 and pickle it in cifar10 file as well as resize and 
blur and pickle them in cifar10_rb

'''
train_dataset = torchvision.datasets.CIFAR10(root='CIFAR_10', train=True, download=True)
test_dataset = torchvision.datasets.CIFAR10(root='CIFAR_10', train=False, download=True)
train_images = [np.array(train_dataset[i][0]) for i in range(len(train_dataset))]
test_images = [np.array(test_dataset[i][0]) for i in range(len(test_dataset))]
full_dataset = train_images + test_images

with open('cifar10', 'wb') as f:
    pickle.dump(full_dataset, f)

print('.....Pickling cifar 10 done .....')

'''Load MNIST and pickle it in mnist file'''
train_dataset = torchvision.datasets.MNIST(root='MNIST', train=True, download=True)
test_dataset = torchvision.datasets.MNIST(root='MNIST', train=False, download=True)
train_images = [np.array(train_dataset[i][0]) for i in range(len(train_dataset))]
test_images = [np.array(test_dataset[i][0]) for i in range(len(test_dataset))]

full_dataset = train_images + test_images


# Convert the gray scale image to 3 channel
with open('mnist', 'wb') as f:
    pickle.dump(full_dataset, f)

with open('mnist', 'rb') as f:
    full_dataset = pickle.load(f)
print('Converting grayscale MNIST to rgb')

rgb_mnist_dataset = list()
for i in tqdm(range(len(full_dataset))):
    rgb_mnist_img = cv2.cvtColor(full_dataset[i],cv2.COLOR_GRAY2RGB)
    rgb_mnist_dataset.append(rgb_mnist_img)

print('Conversion to rgb done')

with open('mnist_rgb', 'wb') as f:
    pickle.dump(rgb_mnist_dataset, f)

print('.....Pickling mnist done .....')


'''Load SVHN and pickle it in svhn file'''
train_dataset = torchvision.datasets.SVHN(root='SVHN', split='train', download=True)
test_dataset = torchvision.datasets.SVHN(root='SVHN', split='test', download=True)
train_images = [np.array(train_dataset[i][0]) for i in range(len(train_dataset))]
test_images = [np.array(test_dataset[i][0]) for i in range(len(test_dataset))]
full_dataset = train_images + test_images

with open('svhn', 'wb') as f:
    pickle.dump(full_dataset, f)

print('.....Pickling svhn done .....')

