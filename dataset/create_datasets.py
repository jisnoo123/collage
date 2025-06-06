import subprocess
import zipfile
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
import shutil

sys.path.insert(1, '../procedure')

from manipulation import *


# bashCommand = "chmod u+rx download_dataset.bash"
# process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
# output, error = process.communicate()

# subprocess.call("./download_dataset.bash")

# print('Downloading anime dataset and imagenet is done')


# print('Extracting the zip files')

# with zipfile.ZipFile('./anime.zip', 'r') as zip_ref:
#     zip_ref.extractall('./')


# with zipfile.ZipFile('./imagenet10.zip', 'r') as zip_ref:
#     zip_ref.extractall('./')

# print('Extracting done')


#------------ Downloading external datasets done -------------- #


# # Creating final dataset for CIFAR 10

# print('Processing CIFAR 10 to make the final dataset')

# train_dataset = torchvision.datasets.CIFAR10(root='CIFAR10', train=True, download=True)
# test_dataset = torchvision.datasets.CIFAR10(root='CIIFAR10', train=False, download=True)
# train_images = [np.array(train_dataset[i][0]) for i in range(len(train_dataset))]
# test_images = [np.array(test_dataset[i][0]) for i in range(len(test_dataset))]
# full_dataset = train_images + test_images

# with open('cifar10', 'wb') as f:
#     pickle.dump(full_dataset, f)


# '''Resize and blur and avg cifar 10 and pickle it'''

# dataset = list()

# with open('cifar10', 'rb') as f:
#     dataset = pickle.load(f)

# rb_av = list()


# for i in tqdm(range(len(dataset))):
#     img = dataset[i]
#     resized_img = resize(img, 10, 10)
#     blurred_img = blur(resized_img, 3)
#     mean_pix = mean(blurred_img)
#     rb_av.append(mean_pix)


# with open('cifar10_rb_av', 'wb') as f:
#     pickle.dump(rb_av, f)

# print('Processed CIFAR10 dataset is created')


# Creating final dataset for SVHN

print('Processing SVHN to create final dataset')

train_dataset = torchvision.datasets.SVHN(root='SVHN', split='train', download=True)
test_dataset = torchvision.datasets.SVHN(root='SVHN', split='test', download=True)
train_images = [np.array(train_dataset[i][0]) for i in range(len(train_dataset))]
test_images = [np.array(test_dataset[i][0]) for i in range(len(test_dataset))]
full_dataset = train_images + test_images

with open('svhn', 'wb') as f:
    pickle.dump(full_dataset, f)


'''Resize and blur and avg svhn and pickle it'''

dataset = list()

with open('svhn', 'rb') as f:
    dataset = pickle.load(f)

rb_av = list()

print('Processing cifar 10')

for i in tqdm(range(len(dataset))):
    img = dataset[i]
    resized_img = resize(img, 10, 10)
    blurred_img = blur(resized_img, 3)
    mean_pix = mean(blurred_img)
    rb_av.append(mean_pix)


with open('svhn_rb_av', 'wb') as f:
    pickle.dump(rb_av, f)

print('Processed SVHN is created')


# Creating final dataset for imagenet 10

print('Processing ImageNet10 to create the final dataset')

dataset = list()

for folders in os.listdir('imagenet-10'):
    for image_file in os.listdir('imagenet-10/'+ folders):
        img = cv2.imread('imagenet-10/'+folders+'/'+image_file)

        if(len(img.shape)<3):
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # Resize the image 
        desired_width = 100
        aspect_ratio = desired_width / img.shape[1]
        desired_height = int(img.shape[0] * aspect_ratio)
        dim = (desired_width, desired_height)

        # Resize image
        resized_img = cv2.resize(img, dsize=dim, interpolation=cv2.INTER_AREA)

        dataset.append(resized_img)

with open('imnet10', 'wb') as f:
    pickle.dump(dataset, f)

'''Resize and blur and avg imnet10 and pickle it'''

dataset = list()

with open('imnet10', 'rb') as f:
    dataset = pickle.load(f)

rb_av = list()


for i in tqdm(range(len(dataset))):
    img = dataset[i]
    resized_img = resize(img, 10, 10)
    blurred_img = blur(resized_img, 3)
    mean_pix = mean(blurred_img)
    rb_av.append(mean_pix)


with open('imnet10_rb_av', 'wb') as f:
    pickle.dump(rb_av, f)

print('Processed ImageNet10 is created')



# Create final dataset anime

print('Processing anime dataset to create final dataset')

dataset = list()

for folders in os.listdir('data/anime_images'):
    for image_file in os.listdir('data/anime_images/'+ folders):
        img = cv2.imread('data/anime_images/'+folders+'/'+image_file)

        if(len(img.shape)<3):
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # Resize the image 
        desired_width = 100
        aspect_ratio = desired_width / img.shape[1]
        desired_height = int(img.shape[0] * aspect_ratio)
        dim = (desired_width, desired_height)

        # Resize image
        resized_img = cv2.resize(img, dsize=dim, interpolation=cv2.INTER_AREA)

        dataset.append(resized_img)

with open('anime', 'wb') as f:
    pickle.dump(dataset, f)


'''Resize and blur and avg anime and pickle it'''

dataset = list()

with open('anime', 'rb') as f:
    dataset = pickle.load(f)

rb_av = list()


for i in tqdm(range(len(dataset))):
    img = dataset[i]
    resized_img = resize(img, 10, 10)
    blurred_img = blur(resized_img, 3)
    mean_pix = mean(blurred_img)
    rb_av.append(mean_pix)


with open('anime_rb_av', 'wb') as f:
    pickle.dump(rb_av, f)

print('Processed anime dataset is created')

# ------------ Dataset creation is done --------------------#


''' Remaining Tasks: Create actual and rb_av folders and delete the zip files, extracted folders'''


# First create the folders

actual_folder = './actual'

try:
    os.makedirs(actual_folder)
except:
    pass

rb_av_folder = './rb_av'

try:
    os.makedirs(rb_av_folder)
except:
    pass


# Move the datasets to actual and rb_av respecctively

bashCommand = "mv anime cifar10 imnet10 svhn ./actual"
process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
output, error = process.communicate()


bashCommand = "mv anime_rb_av cifar10_rb_av imnet10_rb_av svhn_rb_av ./rb_av"
process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
output, error = process.communicate()

# Delete the zip files

bashCommand = "rm -rf anime.zip imagenet10.zip"
process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
output, error = process.communicate()

# Delete the extracted folders

shutil.rmtree('./data')
shutil.rmtree('./imagenet-10')

shutil.rmtree('./SVHN')
shutil.rmtree('./CIFAR10')
# --------- DATASET IS DONE ------------


print('Dataset creation is completed. You can now run the app')