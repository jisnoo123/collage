import os
import cv2 
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)

train_images = [np.array(train_dataset[i][0]) for i in range(len(train_dataset))]
test_images = [np.array(test_dataset[i][0]) for i in range(len(test_dataset))]

full_data = train_images + test_images

'''Take the image as input'''

#Load big image
img = cv2.imread('images/jimut.png', cv2.IMREAD_UNCHANGED)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
print(img_rgb.shape)
plt.imshow(img_rgb)
plt.show()



''' Resizing and blurring the CIFAR 10 dataset '''

new_height = 10
new_width = 10

after_blur_train = list()
after_blur_test = list()

for i in range(len(train_images)):
    resized_image = cv2.resize(train_images[i], (new_width, new_height))
    blurred_image = cv2.GaussianBlur(resized_image,(3,3),0)
    after_blur_train.append(blurred_image)

for i in range(len(test_images)):
    resized_image = cv2.resize(test_images[i], (new_width, new_height))
    blurred_image = cv2.GaussianBlur(resized_image,(3,3),0)
    after_blur_test.append(blurred_image)


'''Make a seperate list containing the avg pixel values of the combined dataset'''
avg_pix_train = list()
avg_pix_test = list()
for i in range(len(after_blur_train)):
    sum = 0
    for row in range(10):
        for col in range(10):
            # print('Sum:', sum)
            sum += int(after_blur_train[i][row][col][0]) + int(after_blur_train[i][row][col][1]) + int(after_blur_train[i][row][col][2])
    mean = sum/(10*10*3)
    # print('Mean:', mean)
    # input('continue?')
    avg_pix_train.append(mean)

for i in range(len(after_blur_test)):
    sum = 0
    for row in range(10):
        for col in range(10):
            sum += int(after_blur_test[i][row][col][0]) + int(after_blur_test[i][row][col][1]) + int(after_blur_test[i][row][col][2])
    mean = sum/(10*10*3)
    avg_pix_test.append(mean)

avg_pix = avg_pix_train + avg_pix_test


'''Blurring an extracted image, used for the extracted image of the big one'''
def blur_ext(cropped_img):
    blurred_img = cv2.GaussianBlur(cropped_img,(137,137),0)
    return blurred_img


'''Finding the mean of patches of the big img'''
def find_mean_img(blurred_img):
    sum = 0
    rows = blurred_img.shape[0]
    cols = blurred_img.shape[1]
    for i in range(rows):
        for j in range(cols):
            #print('Indiv pixel values of patch', blurred_img[i][j][0], blurred_img[i][j][1], blurred_img[i][j][2])
            sum += int(blurred_img[i][j][0]) + int(blurred_img[i][j][1]) + int(blurred_img[i][j][2])

    return sum/(rows*cols*3)


'''Linear search to find out the cifar10 img having min distance from the patch'''
def linear_search(mean_big_img):
    min = abs(mean_big_img - avg_pix[0])
    min_ind = 0

    for i in range(1,len(avg_pix)):
        if min > abs(mean_big_img - avg_pix[i]):
            min = abs(mean_big_img - avg_pix[i])
            min_ind = i

    return full_data[min_ind] # Return the image

'''Extract patches from the big image'''
def extract():
    m = int(input('Enter m:'))
    n = int(input('Enter n:'))

    y_start = 0
    for i in tqdm(range(1,m+1)):
        y_end = int(i*img_rgb.shape[0]/m)
        x_start = 0
        for j in range(1,n+1):
            x_end = int(j*img_rgb.shape[1]/n)

            cropped_img = img_rgb[y_start:y_end, x_start:x_end]
            blurred_img = blur_ext(cropped_img)
            mean_big_img = find_mean_img(blurred_img)

            cifar_img = linear_search(mean_big_img) #The actual cifar 10 img to be placed in place of patch

            #Resize the cifar img
            cifar_img = cv2.resize(cifar_img, (int(img_rgb.shape[1]/n),int(img_rgb.shape[0]/m)))
            
            #Replace the patch with the cifar
            img_rgb[y_start:y_start+int(img_rgb.shape[0]/m), x_start:x_start+int(img_rgb.shape[1]/n)] = cifar_img
            x_start = x_end
        y_start = y_end

extract()

'''Final image'''
print('Final image')
plt.imshow(img_rgb)
plt.show()