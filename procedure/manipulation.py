import cv2
import numpy as np
from tqdm import tqdm

def mean(img):
    '''Vectorized version - Finds the mean of the pixels of the image of 3 channels'''
    return np.mean(img)

def resize(img, new_ht, new_wt):
    # Resize an image to new_ht and new_wt
    resized_img = cv2.resize(img, (new_ht, new_wt))
    return resized_img

def blur(img, kernel_size):
    '''Applies Gaussian Blur to an image img of kernel_size'''
    blurred_img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    return blurred_img

def linear_search(mean_big_patch, avg_px):
    '''Returns the index of the image in the dataset having nearest pixel intensity
       to the patch'''
    
    distances = np.abs(np.array(avg_px) - mean_big_patch)
    return np.argmin(distances)

def core_img(big_img, m, n, fds, rb_av):
    '''
    Vectorized version of core function with minimal changes.
    '''
    big_img_cpy = big_img.copy()
    # rb_av = np.array(rb_av)  # Convert to numpy array for vectorized operations

    y_start = 0
    for i in tqdm(range(1,m+1)):
        y_end = int(i*big_img_cpy.shape[0]/m)
        x_start = 0
        for j in range(1,n+1):
            x_end = int(j*big_img_cpy.shape[1]/n)

            cropped_img = big_img_cpy[y_start:y_end, x_start:x_end]
            blurred_img = blur(cropped_img, 3)
            mean_big_img = mean(blurred_img)

            dataset_img_ind = linear_search(mean_big_img, rb_av)

            dataset_img = fds[dataset_img_ind]

            # Resize the image to fit in the patch
            dataset_img = cv2.resize(dataset_img, (int(big_img_cpy.shape[1]/n),int(big_img_cpy.shape[0]/m)))
            
            # Replace the dataset image in the patch
            big_img_cpy[y_start:y_start+int(big_img_cpy.shape[0]/m), x_start:x_start+int(big_img_cpy.shape[1]/n)] = dataset_img
            x_start = x_end
        y_start = y_end

    return big_img_cpy

def core_vid(big_img, m, n, fds, rb_av):
    '''
    Vectorized version of core function with minimal changes.
    '''
    big_img_cpy = big_img.copy()
    # rb_av = np.array(rb_av)  # Convert to numpy array for vectorized operations

    y_start = 0
    for i in range(1,m+1):
        y_end = int(i*big_img_cpy.shape[0]/m)
        x_start = 0
        for j in range(1,n+1):
            x_end = int(j*big_img_cpy.shape[1]/n)

            cropped_img = big_img_cpy[y_start:y_end, x_start:x_end]
            blurred_img = blur(cropped_img, 3)
            mean_big_img = mean(blurred_img)

            dataset_img_ind = linear_search(mean_big_img, rb_av)

            dataset_img = fds[dataset_img_ind]

            # Resize the image to fit in the patch
            dataset_img = cv2.resize(dataset_img, (int(big_img_cpy.shape[1]/n),int(big_img_cpy.shape[0]/m)))
            
            # Replace the dataset image in the patch
            big_img_cpy[y_start:y_start+int(big_img_cpy.shape[0]/m), x_start:x_start+int(big_img_cpy.shape[1]/n)] = dataset_img
            x_start = x_end
        y_start = y_end

    return big_img_cpy