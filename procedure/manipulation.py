import cv2
from tqdm import tqdm


def resize_img(img, ht, wd):
    '''Resizes an image into (ht*wd)'''
    resized_img = cv2.resize(img, (ht,wd))
    return resized_img

def blur(img, kernel_size):
    '''Applies Gaussian Blur to an image img of kernel_size'''
    blurred_img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    return blurred_img

def mean(img):
    '''Finds the mean of the pixels of the image of 3 channels'''
    sum = 0
    m = img.shape[0]
    n = img.shape[1]
    
    for i in range(m):
        for j in range(n):
            sum += int(img[i][j][0]) + int(img[i][j][1]) + int(img[i][j][2])
    
    return sum/(m*n*3) # The mean

def linear_search(mean_big_patch, avg_px):
    '''
    Arguments:
        mean_big_patch: Mean of the bigger image's patch
        avg_px: A list containing the values of the avergae pixels of the dataset

    Returns the index of the min img
    '''
    min = abs(mean_big_patch - avg_px[0]) # Min distance
    min_ind = 0                           # Min index

    # Finding the min img from the dataset
    for i in range(1,len(avg_px)):
        if min > abs(mean_big_patch - avg_px[i]):
            min = abs(mean_big_patch - avg_px[i])
            min_ind = i

    return min_ind          # Return the index of the min image


def core(big_img, m, n, fds, rb_av):
    '''
    This function does the whole job, hence the name : core. It returns the new collaged image.

    It extracts patches from the big image having grids mxn, finds the 
    min img from the dataset and replaces the patches.

    Finally it returns the final collaged image    
    '''
    big_img_cpy = big_img.copy()

    y_start = 0
    for i in tqdm(range(1,m+1)):
        y_end = int(i*big_img_cpy.shape[0]/m)
        x_start = 0
        for j in range(1,n+1):
            x_end = int(j*big_img_cpy.shape[1]/n)

            cropped_img = big_img_cpy[y_start:y_end, x_start:x_end]
            blurred_img = blur(cropped_img, 3)
            mean_big_img = mean(blurred_img)

            dataset_img_ind = linear_search(mean_big_img, rb_av)   #Dataset index of img to be replaced

            dataset_img = fds[dataset_img_ind]   #The actual dataset img to be placed in place of patch
            #Resize the dataset img
            dataset_img = cv2.resize(dataset_img, (int(big_img_cpy.shape[1]/n),int(big_img_cpy.shape[0]/m)))
            
            #Replace the patch with the dataset img retrieved
            big_img_cpy[y_start:y_start+int(big_img_cpy.shape[0]/m), x_start:x_start+int(big_img_cpy.shape[1]/n)] = dataset_img
            x_start = x_end
        y_start = y_end

    return big_img_cpy

