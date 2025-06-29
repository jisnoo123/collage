import cv2
import numpy as np
from tqdm import tqdm

def mean(img):
    return tuple(np.mean(np.array(img).reshape(-1, 3), axis=0))

def resize(img, new_ht, new_wt):
    # Resize an image to new_ht and new_wt
    resized_img = cv2.resize(img, (new_ht, new_wt))
    return resized_img

def blur(img, kernel_size):
    '''Applies Gaussian Blur to an image img of kernel_size'''
    blurred_img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    return blurred_img

def linear_search(mean_big_patch, avg_px):
    return np.argmin(np.sum(np.abs(np.array(avg_px) - mean_big_patch), axis=1))


def core_img(big_img, m, n, fds, rb_av, dataset_choice):
    '''
    Enhanced version that preserves original dataset image quality
    by creating a new array with full-sized images instead of resizing.
    '''
    big_img_cpy = big_img.copy()
    
    if dataset_choice == 'CIFAR_10' or dataset_choice == 'SVHN':
        # Get dimensions of dataset images (assuming all are same size)
        dataset_img_height, dataset_img_width = 32, 32
        
        # Calculate the new dimensions for the output image
        new_height = m * dataset_img_height
        new_width = n * dataset_img_width
        
        # Create new array to hold the full-resolution collage
        if len(big_img.shape) == 3:  # Color image
            collage_img = np.zeros((new_height, new_width, big_img.shape[2]), dtype=big_img.dtype)
        else:  # Grayscale image
            collage_img = np.zeros((new_height, new_width), dtype=big_img.dtype)
        
        # Process patches and place full-sized dataset images
        y_start = 0
        for i in tqdm(range(1, m+1)):
            y_end = int(i * big_img_cpy.shape[0] / m)
            x_start = 0
            
            for j in range(1, n+1):
                x_end = int(j * big_img_cpy.shape[1] / n)
                
                # Extract patch from original image for analysis
                cropped_img = big_img_cpy[y_start:y_end, x_start:x_end]
                blurred_img = blur(cropped_img, 3)
                mean_big_img = mean(blurred_img)
                
                # Find best matching dataset image
                dataset_img_ind = linear_search(mean_big_img, rb_av)
                dataset_img = fds[dataset_img_ind]
                
                # Calculate position in the new collage array
                collage_y_start = (i-1) * dataset_img_height
                collage_y_end = i * dataset_img_height
                collage_x_start = (j-1) * dataset_img_width
                collage_x_end = j * dataset_img_width
                
                # Place the full-sized dataset image in the collage
                collage_img[collage_y_start:collage_y_end, 
                        collage_x_start:collage_x_end] = dataset_img
                
                x_start = x_end
            y_start = y_end

        return collage_img
    else:
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

# def core_img(big_img, m, n, fds, rb_av):
#     '''
#     Vectorized version of core function with minimal changes.
#     '''
#     big_img_cpy = big_img.copy()
#     # rb_av = np.array(rb_av)  # Convert to numpy array for vectorized operations


#     y_start = 0
#     for i in tqdm(range(1,m+1)):
#         y_end = int(i*big_img_cpy.shape[0]/m)
#         x_start = 0
#         for j in range(1,n+1):
#             x_end = int(j*big_img_cpy.shape[1]/n)

#             cropped_img = big_img_cpy[y_start:y_end, x_start:x_end]
#             blurred_img = blur(cropped_img, 3)
#             mean_big_img = mean(blurred_img)

#             dataset_img_ind = linear_search(mean_big_img, rb_av)

#             dataset_img = fds[dataset_img_ind]

#             # Resize the image to fit in the patch
#             dataset_img = cv2.resize(dataset_img, (int(big_img_cpy.shape[1]/n),int(big_img_cpy.shape[0]/m)))
            
#             # Replace the dataset image in the patch
#             big_img_cpy[y_start:y_start+int(big_img_cpy.shape[0]/m), x_start:x_start+int(big_img_cpy.shape[1]/n)] = dataset_img
#             x_start = x_end
#         y_start = y_end

#     return big_img_cpy

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