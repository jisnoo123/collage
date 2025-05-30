import cv2

def resize_img(img, ht, wd):
    '''Resizes an image into (ht*wd)'''
    resized_img = cv2.resize(img, (ht,wd))
    return resized_img

def blur(img, kernel_size):
    '''Applies Gaussian Blur to an image img of kernel_size'''
    blurred_img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    return blurred_img