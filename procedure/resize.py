import cv2

def resize_img(img, ht, wd):
    '''Resizes an image into (ht*wd)'''
    resized_img = cv2.resize(img, (ht,wd))
    return resized_img