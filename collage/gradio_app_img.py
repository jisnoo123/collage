import os
import cv2
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import argparse
import pickle
from PIL import Image
import gradio as gr
import threading
import sys
sys.path.insert(0, '../')
from procedure import *

# Global variable to track if image has been generated
image_generated = False

def collage_image(ip, d, m, coll_img_file_name):
    global image_generated
    # Output path
    op = '../op/images/' + coll_img_file_name + '.png'
    
    '''Loading the datasets'''
    m = int(m)  # Convert to int as gradio takes in string
    
    # Calculate n based on input image aspect ratio
    img_big = cv2.imread(ip, cv2.IMREAD_UNCHANGED)
    height, width = img_big.shape[:2]
    aspect_ratio = width / height
    n = int(m * aspect_ratio)
    
    if d == 'CIFAR_10':
        # Load the CIFAR10 dataset
        with open('../dataset/actual/cifar10', 'rb') as f:
            dataset = pickle.load(f)
        # Load the cifar10 dataset by unpickling the pickled data
        with open('../dataset/rb_av/cifar10_rb_av', 'rb') as f:
            rb_av = pickle.load(f)
    elif d == 'SVHN':
        # Load the SVHN dataset
        with open('../dataset/actual/svhn', 'rb') as f:
            dataset = pickle.load(f)
        # Load the svhn dataset by unpickling the pickled data
        with open('../dataset/rb_av/svhn_rb_av', 'rb') as f:
            rb_av = pickle.load(f)
    elif d == 'ImageNet10':
        # Load the ImageNet10 dataset
        with open('../dataset/actual/imnet10', 'rb') as f:
            dataset = pickle.load(f)
        # Load the ImageNet10 dataset by unpickling the pickled data
        with open('../dataset/rb_av/imnet10_rb_av', 'rb') as f:
            rb_av = pickle.load(f)
    elif d == 'Anime':
        # Load the Anime dataset
        with open('../dataset/actual/anime', 'rb') as f:
            dataset = pickle.load(f)
        # Load the Anime dataset by unpickling the pickled data
        with open('../dataset/rb_av/anime_rb_av', 'rb') as f:
            rb_av = pickle.load(f)
    
    '''Loading the big image'''
    # plt.imshow(img_big)
    # plt.show()
    img = cv2.cvtColor(img_big, cv2.COLOR_BGR2RGB)
    
    '''Perform extraction of patch and replacement with dataset image'''
    # print('Performing core execution')
    blr_big_img = blur(img, 1) #Applying a kernel size of 1
    final_img = core_img(blr_big_img, m, n, dataset, rb_av, d)
    # print('Core execution completed')
    
    '''Show the final image'''
    # print('Your final image is ready')
    # plt.imshow(final_img)
    # plt.show()
    
    '''Save it in the output path'''
    pil_image = Image.fromarray(final_img)
    pil_image.save(op)
    
    # Mark that image has been generated and start monitoring for close prompt
    image_generated = True
    threading.Thread(target=monitor_close_prompt, daemon=True).start()
    
    return final_img

def monitor_close_prompt():
    """Monitor for user input to close server"""
    print("\nPress 'y' to close the server: ", end='', flush=True)
    user_input = input().strip().lower()
    if user_input == 'y':
        print("Closing server...")
        os._exit(0)


with gr.Blocks(css=".gradio-container {background-color: #0E79B2;}") as demo:
    with gr.Row(equal_height = True):
        ip = gr.Image(label = 'Original Image', type='filepath', height=400, width = 300)
        with gr.Column():
            d = gr.Radio(['Anime', 'ImageNet10', 'CIFAR_10', 'SVHN'], label="Dataset")
            m = gr.Textbox(label = 'No. of grids across the height')
            # n = gr.Textbox(label = 'n')
            coll_img_file_name = gr.Textbox(label = 'Enter collaged image filename [To be saved in collage/dataset/op/videos]')
    generate_btn = gr.Button('Generate')

    output = gr.Image(label='Collaged image', height=400, width=600)
    
    generate_btn.click(fn=collage_image, 
                      inputs=[ip, d, m, coll_img_file_name], 
                      outputs=output)

demo.launch()