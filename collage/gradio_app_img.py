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

# parser = argparse.ArgumentParser()

# parser.add_argument("--ip", help="Enter the big image input path", required=True)
# parser.add_argument("--op", help="Enter the collaged image output path", required=True)
# parser.add_argument("--m", help="Enter the m of grid", required=True)
# parser.add_argument("--n", help="Enter the n of grid", required=True)
# parser.add_argument("--d", help="Enter c for cifar10, m for mist and s for svhn", 
#                                                                     required=True)


# argparser = parser.parse_args()

# dataset_choice = argparser.d
# ip = argparser.ip
# op = argparser.op
# m = int(argparser.m)
# n = int(argparser.n)

# Global variable to track if image has been generated
image_generated = False

def collage_image(ip, d, m, n, op):
    global image_generated
    '''Loading the datasets'''

    m = int(m) # Convert to int as gradio takes in string
    n = int(n) # Convert to int as gradio takes in string

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

    img_big = cv2.imread(ip, cv2.IMREAD_UNCHANGED)
    # plt.imshow(img_big)
    # plt.show()
    img = cv2.cvtColor(img_big, cv2.COLOR_BGR2RGB)


    '''Perform extraction of patch and replacement with dataset image'''

    # print('Performing core execution')

    blr_big_img = blur(img, 151) #Applying a kernel size of 151
    final_img = core_img(blr_big_img, m, n, dataset, rb_av)

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



with gr.Blocks() as demo:
    with gr.Row(equal_height = True):
        ip = gr.Image(label = 'input image', type = 'filepath', height=400, width = 300)
        with gr.Column():
            d = gr.Radio(['Anime', 'CIFAR_10', 'SVHN', 'ImageNet10'], label="Dataset")
            with gr.Row():
                m = gr.Textbox(label = 'm')
                n = gr.Textbox(label = 'n')
            op = gr.Textbox(label = 'Output Path')

    generate_btn = gr.Button('Generate')

    output = gr.Image(label='Collaged image', height=400, width=400)

    generate_btn.click(fn = collage_image, inputs = [ip, d, m, n, op], outputs = output)

    
demo.launch()

