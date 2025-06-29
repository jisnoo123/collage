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

# Custom CSS for enhanced UI
custom_css = """
body {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

.gradio-container {
    max-width: 1400px !important;
    margin: 0 auto;
    background: rgba(255, 255, 255, 0.95);
    border-radius: 20px;
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
    backdrop-filter: blur(10px);
    padding: 30px;
}

.header-title {
    text-align: center;
    color: #4a5568;
    font-size: 3.5rem;
    font-weight: 700;
    margin-bottom: 40px;
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
    background: linear-gradient(135deg, #667eea, #764ba2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.input-section {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    border-radius: 15px;
    padding: 25px;
    box-shadow: 0 10px 30px rgba(240, 147, 251, 0.3);
    border: 2px solid rgba(255, 255, 255, 0.2);
}

.controls-section {
    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    border-radius: 15px;
    padding: 25px;
    box-shadow: 0 10px 30px rgba(79, 172, 254, 0.3);
    border: 2px solid rgba(255, 255, 255, 0.2);
}

.output-section {
    background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
    border-radius: 15px;
    padding: 25px;
    box-shadow: 0 10px 30px rgba(67, 233, 123, 0.3);
    border: 2px solid rgba(255, 255, 255, 0.2);
    margin-top: 30px;
}

.gr-button {
    background: linear-gradient(135deg, #ff6b6b, #ee5a24) !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 15px 30px !important;
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    color: white !important;
    box-shadow: 0 8px 25px rgba(238, 90, 36, 0.4) !important;
    transition: all 0.3s ease !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
}

.gr-button:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 12px 35px rgba(238, 90, 36, 0.6) !important;
}

.gr-radio label {
    background: rgba(255, 255, 255, 0.9) !important;
    border-radius: 8px !important;
    padding: 12px 20px !important;
    margin: 5px !important;
    border: 2px solid transparent !important;
    transition: all 0.3s ease !important;
    font-weight: 500 !important;
}

.gr-radio label:hover {
    background: rgba(255, 255, 255, 1) !important;
    border-color: #667eea !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3) !important;
}

.gr-textbox input {
    border-radius: 10px !important;
    border: 2px solid rgba(255, 255, 255, 0.3) !important;
    background: rgba(255, 255, 255, 0.9) !important;
    padding: 12px 16px !important;
    font-size: 1rem !important;
    transition: all 0.3s ease !important;
}

.gr-textbox input:focus {
    border-color: #667eea !important;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.2) !important;
    background: white !important;
}

.gr-form label {
    color: white !important;
    font-weight: 600 !important;
    font-size: 1.1rem !important;
    text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.3) !important;
    margin-bottom: 8px !important;
}

.developer-link {
    text-align: center;
    margin-top: 15px;
    padding: 10px;
    background: rgba(255, 255, 255, 0.2);
    border-radius: 8px;
    backdrop-filter: blur(5px);
}

.developer-link a {
    color: white !important;
    text-decoration: none !important;
    font-weight: 500 !important;
    font-size: 0.95rem !important;
    text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.3) !important;
}

.developer-link a:hover {
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5) !important;
}

.gr-image {
    border-radius: 15px !important;
    overflow: hidden !important;
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.2) !important;
    border: 3px solid rgba(255, 255, 255, 0.3) !important;
}
"""

with gr.Blocks(css=custom_css, title="Collager") as demo:
    # Header
    gr.HTML('<h1 class="header-title">Collager</h1>')
    
    # Top row with input image and controls
    with gr.Row():
        with gr.Column(elem_classes="input-section"):
            ip = gr.Image(label='Input Image', type='filepath', height=350)
        
        with gr.Column(elem_classes="controls-section"):
            d = gr.Radio(['Anime', 'CIFAR_10', 'SVHN', 'ImageNet10'], 
                        label="Dataset", value="CIFAR_10")
            m = gr.Textbox(label='Number of Images on Height', 
                          placeholder="e.g., 50", value="50")
            coll_img_file_name = gr.Textbox(label='Output Filename', 
                                          placeholder="e.g., my_collage")
            generate_btn = gr.Button('Generate Collage', size="lg")
            
            gr.HTML('''
                <div class="developer-link">
                    <a href="https://jisnoo123.github.io/" target="_blank">
                        Developed by Jisnoo Dev Pal
                    </a>
                </div>
            ''')
    
    # Bottom section for output
    with gr.Column(elem_classes="output-section"):
        output = gr.Image(label='Generated Collage', height=500, show_label=True)
    
    generate_btn.click(fn=collage_image, 
                      inputs=[ip, d, m, coll_img_file_name], 
                      outputs=output)

demo.launch()