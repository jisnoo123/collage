<a href="https://huggingface.co/spaces/jisnoo/collage_image"><img src="badge_readme/open-in-hf-spaces-lg-dark.svg" alt="Open Demo in Hugging Face Spaces"/></a>

# Collager

Collager turns your images and videos to dataset collages, with a Gradio UI.
The original image is broken into patches and is replace with nearest neighbouring dataset image with respect to the mean pixel value of the three channels. 

# Datasets 

1. Anime
2. ImageNet10
3. SVHN
4. CIFAR_10

# Sample

## Image

Original Picture of RKMVERI

![Sample image](sample/images/rkmveri_golden.jpg)

Anime with 50 grids across the height. Zoom into the picture to see the Anime dataset images that replaced your image grids.

![Anime](sample/images/rkmveri_anime.png)

ImageNet10 with 50 grids across the height. Zoom into the picture to see ImageNet10 images.

![ImageNet10](sample/images/rkmveri_imnet10.png)

SVHN with 50 grids. Zoom into the picture to see SVHN images.

![SVHN](sample/images/rkmveri_svhn.png)

CIFAR_10 with 50 grids across the height. Zoom into the picture to see CIFAR_10 images.

![CIFAR_10](sample/images/rkmveri_cifar.png)

## Video

Original Video

![Bleach](sample/videos/bleach.gif)

ImageNet video with 100 grids across the height.

![ImageNet10](sample/videos/bleach_imnet10.gif)

# Prerequisites

## Install the packages required

From working directory, run `pip install -r requirements.txt`

## Creation of processed datasets

In the dataset folder, run **create_datasets.py**

`python3 create_datasets.py`

# Usage with Gradio UI 

## Image

In collage, run **gradio_app_img.py**

`python3 gradio_app_img.py`

Copy the local server link and paste it in your browser to work in the interface

![Image_UI](sample/UI/UI.png)

## Video

In collage, run **gradio_app_img.py**

`python3 gradio_app_img.py`


Copy the local server link and paste it in your browser to work in the interface

# Conclusion 

This is my first open source project. I thank my brother <a href = "https://jimut123.github.io/">Jimut</a> for giving me the idea of this project and <a href = "https://cs.rkmvu.ac.in/~sp/">Sw. Punyeshwarananda</a>, HOD, Dept. of Computer Science, RKMVERI for providing suggestions for improvements. 