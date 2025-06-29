<a href="https://huggingface.co/spaces/jisnoo/collage_img"><img src="badge_readme/open-in-hf-spaces-lg-dark.svg" alt="Open Demo in Hugging Face Spaces"/></a>

# Collager

Turn your images and videos into a dataset collage. Look at it from a distance, you would see the original shadow (I don't think if I should call it that) of your image or video, replaced with the dataset images.

# What's in here?

A Gradio UI that makes you collages of your images or videos. I do not recommend the video part of it, its pretty boring, however, I do recommend to try it on your portraits. Its cool on images! The more the number of grids the better it is. Go through the samples, you will understand what I am talking about. 

# Datasets 

1. Anime
2. ImageNet10
3. SVHN
4. CIFAR_10

# Sample

## Image

Original Picture

![Sample image](sample/sample_pic.jpg)

Anime with 10000 grids. Zoom into the picture to see the Anime dataset images that replaced your image grids.

![Anime](sample/images/girl_anime.png)

ImageNet10 with 10000 grids. Zoom into the picture to see ImageNet10 images.

![ImageNet10](sample/images/girl_imnet.png)

SVHN with 10000 grids. Zoom into the picture to see SVHN images.

![SVHN](sample/images/girl_svhn.png)

CIFAR_10 with 10000 grids. Zoom into the picture to see CIFAR_10 images.

![CIFAR_10](sample/images/girl_cifar.png)

## Video

Original Video

![Bleach](sample/videos/video_bleach.gif)

ImageNet video with 900 Grids.

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

![Image_UI](sample/UI/image.png)

## Video

In collage, run **gradio_app_img.py**

`python3 gradio_app_img.py`


Copy the local server link and paste it in your browser to work in the interface

![Video_UI](sample/UI/video.png)

# Usage for Pro users

## Images

Edit the **start_collage_img.bash** according to your requirements. Then run it.

`bash start_collage_img.bash`

## Videos

Edit the **start_collage_vid.bash** according to your requirements. Then run it.

`bash start_collage_vid_batch.bash`

### Remarks

Wondering why the **gradio_app_vid.py** and **start_collage_vid.bash** is there for?
It doesn't process your videos in batches so your 10s video will take 10 days to be ready. I kept it there because I implemented it first. 

# Conclusion

This is pretty much it. The idea of this project was given by my brother, which he told me to extend it into an open source software. Thank you.