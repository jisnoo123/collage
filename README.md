# collage

Turn your images and videos into dataset shadows.

## Datasets available:

1. Anime
2. ImageNet10
3. SVHN
4. CIFAR_10

## Samples

## Image

Original Picture

![Sample image](sample/sample_pic.jpg)

Anime with 10000 grids

![Anime](sample/images/girl_anime.png)

ImageNet10 with 10000 grids

![ImageNet10](sample/images/girl_imnet.png)

SVHN with 10000 grids

![SVHN](sample/images/girl_svhn.png)

CIFAR_10 with 10000 grids

![CIFAR_10](sample/images/girl_cifar.png)

## Video

Original Video

![Bleach](sample/videos/video_bleach.gif)

ImageNet video with 900 Grids

![ImageNet10](sample/videos/bleach_imnet10.gif)

## Usage

### Download the datasets

In dataset, run the download_data_grdive.py

`python3 download_data_gdrive.py`

### UI for images

In collage, run gradio_app_img.py

`python3 gradio_app_img.py`

Copy the local server link and paste it in your browser to work in the interface

![Image_UI](sample/UI/image.png)

### UI for videos

In collage, run gradio_app_vid_batch.py for faster execution that processes video frames in batches, otherwise you can go with gradio_app_vid.py that will take months to process a 1 minute video.

For faster execution:
`python3 gradio_app_vid_batch.py`

For sloth execution:
`python3 gradio_app_vid.py`

Copy the local server link and paste it in your browser to work in the interface

![Video_UI](sample/UI/video.png)