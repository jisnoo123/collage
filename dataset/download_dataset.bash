#!/bin/bash

sudo apt install unzip

curl -L -o anime.zip\
  https://www.kaggle.com/api/v1/datasets/download/diraizel/anime-images-dataset

#!/bin/bash
curl -L -o imagenet10.zip\
  https://www.kaggle.com/api/v1/datasets/download/liusha249/imagenet10

unzip anime.zip

unzip imagenet10.zip