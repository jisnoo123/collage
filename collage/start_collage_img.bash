# Input image path
ip="../ip/images/sample_img.jpg"

# Output image path along with the name of the image to be saved as.
op="../op/images/collaged_img.jpg"

# m x n grids
m=30
n=30

# Dataset choice
# a for anime, c for CIFAR10, i for ImageNet10 and  s for SVHN
d='c'


# Run collage_img.py
python3 collage_img.py --ip $ip --m $m --n $n --d $d --op $op