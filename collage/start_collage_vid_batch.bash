# Input image path
ip="../ip/videos/actual/bleach_cropped_2s.mp4"

# Output image path along with the name of the image to be saved as.
op="../op/videos/collaged_vid_batch.mp4"

# Convert the video to suitable format for processing
ffmpeg -y -loglevel quiet -i $ip -c:v libx264 -c:a aac "../ip/videos/converted/converted_video.mp4"


# Change ip to converted format video
ip="../ip/videos/converted/converted_video.mp4"

# m x n grids
m=30
n=30

# Dataset choice
# c for CIFAR10 and s for SVHN
d='c'


# Run collage_vid.py
python3 collage_vid_batch.py --ip $ip --m $m --n $n --d $d --op $op