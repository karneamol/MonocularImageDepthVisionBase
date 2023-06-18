import shutil
import os
from progressbar import Percentage, ProgressBar,Bar,ETA, Counter
from fill_depth_colorization import *
from PIL import Image
from numpy import asarray
import time
from tqdm import tqdm

rgb_file = 'dataset/final_kitti_train.txt'
depth_file = 'dataset/final_depth_train.txt'

with open(rgb_file) as f:
    rgb_files = f.readlines()
    rgb_files = [x.strip() for x in rgb_files]
    f.close()
with open(depth_file) as f:
    depth_files = f.readlines()
    depth_files = [x.strip() for x in depth_files]
    f.close()
    
pbar = ProgressBar(widgets=[Bar('>', '[', ']'), ' ', Counter(), ' ', ETA()],maxval=len(depth_files))
    
for rgb, depth in tqdm(zip(rgb_files,depth_files)):

    # Open the image form working directory
    rgb_image = Image.open('/home/pskbalaji_project/project/depth-kitti-unet/dataset/data/train/rgb/' + rgb)

    # convert image to numpy array
    rgb_array = asarray(rgb_image)

    # Open the image form working directory
    depth_image = Image.open('/home/pskbalaji_project/project/depth-kitti-unet/dataset/data/train/depth/' + depth)

    # convert image to numpy array
    depth_array = asarray(depth_image)
    depth_array = depth_array / 256
                       
    output = fill_depth_colorization (rgb_array, depth_array, 1)
    
    # Save output as an image
    image = Image.fromarray(output.astype('uint8'))
    if not os.path.exists('./FillDepth/' + depth[:depth.rfind("/") + 1]):
        os.makedirs('./FillDepth/' + depth[:depth.rfind("/") + 1])
    image.save('./FillDepth/' + depth)