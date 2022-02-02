import glob
import mmcv
from PIL import Image
import json
import numpy as np

img_path = "./data/codebrim_coco/val/"

print("parsing images...")
images = glob.glob(img_path+"*.jpg")
print(images)

print("checking images...")
inverted = []
for f in images:
    if str(mmcv.imread(f).shape) != str(np.array(Image.open(f)).shape):
        print(f'The image {f} is wrong!')
        inverted.append(f)
print(len(inverted),"bad images :",inverted)