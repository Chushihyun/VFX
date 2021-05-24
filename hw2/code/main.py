import re
import time
from datetime import datetime
import os
import numpy as np
import cv2
from copy import deepcopy
from util import projection, feature_detection, load_param, image_stitching, image_blending
import argparse
import glob



## main ##
parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default='parrington')
args = parser.parse_args()
# load image from folder
images = []
for path in sorted(glob.glob(args.path + "/*.JPG"), key = lambda x : int(re.search(r'\d+', os.path.basename(x)).group())):
    print(path)
    img = cv2.imread(path)
    if args.path != 'parrington':
        img = cv2.resize(img, None, fx=1, fy=1)
    images.append(img)

images = np.array(images)

print(images.shape)


# load data
images_gray = np.zeros(images.shape[:-1], dtype=np.uint8)
for i in range(images_gray.shape[0]):
    images_gray[i] = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)
# if args.path != 'parrington':
#     proj_param = [2100] * len(images)
# else:
proj_param=load_param(os.path.join(args.path, 'pano.txt')) #focal lengths


# feature detection(grayscale)
print("Start projection")
starttime = datetime.now()
images_cylindrical=projection(images,proj_param)
images_gray_cylindrical=projection(images_gray,proj_param)
print("projection complete, time = {}".format(datetime.now() - starttime))

print("Start feature detection")
starttime = datetime.now()
feature_list=feature_detection(images_gray_cylindrical)
print("feature detection complete, time = {}".format(datetime.now() - starttime))

print("Start image stitching")
starttime = datetime.now()
result,shifts=image_stitching(feature_list,images_cylindrical)
print("stitching complete, time = {}".format(datetime.now() - starttime))
# cv2.imwrite(path+"/result_gray.png", result)


# image stitching
# images_color_cylindrical=projection(images,proj_param)
# result=image_blending(images_color_cylindrical,shifts)


# save image
cv2.imwrite("result.png", result)



