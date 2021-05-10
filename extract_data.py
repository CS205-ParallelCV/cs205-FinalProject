import os
import sys
from shutil import copy
from PIL import Image
import numpy as np
from skimage.io import imread, imsave
import cv2  # Make sure you have installed opencv-python locally

"""
Step 1: 
navigate to the folder containing stage1_train as a subdirectory
"""

"""
Step 2: 
Create a new subdirectory to store the images if,
if it doesn't already exists.
"""
cell_folder_path = './cell_imgs'
if not os.path.isdir(cell_folder_path):
  os.mkdir(cell_folder_path)
  print("Directory '% s' created" % cell_folder_path)

mask_folder_path = './mask_imgs'
if not os.path.isdir(mask_folder_path):
  os.mkdir(mask_folder_path)
  print("Directory '% s' created" % mask_folder_path)

"""
Step 3: 
Navigate to the folder containing stage1_train
"""
directory = "./stage1_train/"
img = Image

"""
Step 4: 
Iterate through all cell images and copy them to the new directory ./cell_imgs
All images are renamed as integers starting from 1.png, 2.png...
"""
filename = 1

# this variable is for making sure the naming of mask and cell images matches with each other
namedboth = 0

for root, subdirectories, files in os.walk(directory):
  print(os.path.join(root))
  # for cell images
  if "images" in root:
    namedboth += 1
    for file in files:
      copy(os.path.join(root, file), cell_folder_path + "/" + str(filename) + ".png")

  # for aggregating masks
  if "mask" in root:
    namedboth += 1
    for file in files:
      img = cv2.imread(os.path.join(root, file), 0)
      pic_shape = img.shape
      break
    mask = np.zeros(pic_shape)

    for file in files:
      # print(os.path.join(root, file))
      mask += cv2.imread(os.path.join(root, file), 0)

    mask = np.clip(mask, 0, 256)
    im = Image.fromarray(mask)
    if im.mode != 'RGB':
      im = im.convert('RGB')
    im.save(mask_folder_path + "/" + str(filename) + "mask.png")

  if namedboth == 2:
    filename += 1
    namedboth = 0


test_path = '../stage1_test/'
save_path = '../test_imgs/'
test_ids = next(os.walk(test_path))[1]

print("Number of test images: ", len(test_ids))
sys.stdout.flush()
# Make a cleaned folder with all test images from the raw dataset
for n, id_ in enumerate(test_ids):
  path = test_path + id_
  img = imread(path + '/images/' + id_ + '.png')[:, :, :3]
  test_name = "test_%d.png" % n
  imsave(save_path + test_name, img)

