import os
import sys
import warnings

import numpy as np

from tqdm import tqdm
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize


warnings.filterwarnings('ignore', category=UserWarning, module='skimage')


def load_train_test(train_path, mask_path, test_path, width=512, height=512, channels=3):
  # Get train and test IDs
  train_ids = [str(i) for i in range(1, 671, 1)]
  test_ids = [str(i) for i in range(65)]
  np.random.seed(10)

  # Get and resize train images and masks
  X_train = np.zeros((len(train_ids), height, width, channels), dtype=np.uint8)
  Y_train = np.zeros((len(train_ids), height, width, 1), dtype=np.bool)

  print('Getting and resizing train images and masks ... ')
  sys.stdout.flush()
  for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    img = imread(train_path + id_ + '.png')[:, :, :channels]
    img = resize(img, (height, width), mode='constant', preserve_range=True)
    X_train[n-1] = img

    mask_ = imread(mask_path + id_ + 'mask.png', plugin='matplotlib')[:,:,0]
    mask_ = resize(mask_, (height, width), mode='constant', preserve_range=True)
    mask = np.expand_dims(mask_, axis=-1)
    Y_train[n-1] = mask

  # Get and resize test images
  X_test = np.zeros((len(test_ids), height, width, channels), dtype=np.uint8)
  sizes_test = []
  print('Getting and resizing test images ... ')
  sys.stdout.flush()
  for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    img = imread(test_path + 'test_' + id_ + '.png')[:, :, :channels]
    sizes_test.append([img.shape[0], img.shape[1]])
    img = resize(img, (height, width), mode='constant', preserve_range=True)
    X_test[n] = img

  print('Finished loading train, test data!')

  return X_train, Y_train, X_test, train_ids, test_ids, sizes_test



# load_train_test("../cell_imgs/", "../mask_imgs/", "../test_imgs/")