import os
import sys
from skimage.io import imread, imsave

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

