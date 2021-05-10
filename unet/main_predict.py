import argparse
import numpy as np
import time
import sys

from tqdm import tqdm
from skimage.io import imsave, imread
from skimage.transform import resize

from keras.models import Model, load_model


# Set some parameters
IMG_WIDTH = 512  # for faster computing on kaggle
IMG_HEIGHT = 512  # for faster computing on kaggle
IMG_CHANNELS = 3


def parse_to_argdict():
  parser = argparse.ArgumentParser(description='Testing Script')
  parser.add_argument('--root_dir', type=str, required=True)
  parser.add_argument('--test_dir', type=str, required=True)
  parser.add_argument('--test_size', type=int, required=True)
  parser.add_argument('--weights_fp', type=str, required=True)
  args = vars(parser.parse_args())

  return args


def main():
  # Read input args
  args = parse_to_argdict()

  ROOT_DIR = args['root_dir']
  TEST_PATH = args['test_dir']
  TEST_SIZE = args['test_size']
  WEIGHTS = args['weights_fp']

  RESULTS_DIR = ROOT_DIR + '/outs/results'

  # Load and resize test images
  test_ids = [str(i) for i in range(TEST_SIZE)]
  X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
  sizes_test = []

  print('Getting and resizing test images ... ')
  function_time_outs = ""
  start = time.time()
  sys.stdout.flush()
  for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    img = imread(TEST_PATH + '/test_' + id_ + '.png')[:, :, :IMG_CHANNELS]
    sizes_test.append([img.shape[0], img.shape[1]])
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_test[n] = img
  function_time_outs += "Load test data: %.3f sec\n" % (time.time() - start)
  print('Finished loading test data!')

  # load model
  start = time.time()
  model = load_model(WEIGHTS)
  function_time_outs += "Load model: %.3f sec\n" % (time.time() - start)

  start = time.time()
  preds_test = model.predict(X_test, verbose=1)
  preds_test_t = (preds_test > 0.5).astype(np.uint8)
  function_time_outs += "Predict masks: %.3f sec\n" % (time.time() - start)

  # Save predicted masks to results dir
  start = time.time()
  for i, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    test_mask = resize(np.squeeze(preds_test_t[i]), (sizes_test[i][0], sizes_test[i][1]),
                       mode='constant', preserve_range=True)
    imsave(RESULTS_DIR + '/test%d_pred.png' % i, test_mask)
  function_time_outs += "Save predicted masks: %.3f sec\n" % (time.time() - start)

  return function_time_outs


if __name__ == "__main__":
  start = time.time()
  function_times = main()
  print("Script total time: %.3f sec" % (time.time() - start))
  print("\nFunction Time Breakdown:\n")
  print(function_times)
