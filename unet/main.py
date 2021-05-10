import argparse
import time

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tqdm import tqdm
from skimage.io import imshow, imsave
from skimage.transform import resize

from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint

from load_data import load_train_test
from preprocessing import create_image_mask_generator
from unet import build_unet


# Set some parameters
BATCH_SIZE = 1  # the higher the better
IMG_WIDTH = 512  # for faster computing on kaggle
IMG_HEIGHT = 512  # for faster computing on kaggle
IMG_CHANNELS = 3


seed = 42


def parse_to_argdict():
  parser = argparse.ArgumentParser(description='Training Script')
  parser.add_argument('--root_dir', type=str, required=True)
  parser.add_argument('--model_name', default='unet_model', type=str, required=False)
  parser.add_argument('--epochs', default=10, type=int, required=False)
  parser.add_argument('--save_demo_results', default=True, required=False)
  parser.add_argument('--save_preds', default=False, type=bool, required=False)
  args = vars(parser.parse_args())

  return args


def plot_learning(model_results, savepath):

  plt.figure(figsize=[10, 6])
  for key in model_results.history.keys():
    plt.plot(model_results.history[key], label=key)

  plt.legend()
  plt.savefig(savepath)


def show_images(i, ti, orgimg, y_true, preds, preds_t, savename):

  if y_true is not None:
    f = plt.figure(figsize=(10,10))
    plt.subplot(221)
    imshow(orgimg[i])
    plt.title('Image to be Segmented')
    plt.subplot(222)
    imshow(y_true[ti])
    plt.title('Segmentation Ground Truth')
    plt.subplot(223)
    imshow(preds[ti])
    plt.title('Predicted Segmentation')
    plt.subplot(224)
    imshow(preds_t[ti])
    plt.title('Thresholded Segmentation')
    f.savefig(savename)
  else:
    f = plt.figure(figsize=(12, 7))
    plt.subplot(131)
    imshow(orgimg[i])
    plt.title('Image to be Segmented')
    plt.subplot(132)
    imshow(preds[ti])
    plt.title('Predicted Segmentation')
    plt.subplot(133)
    imshow(preds_t[ti])
    plt.title('Thresholded Segmentation')
    f.savefig(savename)


def main():
  # Read input args
  args = parse_to_argdict()

  ROOT_DIR = args['root_dir']
  MODEL_NAME = args['model_name']
  EPOCHS = args['epochs']
  SAVE_DEMO = args['save_demo_results']
  SAVE_PREDS = args['save_preds']

  TRAIN_PATH = ROOT_DIR + '/data/cell_imgs/'
  MASK_PATH = ROOT_DIR + '/data/mask_imgs/'
  TEST_PATH = ROOT_DIR + '/data/test_imgs/'

  OUTPUT_DIR = ROOT_DIR + '/outs'
  WEIGHTS = OUTPUT_DIR + '/' + MODEL_NAME +'_weights.h5'
  LOG_DIR = OUTPUT_DIR + "/logs"
  RESULTS_DIR = OUTPUT_DIR + '/results'

  # Load train test data
  function_time_outs = ""
  start = time.time()
  X_train, Y_train, X_test, train_ids, test_ids, sizes_test = load_train_test(TRAIN_PATH, MASK_PATH, TEST_PATH,
                                                                              IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)
  function_time_outs += "Load train test: %.3f sec\n" % (time.time() - start)
  train_size = int(X_train.shape[0] * 0.9)

  # Data augmentation
  train_generator, val_generator = create_image_mask_generator(X_train, Y_train, BATCH_SIZE, seed)

  # Build U-Net model
  start = time.time()
  model = build_unet(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
  function_time_outs += "Set up UNet: %.3f sec\n" % (time.time() - start)

  # Fit model
  start = time.time()
  earlystopper = EarlyStopping(patience=100, verbose=1)
  function_time_outs += "Set up EarlyStopper: %.3f sec\n" % (time.time() - start)
  start = time.time()
  checkpointer = ModelCheckpoint(WEIGHTS, verbose=1, save_best_only=True)
  function_time_outs += "Set up Checkpointer: %.3f sec\n" % (time.time() - start)
  start = time.time()
  tensorboard = tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR, histogram_freq=1, profile_batch='500,520')
  function_time_outs += "Set up TensorBoard: %.3f sec\n" % (time.time() - start)

  start = time.time()
  model_results = model.fit_generator(train_generator, validation_data=val_generator, validation_steps=10,
                                      steps_per_epoch=200, epochs=EPOCHS,
                                      callbacks=[tensorboard, earlystopper, checkpointer])
  # model_results = model.fit(X_train[:train_size], Y_train[:train_size], BATCH_SIZE,
  #                           validation_data=(X_train[train_size:], Y_train[train_size:]), validation_steps=10,
  #                           steps_per_epoch=200, epochs=EPOCHS,
  #                           callbacks=[tensorboard, earlystopper, checkpointer])
  function_time_outs += "Model training: %.3f sec\n" % (time.time() - start)


  # Predict on train, val and test
  start = time.time()
  model = load_model(WEIGHTS)  # custom_objects={'mean_iou': mean_iou}
  function_time_outs += "Load pretrained model: %.3f sec\n" % (time.time() - start)

  start = time.time()
  preds_train = model.predict(X_train[:train_size], verbose=1)
  preds_val = model.predict(X_train[train_size:], verbose=1)
  preds_test = model.predict(X_test, verbose=1)

  # Threshold predictions
  preds_train_t = (preds_train > 0.5).astype(np.uint8)
  preds_val_t = (preds_val > 0.5).astype(np.uint8)
  preds_test_t = (preds_test > 0.5).astype(np.uint8)
  function_time_outs += "Predict on train & test: %.3f sec\n" % (time.time() - start)

  # Validation Loss and Acc
  start = time.time()
  val_results = model.evaluate(X_train[train_size:], Y_train[train_size:], batch_size=BATCH_SIZE)
  function_time_outs += "Validation eval: %.3f sec" % (time.time() - start)
  print("Validation Loss:", val_results[0])
  print("Validation Accuracy :", val_results[1] * 100, "%")

  # Save all predictions
  if SAVE_PREDS:
    start = time.time()
    for i, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
      test_mask = resize(np.squeeze(preds_test_t[i]), (sizes_test[i][0], sizes_test[i][1]),
                         mode='constant', preserve_range=True)
      imsave(RESULTS_DIR + '/test%d_pred.png' % i, test_mask)
    function_time_outs += "Save predicted masks: %.3f sec\n" % (time.time() - start)

  # Save example prediction results
  if SAVE_DEMO:
    # Plot learning curve
    plot_learning(model_results, savepath=OUTPUT_DIR + "/learning_curve.png")

    i = 58
    show_images(i, i, X_train, Y_train, preds_train, preds_train_t, savename=RESULTS_DIR + '/train%d_pred.png' % i)

    i = 20
    show_images(i, i, X_train[train_size:], Y_train[train_size:], preds_val, preds_val_t, savename=RESULTS_DIR + '/val%d_pred.png' % i)

    i = 18
    show_images(i, i, X_test, None, preds_test, preds_test_t, savename=RESULTS_DIR + '/test%d_pred.png' % i)

  return function_time_outs


if __name__ == "__main__":
  start = time.time()
  function_times = main()
  print("Script total time: %.3f sec" % (time.time() - start))
  print("\nFunction Time Breakdown:\n")
  print(function_times)

  # NOTE: Tesla P4 does not have enough memory. 8GB is not enough, it needs at least ~14GB.

  # In cmdline:
  # 1. python -m cProfile -o main.profile main.py
  # 2. python -m pstats main.profile
  # 3. strip
  # 4. sort
  # 5. sort time
  # 6. stats 10
