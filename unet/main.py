import argparse
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from skimage.transform import resize

from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint


from load_data import load_train_test
from preprocessing import create_image_mask_generator
from unet import build_unet
from evals import prob_to_rles

import tensorflow as tf
#import googlecloudprofiler


# Set some parameters
BATCH_SIZE = 32  # the higher the better
IMG_WIDTH = 512  # for faster computing on kaggle
IMG_HEIGHT = 512  # for faster computing on kaggle
IMG_CHANNELS = 3


seed = 42


if __name__ == "__main__":
  # Profiler initialization. It starts a daemon thread which continuously
  # collects and uploads profiles. Best done as early as possible.
  # try:
  #   googlecloudprofiler.start(
  #           service='hello-profiler',
  #           service_version='1.0.1',
  #           # verbose is the logging level. 0-error, 1-warning, 2-info,
  #           # 3-debug. It defaults to 0 (error) if not set.
  #           verbose=3,
  #           # project_id must be set if not running on GCP.
  #           # project_id='my-project-id',
  #       )
  # except (ValueError, NotImplementedError) as exc:
  #   print(exc)  # Handle errors here


  parser = argparse.ArgumentParser(description='Training Script')
  parser.add_argument('--data_dir', type=str, required=True)
  parser.add_argument('--log_dir', default='./', type=str, required=True)
  parser.add_argument('--epochs', default=20, type=int, required=False)
  parser.add_argument('--weights', default='unet_model.h5', type=str, required=False)
  args = vars(parser.parse_args())

  ROOT_DIR = args['data_dir']
  TRAIN_PATH = ROOT_DIR + '/cell_imgs/'
  MASK_PATH = ROOT_DIR + '/mask_imgs/'
  TEST_PATH = ROOT_DIR + '/test_imgs/'
  EPOCHS = args['epochs']
  WEIGHTS = args['weights']
  LOG_DIR = args['log_dir']

  # Load train test data
  X_train, Y_train, X_test, train_ids, test_ids, sizes_test = load_train_test(TRAIN_PATH, MASK_PATH, TEST_PATH,
                                                                              IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)
  train_size = int(X_train.shape[0] * 0.9)

  # Data augmentation
  train_generator, val_generator = create_image_mask_generator(X_train, Y_train, BATCH_SIZE, seed)

  # Build U-Net model
  model = build_unet(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

  # Fit model
  earlystopper = EarlyStopping(patience=100, verbose=1)
  checkpointer = ModelCheckpoint(WEIGHTS, verbose=1, save_best_only=True)
  tensorboard = tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR, histogram_freq=1, profile_batch='500,520')

  results = model.fit_generator(train_generator, validation_data=val_generator, validation_steps=10,
                                steps_per_epoch=200,
                                epochs=EPOCHS, callbacks=[tensorboard, earlystopper, checkpointer])

  # Predict on train, val and test
  model = load_model(WEIGHTS) #custom_objects={'mean_iou': mean_iou}
  preds_train = model.predict(X_train[:train_size], verbose=1)
  preds_val = model.predict(X_train[train_size:], verbose=1)
  preds_test = model.predict(X_test, verbose=1)

  # Threshold predictions
  preds_train_t = (preds_train > 0.5).astype(np.uint8)
  preds_val_t = (preds_val > 0.5).astype(np.uint8)
  preds_test_t = (preds_test > 0.5).astype(np.uint8)

  val_results = model.evaluate(X_train[train_size:], Y_train[train_size:], batch_size=BATCH_SIZE)
  print("Test Loss:", val_results[0])
  print("Test Acc :", val_results[1] * 100, "%")

  # # Compute Accuracy of on train and validation set
  # iou_train, iou_val = 0, 0
  # for i in range(train_size):
  #   iou_train += mean_iou(Y_train[i], preds_train_t[i])
  # iou_train /= train_size
  #
  # for i in range(train_size, X_train.shape[0] + 1, 1):
  #   iou_val += mean_iou(Y_train[i], preds_val_t[i])
  # iou_val /= (X_train.shape[0] - train_size)
  #
  # print("Average mean IoU on training set: ", iou_train)
  # print("Average mean IoU on validation set: ", iou_val)

  # Create list of upsampled test masks
  preds_test_upsampled = []
  for i in range(len(preds_test)):
    preds_test_upsampled.append(resize(np.squeeze(preds_test[i]),
                                       (sizes_test[i][0], sizes_test[i][1]),
                                       mode='constant', preserve_range=True))

  new_test_ids = []
  rles = []
  for n, id_ in enumerate(test_ids):
    rle = list(prob_to_rles(preds_test_upsampled[n]))
    rles.extend(rle)
    new_test_ids.extend([id_] * len(rle))

