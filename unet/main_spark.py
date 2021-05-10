"""
This is a spark implementation of the keras model. It currently fails due to memory error
on GCP Dataproc with e2-standard-4 (4 vCPUs, 16 GB memory) cluster.
"""

# import argparse
# import time
#
# import numpy as np
# import matplotlib.pyplot as plt
# import tensorflow as tf
#
# from tqdm import tqdm
# from skimage.io import imshow
# from skimage.transform import resize
#
# from pyspark import SparkContext, SparkConf
# from elephas.utils.rdd_utils import to_simple_rdd
# from elephas.spark_model import SparkModel
#
# from load_data import load_train_test
# from preprocessing import create_image_mask_generator
# from unet import build_unet
# from evals import prob_to_rles
#
#
#
# # Set some parameters
# BATCH_SIZE = 1  # the higher the better
# IMG_WIDTH = 512  # for faster computing on kaggle
# IMG_HEIGHT = 512  # for faster computing on kaggle
# IMG_CHANNELS = 3
#
#
# seed = 42
#
#
# def parse_to_argdict():
#   parser = argparse.ArgumentParser(description='Training Script')
#   parser.add_argument('--root_dir', type=str, required=True)
#   parser.add_argument('--model_name', default='unet_model_spark', type=str, required=False)
#   parser.add_argument('--epochs', default=20, type=int, required=False)
#   parser.add_argument('--threads', default=2, type=int, required=False)
#   parser.add_argument('--save_demo_results', default=True, required=False)
#   args = vars(parser.parse_args())
#
#   return args
#
#
# def plot_learning(model_results, savepath):
#
#   plt.figure(figsize=[10, 6])
#   for key in model_results.history.keys():
#     plt.plot(model_results.history[key], label=key)
#
#   plt.legend()
#   plt.savefig(savepath)
#
#
# def show_images(i, ti, orgimg, y_true, preds, preds_t, savename):
#
#   if y_true is not None:
#     f = plt.figure(figsize=(10,10))
#     plt.subplot(221)
#     imshow(orgimg[i])
#     plt.title('Image to be Segmented')
#     plt.subplot(222)
#     imshow(y_true[ti])
#     plt.title('Segmentation Ground Truth')
#     plt.subplot(223)
#     imshow(preds[ti])
#     plt.title('Predicted Segmentation')
#     plt.subplot(224)
#     imshow(preds_t[ti])
#     plt.title('Thresholded Segmentation')
#     f.savefig(savename)
#   else:
#     f = plt.figure(figsize=(12, 7))
#     plt.subplot(131)
#     imshow(orgimg[i])
#     plt.title('Image to be Segmented')
#     plt.subplot(132)
#     imshow(preds[ti])
#     plt.title('Predicted Segmentation')
#     plt.subplot(133)
#     imshow(preds_t[ti])
#     plt.title('Thresholded Segmentation')
#     f.savefig(savename)
#
#
#
# def main():
#   # Read input args
#   args = parse_to_argdict()
#
#   ROOT_DIR = args['root_dir']
#   MODEL_NAME = args['model_name']
#   EPOCHS = args['epochs']
#   THREADS = args['threads']
#   SAVE_DEMO = args['save_demo_results']
#
#   TRAIN_PATH = ROOT_DIR + '/data/cell_imgs/'
#   MASK_PATH = ROOT_DIR + '/data/mask_imgs/'
#   TEST_PATH = ROOT_DIR + '/data/test_imgs/'
#
#   OUTPUT_DIR = ROOT_DIR + '/outs'
#   WEIGHTS = OUTPUT_DIR + '/' + MODEL_NAME +'_weights_spark.h5'
#   RESULTS_DIR = OUTPUT_DIR + '/spark_results'
#
#   # Load train test data
#   conf = SparkConf().setAppName('Elephas_App').setMaster('local[%d]' % THREADS)
#   sc = SparkContext(conf=conf)
#
#   X_train, Y_train, X_test, train_ids, test_ids, sizes_test = load_train_test(TRAIN_PATH, MASK_PATH, TEST_PATH,
#                                                                               IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)
#   train_size = int(X_train.shape[0] * 0.9)
#   xy_rdd = to_simple_rdd(sc, X_train, Y_train)
#
#   # Build U-Net model
#   model = build_unet(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
#   spark_model = SparkModel(model, frequency='epoch', mode='asynchronous')
#
#   # Fit model
#   spark_model.fit(xy_rdd, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0, validation_split=0.1)
#
#   # Predict on train, val and test
#   preds_train = spark_model.predict(X_train[:train_size])
#   print("predict output type: ", type(preds_train))
#   preds_val = spark_model.predict(X_train[train_size:])
#   preds_test = spark_model.predict(X_test)
#
#   # Validation Loss and Acc
#   val_results = model.evaluate(X_train[train_size:], Y_train[train_size:], batch_size=BATCH_SIZE)
#   print("evaluate output type: ", type(val_results))
#   print("Validation Loss:", val_results[0])
#   print("Validation Accuracy :", val_results[1] * 100, "%")
#
#   # Threshold predictions
#   preds_train_t = (preds_train > 0.5).astype(np.uint8)
#   preds_val_t = (preds_val > 0.5).astype(np.uint8)
#   preds_test_t = (preds_test > 0.5).astype(np.uint8)
#
#
#   # Save some example prediction results
#   if SAVE_DEMO:
#     i = 523
#     show_images(i, i, X_train, Y_train, preds_train, preds_train_t, savename=RESULTS_DIR + '/train%d_pred.png' % i)
#
#     i = 20
#     show_images(i, i, X_train[train_size:], Y_train[train_size:], preds_val, preds_val_t, savename=RESULTS_DIR + '/val%d_pred.png' % i)
#
#     i = 18
#     show_images(i, i, X_test, None, preds_test, preds_test_t, savename=RESULTS_DIR + '/test%d_pred.png' % i)
#
#   return
#
#
# if __name__ == "__main__":
#
#   main()
