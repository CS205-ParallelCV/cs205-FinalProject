import numpy as np
import tensorflow as tf

from skimage.morphology import label
from keras import backend as K

# Define IoU metric (average intersection of union)
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        #y_pred_ = tf.cast(y_pred > t, tf.int32)
        #score, up_opt = tf.compat.v1.metrics.mean_iou(y_true, y_pred_, 2)
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)


# # Define IOU metric
# def iou_metric(y_true_in, y_pred_in, print_table=False):
#   labels = label(y_true_in > 0.5)
#   y_pred = label(y_pred_in > 0.5)
#
#   true_objects = len(np.unique(labels))
#   pred_objects = len(np.unique(y_pred))
#
#   intersection = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))[0]
#
#   # Compute areas (needed for finding the union between all objects)
#   area_true = np.histogram(labels, bins=true_objects)[0]
#   area_pred = np.histogram(y_pred, bins=pred_objects)[0]
#   area_true = np.expand_dims(area_true, -1)
#   area_pred = np.expand_dims(area_pred, 0)
#
#   # Compute union
#   union = area_true + area_pred - intersection
#
#   # Exclude background from the analysis
#   intersection = intersection[1:, 1:]
#   union = union[1:, 1:]
#   union[union == 0] = 1e-9
#
#   # Compute the intersection over union
#   iou = intersection / union
#
#   # Precision helper function
#   def precision_at(threshold, iou):
#     matches = iou > threshold
#     true_positives = np.sum(matches, axis=1) == 1  # Correct objects
#     false_positives = np.sum(matches, axis=0) == 0  # Missed objects
#     false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
#     tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
#     return tp, fp, fn
#
#   # Loop over IoU thresholds
#   prec = []
#   if print_table:
#     print("Thresh\tTP\tFP\tFN\tPrec.")
#   for t in np.arange(0.5, 1.0, 0.05):
#     tp, fp, fn = precision_at(t, iou)
#     if (tp + fp + fn) > 0:
#       p = tp / (tp + fp + fn)
#     else:
#       p = 0
#     if print_table:
#       print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
#     prec.append(p)
#
#   if print_table:
#     print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
#   return np.mean(prec)
#
#
# def iou_metric_batch(y_true_in, y_pred_in):
#   batch_size = y_true_in.shape[0]
#   metric = []
#   for batch in range(batch_size):
#     value = iou_metric(y_true_in[batch], y_pred_in[batch])
#     metric.append(value)
#   return np.array(np.mean(metric), dtype=np.float32)
#
#
# def my_iou_metric(label, pred):
#   metric_value = tf.py_function(iou_metric_batch, [label, pred], tf.float32)
#   return metric_value



# Run-length encoding
def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths


def prob_to_rles(x, cutoff=0.5):
    lab_img = label(x > cutoff)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)

#print(tf.__version__)