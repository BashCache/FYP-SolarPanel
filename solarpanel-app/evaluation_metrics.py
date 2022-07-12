import tensorflow as tf
from keras import backend as K 

def IOU(y_true, y_pred):
  smooth = 1.0
  y_true = tf.cast(y_true, tf.float32)
  y_pred = tf.cast(y_pred, tf.float32)
  y_true_f = K.round(K.flatten(y_true))
  y_pred_f = K.round(K.flatten(y_pred))
  intersection = K.sum(y_true_f * y_pred_f)
  union = K.sum(y_true_f + y_pred_f - y_true_f * y_pred_f)
  jacc = (intersection + smooth) / (union + smooth)
  return jacc

def dice_coef(y_true, y_pred):
    smooth = 1.0
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true_f = K.round(K.flatten(y_true))
    y_pred_f = K.round(K.flatten(y_pred))
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def mcc(y_true, y_pred):
  smooth = 1.0
  y_true = tf.cast(y_true, tf.float32)
  y_pred = tf.cast(y_pred, tf.float32)
  y_true_f = K.round(K.flatten(y_true))
  y_pred_f = K.round(K.flatten(y_pred))
  tp = K.sum(y_true_f * y_pred_f)
  tn = K.sum((1-y_true_f) * (1-y_pred_f))
  fp = K.sum((1-y_true_f)*y_pred_f)
  fn = K.sum(y_true_f * (1-y_pred_f))
  up = tp*tn - fp*fn
  down = K.sqrt((tp+fp) * (tp+fn) * (tn+fp) * (tn+fn))
  mcc = (up + smooth) / (down + smooth)
  return mcc

