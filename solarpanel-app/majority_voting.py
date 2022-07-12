import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings

warnings.filterwarnings("ignore")

def majority_voting(preds):
    for i in range(preds.shape[0]):
        count = [0, 0, 0]
        for j in range(preds.shape[1]):
            count[preds[i][j]] += 1
        idxs = count.index(max(count))
        print(count, idxs)
        return idxs

def classification_majority_voting(filename, X):
    CNN = tf.keras.models.load_model('weights-50epochs-mod-dataset2\\CNN-RMSProp-50epochs-mod2')
    predictions_CNN = CNN.predict(X)
    pred_labels_CNN = np.argmax(predictions_CNN, axis = 1)
    
    ResNet = tf.keras.models.load_model('weights-50epochs-mod-dataset2\\ResNet50-50epochs-bs8-mod2')
    predictions_ResNet = ResNet.predict(X)
    pred_labels_ResNet = np.argmax(predictions_ResNet, axis = 1)

    # EfficientNetRMS = tf.keras.models.load_model('weights-50epochs-mod-dataset2\\EfficientNetB4-50epochs-bs16-RMS-mod2')
    # predictions_EfficientNetRMS = EfficientNetRMS.predict(X)
    # pred_labels_EfficientNetRMS = np.argmax(predictions_EfficientNetRMS, axis = 1)

    EfficientNetAdam = tf.keras.models.load_model('weights-50epochs-mod-dataset2\\EfficientNetB4-50epochs-bs16-Adam-mod2')
    predictions_EfficientNetAdam = EfficientNetAdam.predict(X)
    pred_labels_EfficientNetAdam = np.argmax(predictions_EfficientNetAdam, axis = 1)

    VGG = tf.keras.models.load_model('weights-50epochs-mod-dataset2\\VGG16-50epochs-RMS-bs4-mod2')
    predictions_VGG = VGG.predict(X)
    pred_labels_VGG = np.argmax(predictions_VGG, axis = 1)

    print('CNN Result:' , pred_labels_CNN[0])
    print('ResNet Result:', pred_labels_ResNet[0])
    # print('EfficientNetB4-RMS Result: ', pred_labels_EfficientNetRMS)
    print('EfficientNetB4-Adam Result: ', pred_labels_EfficientNetAdam[0])
    print('VGG Result: ', pred_labels_VGG[0])
    
    combined_preds_rev = []
    res1 = pred_labels_CNN[0]
    res2 = pred_labels_ResNet[0]
    res3 = pred_labels_EfficientNetAdam[0]
    res4 = pred_labels_VGG[0]
    combined_preds_rev.append([res1, res2, res3, res4])   

    combined_preds_rev = np.array(combined_preds_rev)
    print("Shape of combined results: {0}, {1}".format(combined_preds_rev.shape, combined_preds_rev))

    pred_results_rev = majority_voting(combined_preds_rev)
    print("Image {0} belongs to class: {1}".format(filename, pred_results_rev))

    return pred_results_rev