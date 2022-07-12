import os
from cv2 import resize
from flask import Flask,render_template, request
from preprocessing import resize_image
import tensorflow as tf
import numpy as np
import warnings
import cv2
import matplotlib.pyplot as plt
from matplotlib import pyplot

from clipping_patching import clipping_satellite_images, clipping_satellite_gt_images, reconstruct_from_patches, patching
from building_extraction import reading_images, preprocessing, draw_bb_color_filling, background_subtraction, crop_and_save
from multires import MultiResUnetBP
from evaluation_metrics import IOU, dice_coef, mcc
from majority_voting import classification_majority_voting
from boundary_detection import white_patch, auto_canny

app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
warnings.filterwarnings("ignore")
ROOFTYPES_FOLDER_UPLOAD_PATH = 'C:\\Users\\Admin\\Desktop\\Shruthi\FYP-2022\\FYP-SolarPanel\\solarpanel-app\\static\\mod-test\\buildings\\'
BUILDING_DETECTION_FOLDER_UPLOAD_PATH = 'C:\\Users\\Admin\\Desktop\\Shruthi\FYP-2022\\FYP-SolarPanel\\solarpanel-app\\static\\mod-test\\satellite-images\\'

@app.route('/')
def index():
    return render_template("home.html")

@app.route('/buildingdetection', methods = ["POST", "GET"])
def buildingdetection():
    if request.method == "POST":
        print("hi")
        for file in request.files.getlist("filename"):
            filename = file.filename
            img_path = BUILDING_DETECTION_FOLDER_UPLOAD_PATH + filename
            print(img_path)

            clipping_satellite_images(filename)
            clipping_satellite_gt_images(filename)

            test_image_paths_list_sorted, test_gt_paths_list_sorted = reading_images()
            print(test_image_paths_list_sorted, test_gt_paths_list_sorted)

            test_X, test_Y = preprocessing(test_image_paths_list_sorted, test_gt_paths_list_sorted)
            print(test_X.shape, test_Y.shape)

            MultiResModel_test = MultiResUnetBP(height=256, width=256, n_channels=3)
            MultiResModel_test.load_weights('C:\\Users\\Admin\\Desktop\\Shruthi\\FYP-2022\\FYP-SolarPanel\\solarpanel-app\\weights-buildingdetection\\multires-pro-1500images-bp-100epochs.tf')

            rows = 36
            columns = 4
            count = 1
            x_crops_pred = []
            x_crops_actual = []
            x_crops_satellite = []
            
            for image_number in range(0, 36):

                # fig = plt.figure(figsize=(15,15))
                print('Image number: {0}'.format(image_number))
                pred_y = MultiResModel_test.predict(test_X[image_number:image_number+1])
                pred_y_thresholded = (pred_y[0,:,:,0]>0.5)

                """
                fig.add_subplot(rows, columns, count)
                plt.subplot(1,4,1)
                plt.imshow(np.squeeze(test_X[image_number:image_number+1]))
                plt.title('Aerial view')

                fig.add_subplot(rows, columns, count+1)
                plt.subplot(1,4,2)
                plt.imshow(np.squeeze(test_Y[image_number:image_number+1]))
                plt.title('Original')

                fig.add_subplot(rows, columns, count+2)
                plt.subplot(1,4,3)
                plt.imshow(np.squeeze(pred_y[0]))
                plt.title('Segmented')

                fig.add_subplot(rows, columns, count+3)
                plt.subplot(1,4,4)
                plt.imshow(np.squeeze(pred_y_thresholded))
                plt.title('After thresholding')
                count += 4
                """

                x_crops_pred.append(pred_y_thresholded)
                x_crops_actual.append(test_Y[image_number:image_number+1][0])
                x_crops_satellite.append(test_X[image_number:image_number+1][0])

                print("IOU Score: {0} \nDice Coefficent: {1} \nMCC: {2}".format(IOU(test_Y[image_number:image_number+1], pred_y), dice_coef(test_Y[image_number:image_number+1], pred_y), mcc(test_Y[image_number:image_number+1], pred_y)))

                plt.show()
            
            x_reconstructed_pred, x_reconstructed_actual, x_reconstructed_satellite = patching(x_crops_pred, x_crops_actual, x_crops_satellite) 

            fig = plt.figure(figsize=(20,20))
            fig.add_subplot(3,1,1)
            plt.title('Predicted masks')
            plt.imshow(np.squeeze(x_reconstructed_pred[0]), cmap = 'gray')

            fig.add_subplot(3,1,2)
            plt.title('Ground truth')
            plt.imshow(np.squeeze(x_reconstructed_actual[0]), cmap = 'gray')

            fig.add_subplot(3,1,3)
            plt.imshow(np.squeeze(x_reconstructed_satellite[0]))

            print("IOU Score: {0} \nDice Coefficent: {1} \nMCC: {2}".format(IOU(x_reconstructed_pred, x_reconstructed_actual), dice_coef(x_reconstructed_pred, x_reconstructed_actual), mcc(x_reconstructed_pred, x_reconstructed_actual)))

            pred_i = x_reconstructed_pred[0].reshape(1536, 1536)
            pyplot.imsave(BUILDING_DETECTION_FOLDER_UPLOAD_PATH + filename.split('.')[0] + '-rs.png', x_reconstructed_satellite[0])
            pyplot.imsave(BUILDING_DETECTION_FOLDER_UPLOAD_PATH + filename.split('.')[0] + '_vis-rs.png', pred_i, cmap = 'gray')

            building_name = filename.split('.')
            img = cv2.imread(BUILDING_DETECTION_FOLDER_UPLOAD_PATH + filename, cv2.IMREAD_COLOR)
            img_mask = cv2.imread(BUILDING_DETECTION_FOLDER_UPLOAD_PATH + filename.split('.')[0] + '_vis.tif', cv2.IMREAD_COLOR)

            bb, seg_with_bounded_box, result = draw_bb_color_filling(img, img_mask)
            masked_out_new = background_subtraction(img, seg_with_bounded_box)
            crop_and_save(bb, masked_out_new, building_name[0])

    return render_template("buildingdetection.html")

@app.route('/classify', methods = ["POST", "GET"])
def classify():
    if request.method == "POST":
        print("hi")
        for file in request.files.getlist("filename"):
            filename = file.filename
            img_path = ROOFTYPES_FOLDER_UPLOAD_PATH + filename
            print(img_path)
            X = resize_image(img_path)
            print(X.shape)

            result = classification_majority_voting(filename, X)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            image_white_patch = white_patch(img, 85)
            print(image_white_patch.shape)
            gray = cv2.cvtColor(image_white_patch, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)

            fg = cv2.addWeighted(blurred, 1.5, gray, -0.5, 0)
            kernel_sharp = np.array((
                [-1, -1, -1],
                [-1, 9, -1],
                [-1, -1, -1]), dtype='int')
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            laplacian = laplacian.clip(min=0)

            auto = auto_canny(fg)
            auto1 = auto_canny(blurred)
            im = cv2.filter2D(auto, -1, kernel_sharp)
            # dst = cv2.addWeighted(gray, 0.5, auto, 0.5, 0)
            # dst1 = cv2.addWeighted(gray, 0.5, auto1, 0.5, 0)
            x = laplacian.astype(np.uint8)

            auto2 = auto_canny(x)
            im1 = cv2.filter2D(auto2, -1, kernel=kernel_sharp)

            cv2.imwrite('.\Without_blurring.jpg', auto)
            cv2.imwrite('.\After_blurring.jpg', auto1)
            cv2.imwrite('.\Laplacian_Smoothing.jpg', laplacian)

    return render_template("upload.html")

if __name__ == "__main__":
    app.run(port = 3000, debug = True)