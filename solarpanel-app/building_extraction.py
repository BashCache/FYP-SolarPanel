import glob
import os
import cv2
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

scaler = MinMaxScaler()

def reading_images():
    test_image_paths_list = []
    test_gt_paths_list = []
    test_image_path = 'C:\\Users\\Admin\\Desktop\\Shruthi\\FYP-2022\\FYP-SolarPanel\\solarpanel-app\\static\\mod-test\\images\\'
    test_label_path = 'C:\\Users\\Admin\\Desktop\\Shruthi\\FYP-2022\\FYP-SolarPanel\\solarpanel-app\\static\\mod-test\\masks\\'

    for img_path in glob.glob(os.path.join(test_image_path, '*.png')):
        test_image_paths_list.append(str(img_path))        
    print("Total aerial images in test set : ",len(test_image_paths_list))

    for img_path in glob.glob(os.path.join(test_label_path, '*.png')):
        test_gt_paths_list.append(str(img_path))        
    print("Total segmented mask images in test set : ",len(test_gt_paths_list))

    test_image_paths_list.sort()
    test_gt_paths_list.sort()

    test_image_paths_list_sorted = []
    test_gt_paths_list_sorted = []
    count = 0
    for i in range(len(test_image_paths_list)):
        filename = test_image_paths_list[i].split('\\')
        building_name = filename[11]
        building = building_name.split('-')[0]
        building_no = building + '-' + str(count) + '.png'
        # print(building, building_name)
        test_image_paths_list_sorted.append('C:\\Users\\Admin\\Desktop\\Shruthi\\FYP-2022\\FYP-SolarPanel\\solarpanel-app\\static\\mod-test\\images\\' + building_no)
        count += 1
        if count == 36:
            count = 0

    count = 0
    for i in range(len(test_gt_paths_list)):
        filename = test_gt_paths_list[i].split('\\')
        building_name = filename[11]
        building = building_name.split('-')[0]
        building_no = building + '-' + str(count) + '.png'
        # print(building_no)
        test_gt_paths_list_sorted.append('C:\\Users\\Admin\\Desktop\\Shruthi\\FYP-2022\\FYP-SolarPanel\\solarpanel-app\\static\\mod-test\\masks\\' + building_no)
        count += 1
        if count == 36:
            count = 0

    return test_image_paths_list_sorted, test_gt_paths_list_sorted

def preprocessing(test_image_paths_list_sorted, test_gt_paths_list_sorted):
    test_X = []
    test_Y = []
    for img in test_image_paths_list_sorted:
        img = cv2.imread(img, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (256, 256), interpolation = cv2.INTER_CUBIC)
        test_X.append(img)
        
    for img in test_gt_paths_list_sorted:
        img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (256, 256), interpolation = cv2.INTER_CUBIC)
        test_Y.append(img)

    test_X = np.array(test_X)
    test_Y = np.array(test_Y)

    test_Y = test_Y.reshape((test_Y.shape[0], test_Y.shape[1], test_Y.shape[2],1))

    test_X = scaler.fit_transform(test_X.reshape(-1, test_X.shape[-1])).reshape(test_X.shape)
    test_Y = scaler.fit_transform(test_Y.reshape(-1, test_Y.shape[-1])).reshape(test_Y.shape)

    return test_X, test_Y

def draw_bb_color_filling(img, img_mask):
  # convert to grayscale
  gray = cv2.cvtColor(img_mask, cv2.COLOR_BGR2GRAY)

  # threshold
  thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)[1]

  # get contours
  result = img_mask.copy()
  seg_with_bounded_box = img_mask.copy()
  contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  contours = contours[0] if len(contours) == 2 else contours[1]
  print("No of identified buildings: {0}".format(len(contours)))
  bb = []

  for cntr in contours:
      x,y,w,h = cv2.boundingRect(cntr)
      cv2.rectangle(result, (x, y), (x+w, y+h), (0, 0, 255), 1)
      cv2.rectangle(seg_with_bounded_box, (x, y), (x+w, y+h), (255, 255, 255), -1)
      bb.append([x,y,w,h])

  print(bb)   

  # plt.imshow(seg_with_bounded_box)
  # plt.title('Segmented mask with bounding box and color filling')
  # plt.show()

  # plt.imshow(result)
  # plt.title('Segmented mask with bounding box')
  # plt.show()

  return bb, seg_with_bounded_box, result

# Subtract mask from original satellite image
def background_subtraction(img, seg_with_bounded_box):
  new_seg_image_gray = cv2.cvtColor(seg_with_bounded_box, cv2.COLOR_RGB2GRAY) # Convert the mask to grayscale
  img_np = np.asarray(img) # Convert the PIL image to a numpy array
  masked_out = cv2.bitwise_and(img_np, img_np, mask = new_seg_image_gray) # Blend the mask
  masked_out_new = np.where(masked_out != 0, masked_out, 255) # Remove the background
  
  # plt.imshow(masked_out_new)
  # plt.title('Masked-out Image')
  # plt.show()
  
  return masked_out_new

def crop_and_save(bb, masked_out_new, building_name):
  for i in range(len(bb)):
    x = bb[i][0]
    y = bb[i][1]
    w = bb[i][2]
    h = bb[i][3]
    cv2.imwrite('C:\\Users\\Admin\\Desktop\\Shruthi\\FYP-2022\\FYP-SolarPanel\\solarpanel-app\\static\\mod-test\\buildings\\' + building_name + '_' + str(i) + '.tif', masked_out_new[y:y+h, x:x+w])