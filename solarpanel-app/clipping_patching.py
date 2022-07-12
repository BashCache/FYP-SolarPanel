import cv2
import numpy as np

def get_patches(img_arr, size, stride):
    if size % stride != 0:
        print("size % stride must be equal 0")

    patches_list = []
    overlapping = 0
    if stride != size:
        overlapping = (size // stride) - 1

    if img_arr.ndim == 3:
        i_max = img_arr.shape[0] // stride - overlapping

        for i in range(i_max):
            for j in range(i_max):
                # print(i*stride, i*stride+size)
                # print(j*stride, j*stride+size)
                patches_list.append(img_arr[i * stride : i * stride + size, j * stride : j * stride + size])

    else:
        print("img_arr.ndim must be equal 3")

    return np.stack(patches_list)


def plot_patches(img_arr, org_img_size, name, stride=None, size=None):
    # check parameters
    if type(org_img_size) is not tuple:
        raise ValueError("org_image_size must be a tuple")

    if img_arr.ndim == 3:
        img_arr = np.expand_dims(img_arr, axis=0)

    if size is None:
        size = img_arr.shape[1]

    if stride is None:
        stride = size

    # print(filename)
    i_max = (org_img_size[0] // stride) + 1 - (size // stride)
    j_max = (org_img_size[1] // stride) + 1 - (size // stride)

    # fig, axes = plt.subplots(i_max, j_max, figsize=(i_max * 2, j_max * 2))
    # fig.subplots_adjust(hspace=0.05, wspace=0.05)
    jj = 0
    for i in range(i_max):
        for j in range(j_max):
            # axes[i, j].imshow(img_arr[jj])
            # axes[i, j].set_axis_off()
            # print(type(img_arr[jj]), img_arr[jj].shape)
            # print(jj)
            cv2.imwrite(name + str(jj) + '.png', img_arr[jj])
            jj += 1

def clipping_satellite_images(image_filename):
  x = cv2.imread('C:\\Users\\Admin\\Desktop\\Shruthi\\FYP-2022\\FYP-SolarPanel\\solarpanel-app\\static\\mod-test\\satellite-images\\' + image_filename)
#   x = cv2.resize(x, (1536, 1536), interpolation = cv2.INTER_CUBIC)
  x = np.array(x)
  x_crops = get_patches( img_arr=x, size = 1536, stride = 1536)
  print("For {0}, x shape: {1}, x-crops shape: {2}".format(image_filename, x.shape, x_crops.shape))
  filename = image_filename.split('.')
  plot_patches(img_arr = x_crops, org_img_size = (10000, 10000), name = 'C:\\Users\\Admin\\Desktop\\Shruthi\\FYP-2022\\FYP-SolarPanel\\solarpanel-app\\static\\mod-test\\images\\'+ filename[0] + '-', stride = 1536) 

def clipping_satellite_gt_images(image_filename):
  x = cv2.imread('C:\\Users\\Admin\\Desktop\\Shruthi\\FYP-2022\\FYP-SolarPanel\\solarpanel-app\\static\\mod-test\\satellite-images\\' + image_filename.split('.')[0] + '_vis.tif')
#   x = cv2.resize(x, (1536, 1536), interpolation = cv2.INTER_CUBIC)
  x = np.array(x)
  x_crops = get_patches( img_arr=x, size = 1536, stride = 1536)
  print("For {0}, x shape: {1}, x-crops shape: {2}".format(image_filename, x.shape, x_crops.shape))
  filename = image_filename.split('.')
  plot_patches(img_arr = x_crops, org_img_size = (10000, 10000), name = 'C:\\Users\\Admin\\Desktop\\Shruthi\\FYP-2022\\FYP-SolarPanel\\solarpanel-app\\static\\mod-test\\masks\\'+ filename[0] + '-', stride = 1536)

def reconstruct_from_patches(img_arr, org_img_size, stride=None, size=None):
    # check parameters
    if type(org_img_size) is not tuple:
        raise ValueError("org_image_size must be a tuple")

    if img_arr.ndim == 3:
        img_arr = np.expand_dims(img_arr, axis=0)

    if size is None:
        size = img_arr.shape[1]

    if stride is None:
        stride = size

    nm_layers = img_arr.shape[3]

    i_max = (org_img_size[0] // stride) + 1 - (size // stride)
    j_max = (org_img_size[1] // stride) + 1 - (size // stride)

    total_nm_images = img_arr.shape[0] // (i_max ** 2)
    nm_images = img_arr.shape[0]

    averaging_value = size // stride
    images_list = []
    kk = 0
    for img_count in range(total_nm_images):
        img_bg = np.zeros(
            (org_img_size[0], org_img_size[1], nm_layers), dtype=img_arr[0].dtype
        )

        for i in range(i_max):
            for j in range(j_max):
                for layer in range(nm_layers):
                    img_bg[
                        i * stride : i * stride + size,
                        j * stride : j * stride + size,
                        layer,
                    ] = img_arr[kk, :, :, layer]

                kk += 1

        images_list.append(img_bg)

    return np.stack(images_list)

def patching(x_crops_pred, x_crops_actual, x_crops_satellite):
    x_crops_pred = np.stack(x_crops_pred)
    x_crops_actual = np.stack(x_crops_actual)
    x_crops_satellite = np.stack(x_crops_satellite)
    x_crops_pred_2 = np.reshape(x_crops_pred, x_crops_pred.shape + (1,))
    x_crops_actual_2 = x_crops_actual

    x_reconstructed_pred = reconstruct_from_patches(img_arr = x_crops_pred_2, org_img_size = (1536, 1536), stride = 256)
    print("x_reconstructed shape for pred: ", str(x_reconstructed_pred.shape))

    x_reconstructed_actual = reconstruct_from_patches(img_arr = x_crops_actual_2, org_img_size = (1536, 1536), stride = 256) 
    print("x_reconstructed shape for ground truth: ", str(x_reconstructed_actual.shape))

    x_reconstructed_satellite = reconstruct_from_patches(img_arr = x_crops_satellite, org_img_size = (1536, 1536), stride = 256) 
    print("x_reconstructed shape for satellite: ", str(x_reconstructed_satellite.shape))

    return x_reconstructed_pred, x_reconstructed_actual, x_reconstructed_satellite