from hashlib import new
from turtle import width
from typing import final
import numpy as np
import cv2 as cv
import os
import pandas as pd
import random
from fridge_detector import cornerDetector
import time

alex_data_dir = "./corner_detector_dataset/alex_data.csv"
javier_data_dir = "./corner_detector_dataset/javier_data.csv"
jose_data_dir = "./corner_detector_dataset/jose_data.csv"

alex_df = pd.read_csv(alex_data_dir)
javier_df = pd.read_csv(javier_data_dir)
jose_df = pd.read_csv(jose_data_dir)

print(alex_df.head())
print("========================")
print("alex_df shape is: {s}".format(s = alex_df.shape))
print("javier_df shape is: {s}".format(s = javier_df.shape))
print("jose_df shape is: {s}".format(s = jose_df.shape))

full_df = pd.concat([alex_df, javier_df, jose_df], ignore_index=True)
print("full_df shape is: {s}".format(s = full_df.shape))

corner_entries = full_df[full_df.is_fridge_corner == 1.0]
non_corner_entries = full_df[full_df.is_fridge_corner == 0.0]
print("corner entries shape is: {s}".format(s = corner_entries.shape))
print("corner entries indexes are is: {s}".format(s = corner_entries.index))
corner_entries_dirs = corner_entries.image_dir.tolist()
corner_entries_indexes = corner_entries.index.tolist()
print("corner entries indexes are is: {s}".format(s = corner_entries_indexes))

#print(corner_entries_dirs)
# ====== CLEANING DATASET BASED ON HUMAN VISION ======
corners_to_delete = []
"""
for i in range(len(corner_entries_dirs)):
    corner = cv.imread(corner_entries_dirs[i])
    cv.imshow("corner",corner)
    pressed_key = cv.waitKey(0)
    if pressed_key == 0x66:
        # when pressed key is 'f' that data is scheduled to be deleted
        corners_to_delete.append(corner_entries_indexes[i])
"""

full_df = full_df.drop(labels=corners_to_delete, axis=0)
corner_entries = full_df[full_df.is_fridge_corner == 1.0]
corner_entries_dirs = corner_entries.image_dir.tolist()

# ====== MAKING DATA AUGMENTATION FOR DATA WITH CORNERS ======
images_dataset_dir = "./corner_detector_dataset/data_augmentation_images/"
num_saved_images = 0

def brightness(img, low, high):
    value = random.uniform(low, high)
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    hsv = np.array(hsv, dtype = np.float64)
    hsv[:,:,1] = hsv[:,:,1]*value
    hsv[:,:,1][hsv[:,:,1]>255]  = 255
    hsv[:,:,2] = hsv[:,:,2]*value 
    hsv[:,:,2][hsv[:,:,2]>255]  = 255
    hsv = np.array(hsv, dtype = np.uint8)
    img = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    return img

corner_detector = cornerDetector()
new_data_names = []
new_data_features = []
new_data_outputs = []

def forced_imwrite(dir, image):
    global num_saved_images
    saved = False
    while not saved:
        saved = cv.imwrite(dir, image)
        if saved == True:
            num_saved_images += 1


for image_name in corner_entries_dirs:
    image = cv.imread(image_name)
    #print("image shape is: {s}".format(s = image.shape))  ---- 3 channels
    new_img1 = brightness(image, 0.4, 0.6)
    new_img2 = brightness(image, 0.6, 0.8)
    new_img3 = brightness(image, 0.8, 0.99)
    new_img1_gray = cv.cvtColor(new_img1,cv.COLOR_BGR2GRAY)
    new_img2_gray = cv.cvtColor(new_img2,cv.COLOR_BGR2GRAY)
    new_img3_gray = cv.cvtColor(new_img3,cv.COLOR_BGR2GRAY)
    new_img1_features = corner_detector.get_roi_features(new_img1_gray)
    new_img2_features = corner_detector.get_roi_features(new_img2_gray)
    new_img3_features = corner_detector.get_roi_features(new_img3_gray)
    new_img1_name = images_dataset_dir + "dataAugmentation1_1" + "_" + str(time.time()).replace('.', '_') + ".jpg"
    new_img2_name = images_dataset_dir + "dataAugmentation1_2" + "_" + str(time.time()).replace('.', '_') + ".jpg"
    new_img3_name = images_dataset_dir + "dataAugmentation1_3" + "_" + str(time.time()).replace('.', '_') + ".jpg"
    new_data_names.extend([new_img1_name, new_img2_name, new_img3_name])
    new_data_features.extend([new_img1_features, new_img2_features, new_img3_features])
    new_data_outputs.extend([1.0, 1.0, 1.0])
    forced_imwrite(new_img1_name, new_img1)
    forced_imwrite(new_img2_name, new_img2)
    forced_imwrite(new_img3_name, new_img3)

new_data_features = np.array(new_data_features)
new_dim_data = {
        "image_dir": new_data_names,
        "conv_a": new_data_features[:, 0].tolist(),
        "conv_b": new_data_features[:, 1].tolist(),
        "conv_c": new_data_features[:, 2].tolist(),
        "conv_d": new_data_features[:, 3].tolist(),
        "conv_e": new_data_features[:, 4].tolist(),
        "conv_f": new_data_features[:, 5].tolist(),
        "conv_g": new_data_features[:, 6].tolist(),
        "conv_h": new_data_features[:, 7].tolist(),
        "is_fridge_corner": new_data_outputs,
        }
data_augmentation_df1 = pd.DataFrame(new_dim_data)

def rotate_image(image, angle):
    height, width = image.shape[:2] # considering a image with only 3 channels
    image_center = (int(width/2), int(height/2))
    rot_mat = cv.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv.warpAffine(image, rot_mat, (width, height), flags=cv.INTER_LINEAR)
    return result

"""image = cv.imread(corner_entries_dirs[0])
rotated_image10 = rotate_image(image, 10)
cv.imshow("rotated image", rotated_image10)
cv.waitKey(0)"""

print("data augmentation df1 shape is: {s}".format(s=data_augmentation_df1.shape))
positive_samples_after_data_augmentation = data_augmentation_df1.shape[0] + corner_entries.shape[0]
negative_samples_new_sampling_rate = ((6/4)*(positive_samples_after_data_augmentation))/(non_corner_entries.shape[0])
print("negative samples new sampling rate is {s}".format(s=negative_samples_new_sampling_rate))

shuffled_negative_samples = non_corner_entries.sample(frac=negative_samples_new_sampling_rate, random_state=1)
print(shuffled_negative_samples.shape)

final_dataframe = pd.concat([corner_entries, data_augmentation_df1, shuffled_negative_samples], ignore_index=True)
final_dataframe = final_dataframe.sample(frac=1, random_state=1).reset_index() # here we shuffle final df entries
print(final_dataframe.head(10))

while num_saved_images != data_augmentation_df1.shape[0]:
    pass

final_dataframe.to_csv('./corner_detector_dataset/final_data.csv', index=False)






