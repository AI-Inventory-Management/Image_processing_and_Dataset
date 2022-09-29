import numpy as np
import cv2 as cv
import os
import pandas as pd
import random

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

full_df = pd.concat([alex_df, javier_df, jose_df])
print("full_df shape is: {s}".format(s = full_df.shape))

corner_entries = full_df[full_df.is_fridge_corner == 1.0]
print("corner entries shape is: {s}".format(s = corner_entries.shape))
print("corner entries indexes are is: {s}".format(s = corner_entries.index))
corner_entries_dirs = corner_entries.image_dir.tolist()
corner_entries_indexes = corner_entries.index.tolist()
print("corner entries indexes are is: {s}".format(s = corner_entries_indexes))

#print(corner_entries_dirs)
# ====== CLEANING DATASET BASED ON HUMAN VISION ======
corners_to_delete = []
for i in range(len(corner_entries_dirs)):
    corner = cv.imread(corner_entries_dirs[i])
    cv.imshow("corner",corner)
    pressed_key = cv.waitKey(0)
    if pressed_key == 0x66:
        # when pressed key is 'f' that data is scheduled to be deleted
        corners_to_delete.append(corner_entries_indexes[i])

full_df = full_df.drop(labels=corners_to_delete, axis=0)
corner_entries = full_df[full_df.is_fridge_corner == 1.0]
corner_entries_dirs = corner_entries.image_dir.tolist()

# ====== MAKING DATA AUGMENTATION FOR DATA WITH CORNERS ======
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

#for image_name in corner_entries_dirs:
#image = cv.imread(image_name)
image = cv.imread(corner_entries_dirs[0])
new_img1 = brightness(image, 0.4, 0.6)
new_img2 = brightness(image, 0.6, 0.8)
new_img3 = brightness(image, 0.8, 0.99)
cv.imshow("darker 1", new_img1)
cv.imshow("darker 2", new_img2)
cv.imshow("darker 3", new_img3)
cv.waitKey(0)




