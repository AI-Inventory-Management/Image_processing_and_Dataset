from matplotlib import test
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
import time

test_images_dir = "./test1_images/"
raw_images_dirs = []
for image in os.listdir(test_images_dir):
    if image.endswith(".jpg"):
        raw_images_dirs.append(os.path.join(test_images_dir, image))

for image_name in raw_images_dirs:
    test_image = cv.imread(image_name)
    #gray_image = cv.cvtColor(test_image, cv.COLOR_BGR2GRAY)
    """
    kernel = np.ones((5, 5), np.uint8)
    gray_image = cv.erode(gray_image, kernel)
    gray_image = cv.erode(gray_image, kernel)
    gray_image = cv.erode(gray_image, kernel)
    gray_image = cv.erode(gray_image, kernel)
    opening = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
    """
    """
    kernel = np.array([[0, -1, 0],
                   [-1, 5,-1],
                   [0, -1, 0]])
    gray_image = cv.filter2D(src=gray_image, ddepth=-1, kernel=kernel) # we sharpen the image
    gray_image = cv.filter2D(src=gray_image, ddepth=-1, kernel=kernel) # we sharpen the image
    #gray_image = cv.filter2D(src=gray_image, ddepth=-1, kernel=kernel) # we sharpen the image

    gray = np.float32(gray_image)
    dst = cv.cornerHarris(gray,2,3,0.04)
    #result is dilated for marking the corners, not important
    dst = cv.dilate(dst,None)
    print(len(dst))
    # Threshold for an optimal value, it may vary depending on the image.
    gray_image = cv.merge([gray_image, gray_image, gray_image])
    gray_image[dst>0.25*dst.max()]=[0,0,255]

    """
    #gray_image = cv.resize(gray_image, ( int(gray_image.shape[1]/2) , int(gray_image.shape[0]/2) ) )

    Z = test_image.reshape((-1,3))
    # convert to np.float32
    Z = np.float32(Z)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 3
    ret,label,center=cv.kmeans(Z,K,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((test_image.shape))
    #cv.imshow('res2',res2)


    res2 = cv.resize(res2, ( int(test_image.shape[1]/2) , int(test_image.shape[0]/2) ) )

    cv.imshow("processed", res2)
    cv.waitKey(0)
    print("next")
