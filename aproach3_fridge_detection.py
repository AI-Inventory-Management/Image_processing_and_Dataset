from matplotlib import test
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
import time

def generate_tresh_combinations(label_img, num_labels):
    unique_labels = np.unique(label_img)
    step = int(255/num_labels)

    result_1 = label_img
    for i in range(num_labels):
        result_1 = np.where(label_img == unique_labels[i], (i+1)*step, result_1)
    
    j = 1
    tresholds = []
    resulting_images = [result_1]
    while (j+1)*step <= 255:
        tresholds.append( int((j*step + (j+1)*step)/2) )
        j += 1

    print( "tresholds are: {t}".format(t=tresholds) )
    for i in range(num_labels):
        if i == 0:
            lower_bound = 0
            upper_bound = tresholds[0]
        elif i == num_labels-1:
            lower_bound = tresholds[-1]
            upper_bound = 255
        else:
            lower_bound = tresholds[i-1]
            upper_bound = tresholds[i]
        result_n = cv.inRange(result_1, (lower_bound), (upper_bound))
        contours, _ = cv.findContours(result_n, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        #contours = [i for i in contours if len(i) == 4]
        result_n = cv.merge([result_n, result_n, result_n])
        #result_n = cv.drawContours(result_n, contours, -1, (0,255,0), 3)
        image_area = result_n.shape[0]*result_n.shape[1]
        dim = (result_n.shape[1]//2, result_n.shape[0]//2)

        new_contours = []
        for cnt in contours:        
            area = cv.contourArea(cnt)         
            if area > image_area*(1/20):                   
                new_contours.append(cnt)
        
        for i in range(len(new_contours)):
            result_n_copy = result_n.copy()
            result_n_copy = cv.drawContours(result_n_copy, new_contours, i, (0,0,255), 3)
            result_n_copy = cv.resize(result_n_copy, dim)
            cv.imshow("coutour", result_n_copy)
            print("i = " + str(i)) 
            cv.waitKey(0)
        
        resulting_images.append(result_n)
    return resulting_images
    


test_images_dir = "./test1_images/"
raw_images_dirs = []
for image in os.listdir(test_images_dir):
    if image.endswith(".jpg"):
        raw_images_dirs.append(os.path.join(test_images_dir, image))

for image_name in raw_images_dirs:
    test_image = cv.imread(image_name)
    
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
    res2 = cv.resize(res2, ( int(test_image.shape[1]/2) , int(test_image.shape[0]/2) ) )

    print(np.unique(res2))
    print("labels shape is: {s}".format(s=label.shape))
    print("labels unique values: {u}".format(u=np.unique(label)))
    cv.imshow("segmented", res2)
    label_img = np.reshape(label, test_image.shape[:2] )
    label_img = np.uint8(label_img)
    
    processed_images = generate_tresh_combinations(label_img, K)
    
    """
    # displaying processed images after aplying tresh
    for i in range(len(processed_images)):
        img = cv.resize(processed_images[i], ( int(test_image.shape[1]/2) , int(test_image.shape[0]/2) ) )
        cv.imshow("processed " + str(i), img)
    cv.waitKey(0)
    """
    
