from typing_extensions import final
import numpy as np
import cv2 as cv
import os

class FridgeNotFoundException(Exception):
    """
    Exception that must be thrown when fridge canot be found using 
    our image processing pipeline
    """
    def __init__(self):
        super().__init__("Fridge couldn't be found, please ensure that camera is pointing to a fridge")

test_images_dir = "./test2_images/"

def filter_coutours_by_area(contours, hierarchy, area):
    new_contours = []
    new_hierarchy = []
    for i in range(len(contours)):
        cnt = contours[i]
        cnt_hierarchy = hierarchy[0][i,:].tolist()
        if cv.contourArea(cnt) > area:
            new_contours.append(cnt)
            new_hierarchy.append(cnt_hierarchy)
    new_contours = tuple(new_contours)
    new_hierarchy = np.array([new_hierarchy])
    return (new_contours, new_hierarchy)

def sort_contours_by_area(contours, hierarchy):
    areas = list(map(cv.contourArea, contours))
    array_to_sort = list(zip(areas, contours, hierarchy[0].tolist()))
    array_to_sort.sort(key=lambda x:x[0], reverse=True)
    _, sorted_contours, sorted_hierarchies = zip(*array_to_sort)
    sorted_hierarchies = list(sorted_hierarchies)
    sorted_hierarchies = np.array([sorted_hierarchies])
    return (sorted_contours, sorted_hierarchies)

def filter_contours_to_only_inner(contours, hierarchy):
    does_not_have_next_child = hierarchy[0,:,2] == -1
    does_not_have_next_child = does_not_have_next_child.tolist()
    new_contours = []
    new_hierarchy = []
    for i in range(len(does_not_have_next_child)):
        cnt = contours[i]
        cnt_hierarchy = hierarchy[0,i,:].tolist()
        if does_not_have_next_child[i]:
            new_contours.append(cnt)
            new_hierarchy.append(cnt_hierarchy)
    new_contours = tuple(new_contours)
    new_hierarchy = np.array([new_hierarchy])
    return (new_contours, new_hierarchy)

def find_fridge_content_box(image):
    height, width = image.shape[:2]
    morph_kernel_size = int(min(width,height)*0.02)
    morph_kernel = np.ones((morph_kernel_size,morph_kernel_size),np.uint8)

    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    fridge_treshold_1 = cv.inRange(hsv, (160, 70, 50), (180, 255,255))
    fridge_treshold_2 = cv.inRange(hsv, (0, 70, 50), (10, 255,255))
    fridge_treshold = cv.bitwise_or(fridge_treshold_1, fridge_treshold_2)
    fridge_treshold = cv.morphologyEx(fridge_treshold, cv.MORPH_OPEN, morph_kernel)

    contours, hierarchy = cv.findContours(fridge_treshold, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = filter_coutours_by_area(contours, hierarchy, width*height*(1/10))
    contours, hierarchy = filter_contours_to_only_inner(contours, hierarchy)
    contours, hierarchy = sort_contours_by_area(contours, hierarchy)

    if(len(contours)==0):
        raise FridgeNotFoundException()
    else:
        #cv.drawContours(image, contours, 0, (0,255,0), 3)
        min_area_rect = cv.minAreaRect(contours[0])
        box = cv.boxPoints(min_area_rect)
        box = np.int0(box)
        #cv.drawContours(image,[box],0,(255,0,0),2)
        cv.imshow("rectangle content", image)
        #print("box cords : \n {c}".format(c=box))
        #print("inclination_angle = {th}".format(th=min_area_rect[2]))
        cv.waitKey(0)
        return box, min_area_rect 

def sort_rectangle_cords(rectangle_cords):
    '''
    This function ensures that:
    - index 0 of rectangle_cords corresponds to upper left corner,
    - index 1 of rectangle_cords corresponds to upper right corner,
    - index 2 of rectangle_cords corresponds to lower left corner,
    - index 3 of rectangle_cords corresponds to lower right corner
    '''
    rectangle_cords_vector_angles = np.rad2deg(np.arctan(rectangle_cords[:,1]/rectangle_cords[:,0]))
    rectangle_cords_vector_lenghts = np.linalg.norm(rectangle_cords, axis=1)
    cords = rectangle_cords.tolist()
    cords_vector_data = list(zip(rectangle_cords_vector_angles, rectangle_cords_vector_lenghts, cords))
    cords_vector_data.sort(key=lambda x:x[0])
    upper_cords = cords_vector_data[:2]
    lower_cords = cords_vector_data[2:4]
    upper_cords.sort(key=lambda x:x[1])
    lower_cords.sort(key=lambda x:x[1])
    cords_vector_data_sorted = upper_cords + lower_cords
    _, _, rectangle_cords_sorted = zip(*cords_vector_data_sorted)
    rectangle_cords_sorted = np.array(rectangle_cords_sorted)
    return rectangle_cords_sorted

def rotate_fridge_content(image, content_cords, content_rectangle_data):
    m = (content_cords[1,1]-content_cords[0,1])/(content_cords[1,0]-content_cords[0,0])
    angle = np.rad2deg(np.arctan(m))
    fridge_content_center = content_rectangle_data[0]
    ones = np.ones((content_cords.shape[0], 1))
    rotated_image = image.copy()
    rotated_cords = content_cords.copy()

    rot_mat = cv.getRotationMatrix2D(fridge_content_center, angle, 1.0)
    rotated_image = cv.warpAffine(rotated_image, rot_mat, image.shape[1::-1], flags=cv.INTER_LINEAR)
    rotated_cords = np.concatenate((rotated_cords, ones), axis=1)
    rotated_cords = (rot_mat@(rotated_cords.T)).T
    rotated_cords = np.uint(rotated_cords)
    return (rotated_image, rotated_cords)

for image_name in os.listdir(test_images_dir):
    if image_name.endswith(".jpg"):
        image_dir = os.path.join(test_images_dir, image_name)
        image = cv.imread(image_dir)
        height, width = image.shape[:2]
        reduced_dims = ( width//2 , height//2 )

        new_image = cv.resize(image, reduced_dims)
        content_rectangle_cords, content_rectangle = find_fridge_content_box(new_image)
        content_rectangle_cords = sort_rectangle_cords(content_rectangle_cords)

        rotated_image, final_content_cords = rotate_fridge_content(new_image, content_rectangle_cords, content_rectangle)
        content_image = rotated_image[final_content_cords[0,1]:final_content_cords[3,1], final_content_cords[0,0]:final_content_cords[3,0]]
        cv.imshow("content image", content_image)
        cv.waitKey(0)
        #print(final_content_cords)
        