import numpy as np
import cv2 as cv
import os
import pandas as pd
import time

class FridgeNotFoundException(Exception):
    """
    Exception that must be thrown when fridge canot be found using 
    our image processing pipeline
    """
    def __init__(self):
        super().__init__("Fridge couldn't be found, please ensure that camera is pointing to a fridge")

class FridgeContentDetector():
    def __init__(self) -> None:
        self.demo_images_dir = "./sodas_dataset_raw/session 1/"
        self.sodas_sessions_dir = "./sodas_dataset_raw/session {session_num}/"
        self.sodas_sessions_dir_segmented = "./sodas_dataset_raw_segmented/session_{session_num}/"
        #self.sodas_final_dataset_dir = "./sodas_dataset/"
        self.sodas_final_dataset_dir = "./sodas_dataset_w_margin/"

    def correct_image_brightness(self, raw_image):
        height, width = raw_image.shape[:2]
        hsv = cv.cvtColor(raw_image, cv.COLOR_BGR2HSV)
        h,s,v = cv.split(hsv)
        brightness_ratio_1 = 0.70
        brightness_ratio_2 =  (v.sum())/(height*width*255.0)
        m = int((brightness_ratio_1 - brightness_ratio_2)*255.0)
        v = np.int64(v)
        v = v+m
        v[v>255] = 255
        v[v<0] = 0
        v = np.uint8(v)
        new_image = cv.merge((h,s,v))
        new_image = cv.cvtColor(new_image, cv.COLOR_HSV2BGR)
        #cv.imshow("corrected_brightness", new_image)
        return new_image
    
    def filter_coutours_by_area(self, contours, hierarchy, area):
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

    def sort_contours_by_area(self, contours, hierarchy):
        areas = list(map(cv.contourArea, contours))
        array_to_sort = list(zip(areas, contours, hierarchy[0].tolist()))
        array_to_sort.sort(key=lambda x:x[0], reverse=True)
        _, sorted_contours, sorted_hierarchies = zip(*array_to_sort)
        sorted_hierarchies = list(sorted_hierarchies)
        sorted_hierarchies = np.array([sorted_hierarchies])
        return (sorted_contours, sorted_hierarchies)

    def filter_contours_to_only_inner(self, contours, hierarchy):
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

    def find_fridge_content_box(self, image):
        height, width = image.shape[:2]
        opening_kernel_size = int(min(width,height)*0.025)
        closing_kernel_size = int(min(width,height)*0.01)
        opening_kernel = np.ones((opening_kernel_size,opening_kernel_size),np.uint8)
        closing_kernel = np.ones((closing_kernel_size,closing_kernel_size),np.uint8)

        hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        fridge_treshold_1 = cv.inRange(hsv, (160, 50, 50), (180, 255,255))
        fridge_treshold_2 = cv.inRange(hsv, (0, 50, 50), (10, 255,255))
        fridge_treshold = cv.bitwise_or(fridge_treshold_1, fridge_treshold_2)
        fridge_treshold = cv.morphologyEx(fridge_treshold, cv.MORPH_OPEN, opening_kernel)
        fridge_treshold = cv.morphologyEx(fridge_treshold, cv.MORPH_CLOSE, closing_kernel)

        #cv.imshow("tresholded image no cnt", fridge_treshold)          
        contours, hierarchy = cv.findContours(fridge_treshold, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
        contours, hierarchy = self.filter_coutours_by_area(contours, hierarchy, width*height*(1/10))
        if(len(contours)==0):
            raise FridgeNotFoundException()
        contours, hierarchy = self.filter_contours_to_only_inner(contours, hierarchy)
        if(len(contours)==0):
            raise FridgeNotFoundException()
        contours, hierarchy = self.sort_contours_by_area(contours, hierarchy)

        fridge_treshold_copy = fridge_treshold.copy()
        fridge_treshold_copy = cv.merge((fridge_treshold_copy, fridge_treshold_copy, fridge_treshold_copy))
        #cv.drawContours(fridge_treshold_copy, contours, -1, (0,255,0), 3)
        #cv.imshow("tresholded image", fridge_treshold_copy)            
        min_area_rect = cv.minAreaRect(contours[0])
        box = cv.boxPoints(min_area_rect)
        box = np.int0(box)            
        return box, min_area_rect

    def segmentate_fridge(self, image):
        ''' This function is supposed to have the code that binarizes the input image found on
        find_frigde_content_box if that code is changed, please be sure to copy it to this function '''
        height, width = image.shape[:2]
        opening_kernel_size = int(min(width,height)*0.025)
        closing_kernel_size = int(min(width,height)*0.01)
        opening_kernel = np.ones((opening_kernel_size,opening_kernel_size),np.uint8)
        closing_kernel = np.ones((closing_kernel_size,closing_kernel_size),np.uint8)

        hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        fridge_treshold_1 = cv.inRange(hsv, (160, 50, 50), (180, 255,255))
        fridge_treshold_2 = cv.inRange(hsv, (0, 50, 50), (10, 255,255))
        fridge_treshold = cv.bitwise_or(fridge_treshold_1, fridge_treshold_2)
        fridge_treshold = cv.morphologyEx(fridge_treshold, cv.MORPH_OPEN, opening_kernel)
        fridge_treshold = cv.morphologyEx(fridge_treshold, cv.MORPH_CLOSE, closing_kernel)
        
        return fridge_treshold

    def sort_rectangle_cords(self, rectangle_cords):
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

    def rotate_fridge_content(self, image, content_cords, content_rectangle_data):
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

    def cluster_image(self, raw_image):
        '''
        Attempt to improve fridge detection using k means clustering to make the fridges
        red color simpler (invariant to brightness and darkness)
        '''
        hsv_image = cv.cvtColor(raw_image, cv.COLOR_BGR2HSV)
        gray_image = cv.cvtColor(raw_image, cv.COLOR_BGR2GRAY)
        h, s, v = cv.split(hsv_image)
        h = cv.resize(h, (int(raw_image.shape[1]/2) , int(raw_image.shape[0]/2)) )
        cv.imshow("hue image", h)
        gray_image = cv.resize(gray_image, (int(raw_image.shape[1]/2) , int(raw_image.shape[0]/2)) )
        cv.imshow("hue image", gray_image)
        Z = h.reshape((-1,1))
        # convert to np.float32
        Z = np.float32(Z)
        # define criteria, number of clusters(K) and apply kmeans()
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 7
        ret,label,center=cv.kmeans(Z,K,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)
        # Now convert back into uint8, and make original image
        print("center type: {t}".format(t = type(center)))
        print("center shape {s}".format(s = center.shape))
        print("label type: {t}".format(t = type(label)))
        print("label shape {s}".format(s = label.shape))
        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((h.shape))
        res2 = cv.merge( [h, np.ones(h.shape, dtype=np.uint8)*255, np.ones(h.shape, dtype=np.uint8)*255] )
        res2 = cv.cvtColor(res2, cv.COLOR_HSV2BGR) 
        res2 = cv.resize(res2, (int(raw_image.shape[1]/2) , int(raw_image.shape[0]/2)) )
        cv.imshow("corrected image", res2)
        return res2

    def get_fridge_content_image(self, raw_image):
        #new_raw_image = self.correct_image_brightness(raw_image)
        #raw_image_corrected = self.cluster_image(raw_image)
        content_rectangle_cords, content_rectangle = self.find_fridge_content_box(raw_image)
        content_rectangle_cords = self.sort_rectangle_cords(content_rectangle_cords)
        rotated_image, final_content_cords = self.rotate_fridge_content(raw_image, content_rectangle_cords, content_rectangle)
        #content_image = rotated_image[final_content_cords[0,1]:final_content_cords[3,1], final_content_cords[0,0]:final_content_cords[3,0]]
        content_image = rotated_image[final_content_cords[0,1] +5 :final_content_cords[3,1] +5 , final_content_cords[0,0]:final_content_cords[3,0]]
        
        content_image = cv.resize(content_image, (int(content_image.shape[1]/2) , int(content_image.shape[0]/2)) )
        cv.imshow("content image", content_image)
        return content_image
    
    def get_fridge_cells(self, raw_image, fridge_rows_count, fridge_columns_count, output_shape=(150,420)):
        fridge_content = self.get_fridge_content_image(raw_image)
        fridge_content_h, fridge_content_w = fridge_content.shape[:2] 
        vertical_boundaries = np.linspace(0,fridge_content_h, fridge_rows_count+1, dtype=np.int0)
        horizontal_boundaries = np.linspace(0,fridge_content_w, fridge_columns_count+1, dtype=np.int0)
        content_cells = []
        
        for i in range(1,len(vertical_boundaries)):
            for j in range(1,len(horizontal_boundaries)):
                cell = fridge_content[vertical_boundaries[i-1]+10:vertical_boundaries[i]+10, horizontal_boundaries[j-1]:horizontal_boundaries[j]]
                cell = cv.resize(cell, output_shape)
                content_cells.append(cell)

        return content_cells

    def run_demo(self):
        for image_name in os.listdir(self.demo_images_dir):
            if image_name.endswith(".jpg"):
                image_dir = os.path.join(self.demo_images_dir, image_name)
                image = cv.imread(image_dir)
                height, width = image.shape[:2]
                reduced_dims = ( width//2 , height//2 )
                image = cv.resize(image, reduced_dims)
                try:
                    content_cells = self.get_fridge_cells(image, 2, 4)
                    for i in range(len(content_cells)):
                        cv.imshow("cell " + str(i), content_cells[i])
                except FridgeNotFoundException:
                    print("fridge not found on image: {im_name}".format(im_name=image_name))
                cv.waitKey(0)
    
    def generate_dataset(self):
        df = pd.read_excel('./sodas_dataset_raw/session classes.xlsx')
        #df = pd.read_excel('./sodas_dataset_raw/session classes w margin.xlsx')
        df.drop('Session number', axis=1, inplace=True)
        labels = ['fresca lata 355 ml', 'sidral mundet lata 355 ml', 'fresca botella de plastico 600 ml', 'fuze tea durazno 600 ml', 'power ade mora azul botella de plastico 500 ml', 'delaware punch lata 355 ml', 'vacio', 'del valle durazno botella de vidrio 413 ml', 'sidral mundet botella de plastico 600 ml', 'coca cola botella de plastico 600 ml', 'power ade mora azul lata 453 ml', 'coca cola lata 355 ml']
        num_to_label = {}
        label_to_num = {}
        for i in range(len(labels)):
            num_to_label[i] = labels[i]
            label_to_num[labels[i]] = i
        
        for index, row in df.iterrows():
            session_labels = row.tolist()
            session_dir = self.sodas_sessions_dir.format(session_num=index+1)
            #session_dir = self.sodas_sessions_dir.format(session_num=index+52)
            for image_name in os.listdir(session_dir):
                if image_name.endswith(".jpg"):
                    image_dir = os.path.join(session_dir, image_name)
                    image = cv.imread(image_dir)
                    try:
                        content_cells = self.get_fridge_cells(image, 2, 4)
                        for i in range(len(content_cells)):
                            saving_path = os.path.join(self.sodas_final_dataset_dir, str(label_to_num[session_labels[i]]), str(time.time()).replace(".", "_") + ".jpg" )
                            print(saving_path)
                            cv.imwrite(saving_path, content_cells[i])     
                    except FridgeNotFoundException:
                        print("fridge not found on image: {session_dir} {im_name}".format(session_dir = session_dir, im_name=image_name))
                    cv.waitKey(0)

    def generate_segmented_fridge_dataset(self):
        df = pd.read_excel('./sodas_dataset_raw/session classes.xlsx')
        for index, row in df.iterrows():            
            session_dir = self.sodas_sessions_dir.format(session_num=index+1)
            session_dir_segmented = self.sodas_sessions_dir_segmented.format(session_num=index+1)
            for image_name in os.listdir(session_dir):
                if image_name.endswith(".jpg"):
                    image_dir = os.path.join(session_dir, image_name)
                    image = cv.imread(image_dir)
                    segmented_image = self.segmentate_fridge(image)
                    segmented_image_name = os.path.join(session_dir_segmented, str(time.time()).replace(".", "_") + ".jpg")
                    cv.imwrite(segmented_image_name, segmented_image)
                    
        
        

if __name__ == "__main__":
    fridge_content_detector = FridgeContentDetector()
    #fridge_content_detector.run_demo()
    #fridge_content_detector.generate_segmented_fridge_dataset()
    
    