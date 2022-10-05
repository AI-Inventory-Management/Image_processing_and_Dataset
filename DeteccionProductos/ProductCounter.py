import numpy as np
import cv2 as cv
import os
import tensorflow.keras.models as models

from FridgeContentDetector import *

class FridgeContentCounter():
    def __init__(self, model_path = './can_model_normal', labels = ["Coca Cola 600ml Botella", "Fuze Tea 600ml Botella", "Powered Sabor Mora 500ml Botella", "Sidral 600ml Botella", "Fresca 100ml Botella", "Delaware 355ml Lata", "Coca Cola 355ml Lata", "Sidral 355ml Lata", "Fresca 355ml Lata","Powerade Mora 453ml Lata"]):
        self.model_path = model_path
        self.demo_images_dir = "../test3_images"
        self.labels = labels
    
    def get_content_count(self, raw_image, fridge_rows_count = 2, fridge_columns_count = 4, output_shape=(80,200)):
        fridge_content_detector = FridgeContentDetector()
        content_cells = fridge_content_detector.get_fridge_cells(raw_image, fridge_rows_count, fridge_columns_count, output_shape)
        
        # content_count = {}
        model = models.load_model(self.model_path)
        cell_num = 1
        for cell in content_cells:
            expanded_cell = np.expand_dims(cell, axis = 0)
            pred = model.predict(expanded_cell)[0]
            max_pred = np.amax(pred)
            label = self.labels[int(np.where(pred == max_pred)[0])]
            pred_lbl = label 
            pred_pred =  str(max_pred*100) + "% "
            cel_cnt = "Cell: " + str(cell_num)
            font = cv.FONT_HERSHEY_SIMPLEX
            org1 = (1, 15)
            org2 = (1, 23)
            org3 = (1, 31)
            fontScale = 0.2
            color = (255, 0, 0)
            thickness = 1
            cell = cv.putText(cell, pred_lbl, org1, font, fontScale, color, thickness)
            cell = cv.putText(cell, pred_pred, org2, font, fontScale, color, thickness)
            cell = cv.putText(cell, cel_cnt, org3, font, fontScale, color, thickness)
            cv.imshow("CellPred", cell)
            cv.waitKey(0)
            cell_num += 1
            
            
    def run_demo(self):
        for image_name in os.listdir(self.demo_images_dir):
            if image_name.endswith(".jpg"):
                image_dir = os.path.join(self.demo_images_dir, image_name)
                image = cv.imread(image_dir)
                height, width = image.shape[:2]
                reduced_dims = ( width//2 , height//2 )
                image = cv.resize(image, reduced_dims)
                self.get_content_count(image)