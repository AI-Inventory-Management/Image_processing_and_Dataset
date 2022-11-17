import numpy as np
import cv2 as cv
import os
import tensorflow.keras.models as models
import tensorflow.keras.backend as backend
import json
import serial

from FridgeContentDetector import *

class FridgeContentCounter():
    def __init__(self, demo_images_dir = "../test3_images"):
        self.ser = serial.Serial("/dev/ttyACM0", 9600)
        self.demo_images_dir = demo_images_dir
        self.labels = []
        self.ean = []
        self.content_detector = FridgeContentDetector()
        
        with open("./data/product_data.json", 'r') as f:
            data = json.load(f)
            f.close()
            self.labels = data["labels"]
            self.ean = data["eans"]
        
        self.prev_pred = np.zeros((8, len(self.labels) - 1))
        self.prev_pred[:, 6] = 1
        
        with open("./data/product_data.json", 'r') as f:
            data = json.load(f)
            f.close()
        
        with open("./data/product_data.json", 'w') as f:
            data["prev_pred"] = self.prev_pred.tolist()
            json.dump(data, f)
        
        self.model_path = ""
        self.alfa = 0.5
        self.beta = 0.5
        self.thresh = 1.0
        
        with open("./data/model_data.json", 'r') as f:
            data = json.load(f)
            f.close
            self.model_path = data["model_path"]
            self.alfa = data["alfa"]
            self.beta = data["beta"]
            self.thresh = data["thresh"]
        
        self.model = models.load_model(self.model_path)
        
        self.fridge_cols = 4
        self.fridge_rows = 2
        
        try:
            with open("./data/fridge_data.json", 'r') as f:
               data = json.load(f)
               f.close()
               self.fridge_cols, self.fridge_rows = data["fridge_dimensions"]
        except:
            pass
    
    def show_count_result(self, label, max_pred, cell_num, cell):
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
        cv.destroyAllWindows()
    
    def get_ultrasonic_count(self):
        numSens = self.fridge_rows * self.fridge_cols
        values = []
        while len(values) != numSens:
            line = self.ser.readline()
            values = line.split()
        
        quantity = []
        for value in values:
            if int(value) > 25:
                quantity.append(0)
            elif int(value) > 15:
                quantity.append(1)
            elif int(value) > 5:
                quantity.append(2)
            else:
                quantity.append(3)
            
        return quantity
        
    
    def get_classification(self, raw_image, output_shape, verbose = False):
        contents = []
        
        cell_num = 1
        try:
            content_cells = self.content_detector.get_fridge_cells(raw_image, self.fridge_rows, self.fridge_cols, output_shape)
            
            print("Fridge found")
            
            for cell in content_cells:
                # gray_cell = cv.cvtColor(cell, cv.COLOR_BGR2GRAY)
                cell = cell/255.0
                expanded_cell = np.expand_dims(cell, axis = 0)
                # print(expanded_cell.shape)
                # expanded_cell = np.expand_dims(expanded_cell, axis = 3)
                pred = self.model.predict(expanded_cell)[0]
                
                sum_preds = self.alfa * self.prev_pred[cell_num - 1] + self.beta * pred
                
                self.prev_pred[cell_num - 1] = pred
                
                with open("./data/product_data.json", 'r') as f:
                    data = json.load(f)
                    f.close()
                
                with open("./data/product_data.json", 'w') as f:
                    data["prev_pred"] = self.prev_pred.tolist()
                    json.dump(data, f)
                
                max_pred = np.amax(sum_preds)
                
                if max_pred < self.thresh:
                    label = self.labels[-1]
                else:
                    label = self.labels[int(np.where(sum_preds == max_pred)[0][0])]
                
                
                if verbose:                    
                    self.show_count_result(label, max_pred, cell_num, cell)
                    
                cell_num += 1
                
                contents.append(label)
                
        except FridgeNotFoundException:
            
            if verbose:
                print("Fridge not found")

            for i in range(8):
                sum_preds = self.prev_pred[i]

                max_pred = np.amax(sum_preds)
                
                if max_pred < self.thresh:
                    label = self.labels[-1]
                else:
                    label = self.labels[int(np.where(sum_preds == max_pred)[0][0])]
                
                contents.append(label)
        return contents
    
    def get_content_count(self, raw_image, output_shape=(150,420), verbose = False):
        ultrasonic_count = self.get_ultrasonic_count()
        contents = self.get_classification(raw_image, output_shape, verbose)
        
        content_count = {}
        for org_label in self.labels:
            if org_label != "vacio":
                content_count[org_label] = 0
        
        ean_count = {}
        
        for num in self.ean:
            if num != "0":
                ean_count[num] = 0
        
        for i in range(len(contents)):
            if contents[i] in content_count:
                content_count[contents[i]] += int(ultrasonic_count[i])
            else:
                content_count[contents[i]] = int(ultrasonic_count[i])
        
        content_count.pop("vacio")
        i = 0
        for content in content_count:
            ean_count[self.ean[i]] = content_count[content]
            i += 1

        if verbose:
            print(content_count)

        return ean_count
    
    def update_software(self, verbose = False):
        with open("./data/product_data.json", 'r') as f:
            data = json.load(f)
            f.close()
            self.labels = data["labels"]
            self.ean = data["eans"]
        
        with open("./data/product_data.json", 'r') as f:
            data = json.load(f)
            f.close()
        
        with open("./data/product_data.json", 'w') as f:
            data["prev_pred"] = self.prev_pred.tolist()
            json.dump(data, f)
            
        with open("./data/model_data.json", 'r') as f:
            data = json.load(f)
            f.close
            self.model_path = data["model_path"]
            self.alfa = data["alfa"]
            self.beta = data["beta"]
            self.thresh = data["thresh"]
        
        backend.clear_session()
        self.model = models.load_model(self.model_path)
        
        if verbose:
            print()
            print("Loaded.................................")
            print("lablels:")
            print(self.labels)
            print("eans:")
            print(self.ean)
            print("Model")
            print(self.model_path)
            print("alfa:")
            print(self.alfa)
            print("beta")
            print(self.beta)
            print("thresh")
            print(self.thresh)
            print("Fridge software updated")
            print()
            print("Last saved predictions:")
            print(self.prev_pred)
    
    def run_demo(self, verbose = True):
        for image_name in os.listdir(self.demo_images_dir):
            if image_name.endswith(".jpg"):
                image_dir = os.path.join(self.demo_images_dir, image_name)
                image = cv.imread(image_dir)
                height, width = image.shape[:2]
                reduced_dims = ( width//2 , height//2 )
                image = cv.resize(image, reduced_dims)
                content = self.get_content_count(image, verbose = verbose)
                print(content)
                
        cv.destroyAllWindows()
                
if __name__ == "__main__":
    product_counter = FridgeContentCounter()
    product_counter.run_demo(verbose = True)