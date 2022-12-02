"""
Product Counter.

Classes:
    FridgeContentCounter
    
Author:
    Alejandro Dominguez
    
"""
#_________________________________Libraries____________________________________
import numpy as np
import cv2 as cv
import os
import tensorflow.keras.models as models
import tensorflow.keras.backend as backend
import json
import serial

from FridgeContentDetector import *

#__________________________________Classes_____________________________________
class FridgeContentCounter():
    """
    Class to count fridge products.
    
    ...
    
    Attributes
    ----------
    ser : Serial
        Serial connection.
    
    demo_images_dir : String
        Path to demo images folder.
    
    classifier_input_image_shape : tuple
        Dimensions of classifiers.
    
    labels : list
        Labels of products.
    
    ean : list
        Eans of products.
        
    content_detector : FridgeContentDetector
        Fridge content detector
    
    prev_pred : np array
        Previous predictions.
    
    model_path : string
        Path to model 1.
    
    model_path2 : string
        Path to model 2.
    
    model_path3 : string
        Path to model 3
    
    thresh : int
        Threshold for classification.
    
    model : Model
        Model 1.
    
    model2 : Model
        Model 2
        
    model3 : Model
        Model 3
    
    model_output_layer : Model Output
        Model 1 output layer.
    
    model_output_layer2 : Model Output
        Model 2 output layer.
        
    model_output_layer3 : Model Output
        Model 3 output layer.
    
    fridge_cols : int
        Columns of fridge.
        
    fridge_rows : int
        Rows of fridge.
        
    Methods
    -------
    show_count_result(label, max_pred, cell_num, cell):
        Show prediction of cell in image.
    
    get_ultrasonic_count():
        Read serial connection for ultrasonic values.
    
    get_classification(raw_image, classifier_input_image_shape, verbose = False):
        Clasify contents of fridge in image.
        
    get_content_count(raw_image, verbose = False):
        Make count of contents of fridge.
    
    update_software(verbose = False):
        Update models and values of system.
        
    run_demo(verbose = True):
        Demonstrate class functionality.
        
    """ 
    
    def __init__(self, demo_images_dir = "../test1_images"):  
        """
        Contruct class attributes.

        Parameters
        ----------
        demo_images_dir : string, optional
            Path to test images folder. The default is "../test1_images".

        Returns
        -------
        None.

        """
        self.ser = None
        self.demo_images_dir = demo_images_dir
        self.classifier_input_image_shape = (150,420)
        self.labels = []
        self.ean = []
        self.content_detector = FridgeContentDetector()        
        
        with open( os.path.join( os.path.dirname(__file__), "./data/product_data.json"), 'r') as f:
            data = json.load(f)
            f.close()
            self.labels = data["labels"]
            self.ean = data["eans"]
        
        self.prev_pred = np.zeros((8, len(self.labels) - 1))
        self.prev_pred[:, 6] = 1
        
        with open( os.path.join(os.path.dirname(__file__), "./data/product_data.json"), 'w') as f:
            data["prev_pred"] = self.prev_pred.tolist()
            json.dump(data, f)
        
        self.model_path = ""
        self.model_path2 = ""
        self.model_path3 = ""        
        self.thresh = 0.5
        
        with open( os.path.join(os.path.dirname(__file__), "./data/model_data.json"), 'r') as f:
            data = json.load(f)
            f.close
            self.model_path = os.path.join(os.path.dirname(__file__), data["model_path"])
            self.model_path2 = os.path.join(os.path.dirname(__file__), data["model_path2"])
            self.model_path3 = os.path.join(os.path.dirname(__file__), data["model_path3"])            
            self.thresh = data["thresh"]
        
        self.model = models.load_model(self.model_path)
        self.model2 = models.load_model(self.model_path2)
        self.model3 = models.load_model(self.model_path3)
        
        self.fridge_cols = 4
        self.fridge_rows = 2
        
        try:
            with open( os.path.join(os.path.dirname(__file__), "./data/fridge_data.json"), 'r') as f:
               data = json.load(f)
               f.close()
               self.fridge_rows, self.fridge_cols = tuple(map(int,data["fridge_dimensions"]))
        except:
            pass
    
    def show_count_result(self, label, max_pred, cell_num, cell):
        """
        Show results of predictions.

        Parameters
        ----------
        label : string
            Predicted label.
            
        max_pred : float
            Level of ceirtanty in prediction.
            
        cell_num : int
            Number of cell.
            
        cell : cv2 Image
            Cell image.

        Returns
        -------
        None.

        """
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
        """
        Read serial connection for ultrasonic information.

        Returns
        -------
        list
            Products per cell.

        """
        numSens = self.fridge_rows * self.fridge_cols
        if self.ser is None:
            try:
                self.ser = serial.Serial("/dev/ttyACM0", 9600)
            except serial.serialutil.SerialException:
                print("Error: Arduino disconnected")
                return [1]*numSens
                        
        values = []        
        while len(values) != numSens:
            line = self.ser.readline()
            values = line.split()
            
        quantity = []
        for value in values:
            if int(value) > 20:
                quantity.append(1)
            elif int(value) > 12:
                quantity.append(2)
            else:
                quantity.append(3)
            
        return quantity
         
    def get_classification(self, raw_image, classifier_input_image_shape, verbose = False):
        """
        Classify products in fridge.

        Parameters
        ----------
        raw_image : cv2 Image
            Image of fridge.
            
        classifier_input_image_shape : tuple
            Shape of output of classifier.
            
        verbose : bool, optional
            For further information. The default is False.

        Returns
        -------
        contents : list
            Clasification of each cell.

        """
        contents = []        
        cell_num = 0
        try:
            content_cells = self.content_detector.get_fridge_cells(raw_image, self.fridge_rows, self.fridge_cols, classifier_input_image_shape)            
            print("Fridge found")
            
            for cell in content_cells:                
                cell = cell/255.0
                expanded_cell = np.expand_dims(cell, axis = 0)                
                pred = self.model.predict(expanded_cell)[0]
                pred2 = self.model2.predict(expanded_cell)[0]
                pred3 = self.model3.predict(expanded_cell)[0]
                votaciones = [0,0,0,0,0,0,0,0,0,0,0,0]
                votaciones[int(np.argmax(pred))] = votaciones[int(np.argmax(pred))]+1
                votaciones[int(np.argmax(pred2))] = votaciones[int(np.argmax(pred2))]+1
                votaciones[int(np.argmax(pred3))] = votaciones[int(np.argmax(pred3))]+1
                print(votaciones)
                max_value = max(votaciones)
                max_pred = np.amax(pred)
                max_pred2 = np.amax(pred2)
                max_pred3 = np.amax(pred3)               
                label_index = votaciones.index(max_value)
                max_pred_gen = []
                if np.argmax(pred) == label_index:
                    max_pred_gen.append(max_pred)
                elif np.argmax(pred2) == label_index:
                    max_pred_gen.append(max_pred2)
                elif np.argmax(pred3) == label_index:
                    max_pred_gen.append(max_pred3)
                max_pred_average = sum(max_pred_gen)/len(max_pred_gen)
                #max_pred_gen = max(max_pred_gen)
                prev_pred_temp = [0]*len(votaciones)
                prev_pred_temp[label_index] = max_pred_average
                self.prev_pred[cell_num] = prev_pred_temp                                

                if max_pred_average < self.thresh:
                    label = self.labels[-1]
                else:
                    label = self.labels[int(label_index)]
                    print(label)

                if verbose:                    
                    self.show_count_result(label, max_pred, cell_num, cell)
                                                    
                contents.append(label)
                cell_num += 1

            with open( os.path.join(os.path.dirname(__file__), "./data/product_data.json"), 'r') as f:
                data = json.load(f)
                f.close()
                
            with open( os.path.join(os.path.dirname(__file__), "./data/product_data.json"), 'w') as f:
                data["prev_pred"] = self.prev_pred.tolist()
                json.dump(data, f)
                
        except FridgeNotFoundException:
            
            if verbose:
                print("Fridge not found")

            for i in range(len(self.prev_pred)):
                sum_preds = self.prev_pred[i]
                max_pred = np.amax(sum_preds)
                
                if max_pred < self.thresh:
                    label = self.labels[-1]
                else:
                    label = self.labels[int(np.where(sum_preds == max_pred)[0][0])]
                
                contents.append(label)
        return contents
    
    def get_content_count(self, raw_image, verbose = False):
        """
        Add and return sum of count of fridge contents.

        Parameters
        ----------
        raw_image : cv2 Image
            Image of fridge.
            
        verbose : bool, optional
            For further information. The default is False.

        Returns
        -------
        ean_count : dict
            Total sum for each registered product.

        """
        ultrasonic_count = self.get_ultrasonic_count()
        contents = self.get_classification(raw_image, self.classifier_input_image_shape, verbose)
        
        content_count = {}
        for org_label in self.labels:            
            content_count[org_label] = 0
        
        ean_count = {}
        
        for num in self.ean:            
            ean_count[num] = 0
        
        i = 0
        for label in contents:
            if label in content_count:
                content_count[label] += int(ultrasonic_count[i])
                ean_count[self.ean[self.labels.index(label)]] += int(ultrasonic_count[i])
            else:
                content_count[label] = int(ultrasonic_count[i])
                ean_count[self.ean[self.labels.index(label)]] = int(ultrasonic_count[i])
            i += 1
        
        content_count.pop("vacio")
        ean_count.pop("0")        

        if verbose:
            print("content count:")
            print(content_count)

        return ean_count
    
    def update_software(self, verbose = False):
        """
        Update models and values of system.

        Parameters
        ----------
        verbose : bool, optional
            For further information. The default is False.

        Returns
        -------
        None.

        """
        with open( os.path.join(os.path.dirname(__file__), "./data/product_data.json"), 'r') as f:
            data = json.load(f)
            f.close()
            self.labels = data["labels"]
            self.ean = data["eans"]
        
        with open( os.path.join(os.path.dirname(__file__), "./data/product_data.json"), 'r') as f:
            data = json.load(f)
            f.close()
        
        with open( os.path.join(os.path.dirname(__file__), "./data/product_data.json"), 'w') as f:
            data["prev_pred"] = self.prev_pred.tolist()
            json.dump(data, f)
            
        with open( os.path.join(os.path.dirname(__file__), "./data/model_data.json"), 'r') as f:
            data = json.load(f)
            f.close
            self.model_path = data["model_path"]
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
            print("thresh")
            print(self.thresh)
            print("Fridge software updated")
            print()
            print("Last saved predictions:")
            print(self.prev_pred)
    
    def run_demo(self, verbose = True):
        """
        Demonstrate functionality of class.

        Parameters
        ----------
        verbose : bool, optional
            For further information. The default is True.

        Returns
        -------
        None.

        """
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
             
#____________________________________Main______________________________________
if __name__ == "__main__":
    product_counter = FridgeContentCounter()
    product_counter.run_demo(verbose = True)
