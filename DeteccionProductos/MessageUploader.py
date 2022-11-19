from datetime import datetime
import time
import cv2 as cv
import numpy as np
import os
from threading import Event
import requests

from ProductCounter import FridgeContentCounter
from Encrypter import Encrypter

class MessageUploader ():
    def __init__(self, server = "http://192.168.195.106:7000", image = [], store_id = "1",  demo_images_dir = "../test5_images"):
        self.store_id = store_id
        self.image = image
        self.demo_images_dir = demo_images_dir
        self.message = {}
        self.severs_handler_endpoint = server + "/constant_messages"
        self.fridge = FridgeContentCounter()
        self.camera = None
        self.encrypter = Encrypter()
    
    def set_store_id(self, store_id):
        self.store_id = store_id

    def read_image(self, image_name):
        self.image = cv.imread(image_name)
        return self.image
    
    def capture_image(self):
        self.camera = cv.VideoCapture(1)
        #self.camera = cv.VideoCapture(0)     # Uncomment this for testing on a personal computer
        res, self.image = self.camera.read()
        self.camera.release()
        return self.image
    
    def build_message(self, verbose = False):
        content_count = self.fridge.get_content_count(self.image, verbose=verbose)        
        timestamp = int(time.time())
        
        message = {}
        #self.message = "TEMP_MESSAGE\n" + self.store_id + "\n" + str(content_count) + "\n" + str(timestamp)
        message["store_id"] = self.store_id
        message["content_count"] = content_count
        message["timestamp"] = str(timestamp)       
        
        if verbose:
            print("message built")
            print(message)

        self.message = self.encrypter.encrypt(message, verbose)
        
        if verbose:
            print("Encrypted message")
            print(self.message)
        
    def upload_message(self, verbose = False) -> bool:
        try:        
            res = requests.post(self.severs_handler_endpoint, json=self.message)
            if res.ok:
                if verbose:
                    print("data sended to server succesfully")                            
                return True
        except requests.exceptions.RequestException:
            print("Unable to connect with server, plase check the wifi connection")
        return False                        
    
    def update_software(self, verbose = False):
        self.fridge.update_software(verbose = verbose)
        if verbose:
            print("Uploader software updated")

    def upload_test_mesage(self):
        '''method that will be used to send dummy data to test server connection
        PLEASE DO NOT USE THIS IN PRODUCTION'''
        content_count = {'fresca lata 355 ml':4, 'sidral mundet lata 355 ml': 1}
        dt = datetime.now()
        timestamp = datetime.timestamp(dt)
        
        #self.message = "TEMP_MESSAGE\n" + self.store_id + "\n" + str(content_count) + "\n" + str(timestamp)
        self.message["store_id"] = self.store_id
        self.message["content_count"] = content_count
        self.message["timestamp"] = str(timestamp)
        self.upload_message(verbose=True, time_range=(0,0.5))        
    
    def run_demo(self, verbose = False):
        for image_name in os.listdir(self.demo_images_dir):
            if image_name.endswith(".jpg"):
                image_dir = os.path.join(self.demo_images_dir, image_name)
                image = cv.imread(image_dir)
                height, width = image.shape[:2]
                reduced_dims = ( width//2 , height//2 )
                image = cv.resize(image, reduced_dims)
                self.image = image
                self.build_message()
                self.upload_message(time_range = (0, 1), verbose = verbose)
                if verbose:
                    cv.imshow("Imagen Actual", self.image)
                    cv.waitKey(0)
        
        cv.destroyAllWindows()
        
    def run_camera_demo(self):
        while True:
            self.capture_image()
            self.build_message(verbose = True)
            self.upload_message(time_range = (0,0.1), verbose = True)
            cv.imshow("Foto", self.image)
            pressed_key = cv.waitKey(0)
            if pressed_key == 0x15:
                # if user pressed q
                break
            #Event().wait(30)
            
if __name__ == "__main__":
    uploader = MessageUploader()
    uploader.run_camera_demo()
    #uploader.upload_test_mesage()