"""
Message uploader.

    Builds, encrypts and uploads constant messages. Also stores relevant 
    information.

Classes:
    MessageUploader

Author:
    Alejandro Dominguez
    
"""
#_________________________________Libraries____________________________________
from datetime import datetime
import time
import cv2 as cv
import os
import requests

from ProductCounter import FridgeContentCounter
from OVProductCounter import OVFridgeContentCounter
from Encrypter import Encrypter

#__________________________________Classes_____________________________________
class MessageUploader ():
    """
    Message uploader.
        
        Builds, encrypts and uploads constant messages. Also stores relevant 
        information.
    
    ...
    
    Attributes
    ----------
    store_id : string
        Id of store.
    
    image : cv2 Image
        Image of fridge.
        
    demo_images_dir : string
        Path to demo images.
        
    message : dict
        Message.
        
    servers_handler_endpoint : string
        URL of server.
    
    fridge : FridgeContentCounter
        Fridge content counter.
    
    camera : cv2 Video Capture
        Connection to camera.
    
    encrypter : Encrypter
        Encrypter.
        
    running_on_intel_nuc : bool
        True if running on NUC
        
    Methods
    -------
    activate_inference_with_open_vino():
        Initiate inference of products with open vino.
        
    set_store_id(store_id):
        Set store id.
    
    read_image(image_name):
        Read image with the given name.
    
    capture_image():
        Capture image of camera with open vino.
        
    build_message(verbose = False):
        Build message.
    
    upload_message(verbose = False):
        Upload message to cloud handler.
        
    update_software(verbose = False):
        Update models and values of system.
    
    upload_test_message(verbose = False):
        Upload test message to cloud handler.
    
    run_demo(verbose = False):
        Demonstrate class functionality.
    
    run_camera_demo():
        Demonstrate class functionality using camera.
        
    """
    
    def __init__(self, server = "http://192.168.195.106:7000", image = [], store_id = "1",  demo_images_dir = "../test5_images"):
        """
        Construct class attributes.

        Parameters
        ----------
        server : string, optional
            URL to server. The default is "http://192.168.195.106:7000".
        image : cv2 Image, optional
            Image of fridge. The default is [].
        store_id : string, optional
            Id of store. The default is "1".
        demo_images_dir : string, optional
            Path to test images. The default is "../test5_images".

        Returns
        -------
        None.

        """
        self.store_id = store_id
        self.image = image
        self.demo_images_dir = demo_images_dir
        self.message = {}
        self.severs_handler_endpoint = server + "/constant_messages"
        self.fridge = FridgeContentCounter()
        self.camera = None
        self.encrypter = Encrypter()
        self.running_on_intel_nuc = False

    def activate_inference_with_open_vino(self):
        """
        Activate inferecne with open vino.

        Returns
        -------
        None.

        """
        self.fridge = OVFridgeContentCounter()        
    
    def set_store_id(self, store_id):
        """
        Set store id to value.

        Parameters
        ----------
        store_id : string
            Store id.

        Returns
        -------
        None.

        """
        self.store_id = store_id

    def read_image(self, image_name):
        """
        Read image with given name.

        Parameters
        ----------
        image_name : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        self.image = cv.imread(image_name)
        return self.image
    
    def capture_image(self):
        """
        Capture image from camera.

        Returns
        -------
        cv2 Image
            Image of fridge.

        """
        self.camera = cv.VideoCapture(0)
        res, self.image = self.camera.read()
        self.camera.release()
        return self.image
    
    def build_message(self, verbose = False):
        """
        Build message.

        Parameters
        ----------
        verbose : bool, optional
            For further information. The default is False.

        Returns
        -------
        None.

        """
        # Count products
        content_count = self.fridge.get_content_count(self.image, verbose=verbose)   
        
        # Get timestamp
        timestamp = int(time.time())
        
        # Build message
        message = {}
        message["store_id"] = self.store_id
        message["content_count"] = content_count
        message["timestamp"] = str(timestamp)       
        
        if verbose:
            print("message built")
            print(message)
        
        # Encrypt message
        self.message = self.encrypter.encrypt(message, verbose)
        
        if verbose:
            print("Encrypted message")
            print(self.message)
        
    def upload_message(self, verbose = False) -> bool:
        """
        Upload message to cloud handler.

        Parameters
        ----------
        verbose : bool, optional
            For further information. The default is False.

        Returns
        -------
        bool
            True if information sent succesfully. False otherwise.

        """
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
        """
        Update models and information of the system.

        Parameters
        ----------
        verbose : bool, optional
            For further information. The default is False.

        Returns
        -------
        None.

        """
        self.fridge.update_software(verbose = verbose)
        if verbose:
            print("Uploader software updated")

    def upload_test_mesage(self):
        """
        Upload test message to cloud handler.

        Returns
        -------
        None.

        """
        # Generate dummy data
        content_count = {'fresca lata 355 ml':4, 'sidral mundet lata 355 ml': 1}
        dt = datetime.now()
        timestamp = datetime.timestamp(dt)
        
        # Build message
        self.message["store_id"] = self.store_id
        self.message["content_count"] = content_count
        self.message["timestamp"] = str(timestamp)
        
        # Upload
        self.upload_message(verbose=True, time_range=(0,0.5))        
    
    def run_demo(self, verbose = False):
        """
        Demonstrate functionality of class.

        Parameters
        ----------
        verbose : bool, optional
            For further information. The default is False.

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
                self.image = image
                self.build_message()
                self.upload_message(time_range = (0, 1), verbose = verbose)
                if verbose:
                    cv.imshow("Imagen Actual", self.image)
                    cv.waitKey(0)
        
        cv.destroyAllWindows()
        
    def run_camera_demo(self):
        """
        Demonstrate functionality of class with camera.

        Returns
        -------
        None.

        """
        while True:
            self.capture_image()
            self.build_message(verbose = True)
            self.upload_message(time_range = (0,0.1), verbose = True)
            cv.imshow("Foto", self.image)
            pressed_key = cv.waitKey(0)
            if pressed_key == 0x15:
                break

#____________________________________Main______________________________________
if __name__ == "__main__":
    uploader = MessageUploader()
    uploader.run_camera_demo()
    #uploader.upload_test_mesage()
