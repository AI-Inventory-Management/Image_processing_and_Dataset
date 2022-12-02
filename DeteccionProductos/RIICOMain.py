"""
Main code of system.

Classes:
    RIICOMain

Author:
    Alejandro Dominguez
"""
#_________________________________Libraries____________________________________
import time
time.sleep(2)

from InitializationMessageUploader import InitializationMessageUploader as IniMU
from MessageUploader import MessageUploader as ContMU

import cv2 as cv
import sys, getopt
import json
from threading import Event
import random
import os
from hardware_backend_data import SERVER_IP_ADDRESS

#__________________________________Classes_____________________________________
class RIICOMain():
    """
    Main class of system.
    
    ...
    
    Attributes
    ----------
    hardware_backend_server : string
        URL of handler server.
        
    initial_uploader : InitializationMessageUploader
        Initialization message handler.
        
    constant_messages_uploader : MessageUploader
        Message handler.
    
    running_on_nuc : bool
        True if on NUC
    
    post_cycle_time : int
        Time between uploads.
    
    test_images_for_normal_fridge_flow : list
        Test images.
        
    """
    
    def __init__(self, post_cycle_time = 1800):
        """
        Construct of attributes of class.

        Parameters
        ----------
        post_cycle_time : int, optional
            Time in seconds between messages. The default is 1800.

        Returns
        -------
        None.

        """
        self.hardware_backend_server = SERVER_IP_ADDRESS
        self.initial_uploader = IniMU(self.hardware_backend_server)
        self.constant_messages_uploader = ContMU(self.hardware_backend_server)
        self.running_on_nuc = False
        self.post_cycle_time = post_cycle_time
        testing_without_fridge_dir = os.path.join(os.path.dirname(__file__), "..", "test_for_normal_fridge_flow")        
        names_of_test_images_for_normal_fridge_flow = [i for i in os.listdir(testing_without_fridge_dir) if i.endswith(".jpg")]
        self.test_images_for_normal_fridge_flow = list( map(os.path.join , [testing_without_fridge_dir]*len(names_of_test_images_for_normal_fridge_flow) , names_of_test_images_for_normal_fridge_flow) )        
        self.test_images_for_normal_fridge_flow.sort()

    def activate_intel_nuc_features(self):
        """
        Activate features.

        Returns
        -------
        None.

        """
        self.initial_uploader.running_on_intel_nuc = True
        self.constant_messages_uploader.running_on_intel_nuc = True
        self.constant_messages_uploader.activate_inference_with_open_vino()                
        
    def send_initial(self, verbose = False):
        """
        Send initialization message.

        Parameters
        ----------
        verbose : bool, optional
            For further information. The default is False.

        Returns
        -------
        None.

        """
        has_data = self.initial_uploader.obtain_initial_store_data_gui()
        while not has_data:
            self.initial_uploader.upload_message(verbose = verbose)
            self.initial_uploader.build_data_file(verbose = verbose)
            has_data = self.initial_uploader.obtain_initial_store_data_gui()
        if verbose:
            print("First message sent.")
        
        
    def update_store_info(self, verbose = False):
        """
        Update store information.
        
            This function should be called after initial_uploader registers the store in hardware backend,
            once that happens, the store_id is passed to constant_uploader as well as the rest of the modules
            to make them work succesfully 

        Parameters
        ----------
        verbose : bool, optional
            For further information. The default is False.

        Returns
        -------
        None.

        """
        with open( os.path.join(os.path.dirname(__file__), "./data/store_data.json") ) as f:
            data = json.load(f)
            f.close()
            store_id = data["store_id"]            
            self.constant_messages_uploader.set_store_id(store_id = store_id)            
            if verbose:
                print("Store id updated")
                
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
        self.initial_uploader.update_software(verbose = verbose)
        self.constant_messages_uploader.update_software(verbose = verbose)
        if verbose:
            print("Software updated succesfully")
    
    def run(self, verbose = False, testing_with_fridge = True, update_cycles = 1440, running_on_nuc = False):
        """
        Run system.

        Parameters
        ----------
        verbose : bool, optional
            For further information. The default is False.
            
        testing_with_fridge : bool, optional
            True if using fridge. The default is True.
            
        update_cycles : int, optional
            Time in seconds between actualizations. The default is 1440.
            
        running_on_nuc : bool, optional
            True if running on nuc. The default is False.

        Returns
        -------
        None.

        """
        if running_on_nuc:
            self.activate_intel_nuc_features()
        self.send_initial(verbose = verbose)
        self.update_store_info(verbose = verbose)
        
        epoch = 1
        no_fridge_counter = 0
        message_wait = random.randint(0,self.post_cycle_time)
        
        while True:
            if verbose:
                print("Message wait: " + str(self.post_cycle_time/60))
                print("Image capture wait: " + str((message_wait)/60) + " min")
                print("Current epoch: " + str(epoch))
                print("Cycles till update: " +str(update_cycles - epoch))
            
            if epoch % 4 == 0:
                print("Running correctly")
            
            Event().wait(message_wait)
            if testing_with_fridge:
                img = self.constant_messages_uploader.capture_image()
            else:
                img_name = self.test_images_for_normal_fridge_flow[no_fridge_counter]                
                img = self.constant_messages_uploader.read_image(img_name)
            
            if verbose:
                img_copy = cv.resize(img, (720,576))
                cv.imshow("Captura", img_copy)
                cv.waitKey(0)
                cv.destroyAllWindows()
            
            self.constant_messages_uploader.build_message(verbose = verbose)
            self.constant_messages_uploader.upload_message(verbose = verbose)
            
            epoch += 1
            if epoch == update_cycles:
                self.update_software(verbose = verbose)
                epoch = 0
                
            no_fridge_counter = (no_fridge_counter+1)%len(self.test_images_for_normal_fridge_flow)
            
    def run_demo(self):
        """
        Demonstrate functionality of class.

        Returns
        -------
        None.

        """
        self.run(verbose = True, testing_with_fridge=False, update_cycles = 5, running_on_nuc = False)
        
if __name__ == '__main__':
    main = RIICOMain(post_cycle_time = 3)
    command_input_arguments = sys.argv[1:]
    testing_with_fridge = False
    running_on_nuc = False
    verbose = True
    try:
        opts, args = getopt.getopt(command_input_arguments, "", ["testing_with_fridge=", "running_on_nuc=", "verbose="])
        for opt, arg in opts:        
            if opt == "--testing_with_fridge":
                testing_with_fridge = arg == "true" or arg == "True"
            elif opt == "--running_on_nuc":
                running_on_nuc = arg == "true" or arg == "True"
            elif opt == "--verbose":
                verbose = arg == "true" or arg == "True"            
    except getopt.GetoptError:
        pass
    main.run(verbose = verbose, testing_with_fridge=testing_with_fridge, update_cycles = 5, running_on_nuc = running_on_nuc)            
