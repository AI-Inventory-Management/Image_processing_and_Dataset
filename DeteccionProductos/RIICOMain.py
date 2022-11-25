import time
time.sleep(2)

from InitializationMessageUploader import InitializationMessageUploader as IniMU
from MessageUploader import MessageUploader as ContMU

import cv2 as cv
import json
from threading import Event
import random
import os


class RIICOMain():
    def __init__(self, post_cycle_time = 1800):
        self.hardware_backend_server = "http://192.168.28.106:7000"
        self.initial_uploader = IniMU(self.hardware_backend_server)
        self.constant_messages_uploader = ContMU(self.hardware_backend_server)
        self.running_on_nuc = False
        self.post_cycle_time = post_cycle_time
        testing_without_fridge_dir = os.path.join(os.path.dirname(__file__), "..", "test_for_normal_fridge_flow")        
        names_of_test_images_for_normal_fridge_flow = [i for i in os.listdir(testing_without_fridge_dir) if i.endswith(".jpg")]
        self.test_images_for_normal_fridge_flow = list( map(os.path.join , [testing_without_fridge_dir]*len(names_of_test_images_for_normal_fridge_flow) , names_of_test_images_for_normal_fridge_flow) )        
        self.test_images_for_normal_fridge_flow.sort()

    def activate_intel_nuc_features(self):
        self.initial_uploader.running_on_intel_nuc = True
        self.constant_messages_uploader.running_on_intel_nuc = True
        self.constant_messages_uploader.activate_inference_with_open_vino()                
        
    def send_initial(self, verbose = False):
        had_data = self.initial_uploader.obtain_initial_store_data_gui()
        if not had_data:
            self.initial_uploader.upload_message(verbose = verbose)
            self.initial_uploader.build_data_file(verbose = verbose)
            if verbose:
                print("First message sent.")
        else:
            if verbose:
                print("First message already sent.")
        
    def update_store_info(self, verbose = False):
        """
            This function should be called after initial_uploader registers the store in hardware backend,
            once that happens, the store_id is passed to constant_uploader as well as the rest of the modules
            to make them work succesfully 
        """
        with open( os.path.join(os.path.dirname(__file__), "./data/store_data.json") ) as f:
            data = json.load(f)
            f.close()
            store_id = data["store_id"]            
            self.constant_messages_uploader.set_store_id(store_id = store_id)            
            if verbose:
                print("Store id updated")
                
    def update_software(self, verbose = False):
        self.initial_uploader.update_software(verbose = verbose)
        self.constant_messages_uploader.update_software(verbose = verbose)
        if verbose:
            print("Software updated succesfully")
    
    def run(self, verbose = False, testing_with_fridge = True, update_cycles = 1440, running_on_nuc = False):
        """
        inputs: 
            time_range -> tuple with the time range (in minutes) to send a constant message with the store stock
            update_cycles -> time ??
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
        self.run(verbose = False, testing_with_fridge=False, update_cycles = 5, running_on_nuc = True)
        
if __name__ == '__main__':
    main = RIICOMain(post_cycle_time = 3)
    main.run_demo()            
