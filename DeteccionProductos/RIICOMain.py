from InitializationMessageUploader import InitializationMessageUploader as IniMU
from MessageUploader import MessageUploader as ContMU

import json
from threading import Event

class RIICOMain ():
    def __init__(self):
        self.initial_uploader = IniMU()
        self.uploader = ContMU()
        self.wait_time = 1800
        
    def send_initial(self, verbose = False):
        self.initial_uploader.obtain_store_data()
        self.initial_uploader.build_message()
        self.initial_uploader.upload_message(verbose = verbose)
        self.initial_uploader.build_data_file(verbose = verbose)
        
        if verbose:
            print("First message sent.")
    
    def update_store_info (self, verbose = False):
        with open("store_data.json") as f:
            data = json.load(f)
            store_id = data["store_id"]
            
            self.uploader.set_store_id(store_id = store_id)
            
            if verbose:
                print("Store id updated")
    
    def run(self, verbose = False, time_range = (0, 30)):
        self.send_initial(verbose = verbose)
        self.update_store_info(verbose = verbose)
        
        message_wait = 0
        
        while True:
            Event().wait(self.wait_time - message_wait)
            self.uploader.capture_image()
            self.uploader.build_message()
            message_wait = self.uploader.upload_message(time_range = time_range, verbose = verbose)
            
    def run_demo(self):
        self.run(verbose = True, time_range = (0, 0.1))
        
if __name__ == '__main__':
    main = RIICOMain()
    main.run()            