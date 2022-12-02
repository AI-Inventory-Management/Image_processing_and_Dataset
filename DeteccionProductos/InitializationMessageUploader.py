"""
Initialization Message Uploader.

    Builds, saves, encrypts and sends initialization message.

Classes:
    InitializationMessageUploader

Author:
    Jose Angel del Angel
    
"""
#_________________________________Libraries____________________________________
import requests
import json
import time
from Encrypter import Encrypter
import InitializationForm 
import copy
import os

#__________________________________Classes_____________________________________
class InitializationMessageUploader():
    """
    Class to build and upload initialization message.
    
        Builds, saves, encrypts and sends initialization message.
    
    ...
    
    Attributes
    ----------
    message : dict
        Message.
        
    unencrypted_message : dict
        Message before encryption
        
    fridge_info : dict
        Information of the fridge
        
    servers_handler_endpoint : string
        Handler URL
    
    ean : list
        Eans of the products.
    
    soda_labels : list
        Labels of the products.
        
    ean2label : dict
        Dictionary ean to label.
    
    label2ean : dict
        Dictionary label to ean.
    
    encrypter : Encrypter
        Encrypter.
    
    running_on_intel_nuc : bool
        True if running on nuc.
        
    store_id : int
        Id of store.
        
    Methods
    -------
    build_message():
        Build message
    
    obtain_initial_store_data_gui():
        Build and render initialization form.
        
    obtain_inital_store_data():
        Recieve initial store data from terminal.
        
    build_data_file(verbose = False):
        Create file and save obtained information.
        
    upload_message(verbose = False):
        Upload message to handler.
    
    update_software(verbose = False):
        Update information and modules of system.
        
    build_return_test_message(verbose = False):
        Build test message.
    
    upload_test_mesage():
        Upload test message to handler.
        
    run_comms_demo():
        Demonstrate class functionality.

    """
    
    def __init__(self, server = "http://192.168.195.106:7000"):
        """
        Construct class attributes.

        Parameters
        ----------
        server : string, optional
            URL to server. The default is "http://192.168.195.106:7000".

        Returns
        -------
        None.

        """        
        self.message = {}
        self.unencrypted_message = {}
        self.fridge_info = {}
        self.severs_handler_endpoint = server + "/initaialization_messages"
        self.ean = []
        self.soda_labels = []
        self.ean2label = {}
        self.label2ean = {}
        self.encrypter = Encrypter()
        self.running_on_intel_nuc = False
        
        # Load product information
        with open( os.path.join( os.path.dirname(__file__), "./data/product_data.json") , 'r') as f:
            data = json.load(f)
            f.close()
            self.ean = data["eans"]
            self.ean.remove("-1")
            self.soda_labels = data["labels"]            
            self.soda_labels.remove("vacio")
            self.soda_labels.remove("producto no oficial")
            self.ean2label = data["ean2label"]
            self.label2ean = data["label2ean"]
            
        self.store_id = ""
    
    def build_message(self, 
        store_name:str, 
        store_latitude:float,
        store_longitude:float,
        store_state:str,
        store_municipality:str,
        store_zip_code:int,
        store_address:str,
        store_curr_stock:dict, 
        store_min_stocks:dict,
        store_max_stocks:dict,
        fridge_cols:int,
        fridge_rows:int, 
        verbose = False):      
        """
        Build message to be uploaded.

        Parameters
        ----------
        store_name : str
            Name of the store.
            
        store_latitude : float
            Latitude of the store.
            
        store_longitude : float
            Longitude of the store.
            
        store_state : str
            State where the store is located.
            
        store_municipality : str
            Municipality of the store.
            
        store_zip_code : int
            Zip code of the store.
            
        store_address : str
            Address of the store.
            
        store_curr_stock : dict
            Current stock of every product.
            
        store_min_stocks : dict
            Minimum stocj of every product.
            
        store_max_stocks : dict
            Maximum stock of every product.
            
        fridge_cols : int
            Number of columns of the fridge.
            
        fridge_rows : int
            Number of rows in the fridge.
            
        verbose : bool, optional
            For further information. The default is False.

        Returns
        -------
        None.

        """
        # Build message
        self.message["store_name"] = store_name
        self.message["store_latitude"] = store_latitude
        self.message["store_longitude"] = store_longitude
        self.message["store_state"] = store_state
        self.message["store_municipality"] = store_municipality
        self.message["store_zip_code"] = store_zip_code
        self.message["store_address"] = store_address
        self.message["store_curr_stock"] = store_curr_stock
        self.message["store_min_stocks"] = store_min_stocks
        self.message["store_max_stocks"] = store_max_stocks
        self.fridge_info["fridge_cols"] = fridge_cols
        self.fridge_info["fridge_rows"] = fridge_rows
        
        timestamp = int(time.time())
        self.message["timestamp"] = str(timestamp)
        
        if verbose:
            print("Message")
            print(self.message)
        
        # Encrypt message
        if len(self.unencrypted_message) == 0:
            self.unencrypted_message = copy.deepcopy(self.message)
        self.message = self.encrypter.encrypt(self.message)
        
        if verbose:
            print("Message encrypted")
            print(self.message)

    def obtain_initial_store_data_gui(self) ->bool:
        """
        Obtain initial store information with user interface.

        Returns
        -------
        bool
            If failure false.

        """
        # Prepare
        data = None
        ean2label_for_form = copy.deepcopy(self.ean2label)
        ean2label_for_form.pop("0")
        ean2label_for_form.pop("-1")
        
        # Get data
        try:
            with open( os.path.join( os.path.dirname(__file__), "./data/store_data.json") , 'r') as f:
                data = json.load(f)
                f.close()
            if len(data) == 0:
                server = InitializationForm.load_form(ean2label_for_form, self.running_on_intel_nuc)
                while not InitializationForm.form_complete:
                    pass
                self.build_message(
                    InitializationForm.form_data["store_name"],
                    InitializationForm.form_data["store_latitude"],
                    InitializationForm.form_data["store_longitude"],
                    InitializationForm.form_data["store_state"],
                    InitializationForm.form_data["store_municipality"],
                    InitializationForm.form_data["store_zip_code"],
                    InitializationForm.form_data["store_address"],
                    InitializationForm.form_data["current_stock"],
                    InitializationForm.form_data["min_stocks"],
                    InitializationForm.form_data["max_stocks"],
                    InitializationForm.form_data["fridge_cols"],
                    InitializationForm.form_data["fridge_rows"]
                )
                time.sleep(2)
                server.shutdown()
                return False
            else:
                if data["store_id"] == "":                    
                    fridge_rows = 2
                    fridge_cols = 4
                    with open( os.path.join( os.path.dirname(__file__), "./data/fridge_data.json") , 'r') as f:
                        fridge_data = json.load(f)
                        f.close()
                        fridge_rows, fridge_cols = tuple(map(int,fridge_data["fridge_dimensions"]))

                    store_info = dict(data["store_info"])
                    self.build_message(store_info["store_name"], 
                        store_info["store_latitude"],
                        store_info["store_longitude"],
                        store_info["store_state"],
                        store_info["store_municipality"],
                        store_info["store_zip_code"],
                        store_info["store_address"],
                        store_info["store_curr_stock"],
                        store_info["store_min_stocks"],
                        store_info["store_max_stocks"],
                        fridge_cols,
                        fridge_rows)
                    return False
                return True
        except FileNotFoundError:
            server = InitializationForm.load_form(ean2label_for_form, self.running_on_intel_nuc)
            while not InitializationForm.form_complete:
                pass            
            self.build_message(
                InitializationForm.form_data["store_name"],
                InitializationForm.form_data["store_latitude"],
                InitializationForm.form_data["store_longitude"],
                InitializationForm.form_data["store_state"],
                InitializationForm.form_data["store_municipality"],
                InitializationForm.form_data["store_zip_code"],
                InitializationForm.form_data["store_address"],
                InitializationForm.form_data["current_stock"],
                InitializationForm.form_data["min_stocks"],
                InitializationForm.form_data["max_stocks"],
                InitializationForm.form_data["fridge_cols"],
                InitializationForm.form_data["fridge_rows"]
                )
            time.sleep(2)
            server.shutdown()            
            return False
        
    def obtain_initial_store_data(self) -> bool:
        """
        Get initial store information through terminal.

            This function will prompt a technician for the initial data 
            of the store were the system is located.
            
        Returns
        -------
        bool
            If failure false.

        """
        # Get data
        try:
            with open( os.path.join( os.path.dirname(__file__), "./data/store_data.json") , 'r') as f:
                data = json.load(f)
                f.close()
                if len(data) == 0:
                    store_name = input("please write the NAME of the new store: ")
                    store_latitude = float(input("please write the LATITUDE of the new store: "))
                    store_longitude = float(input("please write the LONGITUDE of the new store: "))
                    store_state = input("please write the STATE of the new store: ")
                    store_municipality = input("please write the MUNICIPALITY of the new store: ")
                    store_zip_code = int(input("please write the ZIP_CODE of the new store: "))
                    store_address = input("please write the ADDRESS of the new store: ")
            
                    print("Thanks, please input the store stock details")
                    current_stock = {}        
                    min_stocks = {}
                    max_stocks = {}
                    
                    i = 0
                    for soda in self.soda_labels:
                        print('')
                        print('-------------- {s} --------------'.format(s=soda))
                        current_stock[self.ean[i]] = int(input("how many {s} are on the store right now: ".format(s=soda)))
                        min_stocks[self.ean[i]] = int(input("whats the min of {s} to generate an alert: ".format(s=soda)))
                        max_stocks[self.ean[i]] = int(input("whats amount of {s} to fill the store: ".format(s=soda)))
                        
                        i += 1
                    
                    print("Finally, input fridge info")
                    fridge_cols = int(input("please write the NUMBER of columns in the fridge: "))
                    fridge_rows = int(input("please write the NUMBER of rows in the fridge: "))                    
                        
                else:
                    return True
                    
            self.build_message(store_name, store_latitude, store_longitude, store_state, store_municipality, store_zip_code, store_address, current_stock, min_stocks, max_stocks, fridge_cols, fridge_rows)
            return False
        except FileNotFoundError:
            store_name = input("please write the NAME of the new store: ")
            store_latitude = float(input("please write the LATITUDE of the new store: "))
            store_longitude = float(input("please write the LONGITUDE of the new store: "))
            store_state = input("please write the STATE of the new store: ")
            store_municipality = input("please write the MUNICIPALITY of the new store: ")
            store_zip_code = int(input("please write the ZIP_CODE of the new store: "))
            store_address = input("please write the ADDRESS of the new store: ")
    
            print("Thanks, please input the store stock details")
            current_stock = {}        
            min_stocks = {}
            max_stocks = {}
            
            i = 0
            for soda in self.soda_labels:
                print('')
                print('-------------- {s} --------------'.format(s=soda))
                current_stock[self.ean[i]] = int(input("how many {s} are on the store right now: ".format(s=soda)))
                min_stocks[self.ean[i]] = int(input("whats the min of {s} to generate an alert: ".format(s=soda)))
                max_stocks[self.ean[i]] = int(input("whats amount of {s} to fill the store: ".format(s=soda)))
                i += 1
            
            print("Finally, input fridge info")
            fridge_cols = int(input("please write the NUMBER of columns in the fridge: "))
            fridge_rows = int(input("please write the NUMBER of rows in the fridge: "))
            
            
            self.build_message(store_name, store_latitude, store_longitude, store_state, store_municipality, store_zip_code, store_address, current_stock, min_stocks, max_stocks, fridge_cols, fridge_rows)
            return False
        
    def build_data_file(self, verbose = False):
        """
        Generate and fill data file.

        Parameters
        ----------
        verbose : bool, optional
            For further information. The default is False.

        Returns
        -------
        None.

        """
        # Save store information
        data = {"store_id" : self.store_id, "store_info" : self.unencrypted_message}
        with open( os.path.join(os.path.dirname(__file__),"./data/store_data.json"), 'w') as f:
            json.dump(data, f)
            f.close()
            if verbose:
                print("Store information saved succesfully.")
        
        # Save fridge information
        fridge_data = {"fridge_dimensions" : (self.fridge_info["fridge_cols"], self.fridge_info["fridge_rows"])}
        with open( os.path.join(os.path.dirname(__file__), "./data/fridge_data.json"), 'w') as ff:
            json.dump(fridge_data, ff)
            ff.close()
            if verbose:
                print("Fridge information saved succesfully")
    
    def upload_message(self, verbose = False):
        """
        Upload message to cloud handler.

        Parameters
        ----------
        verbose : bool, optional
            For further information. The default is False.

        Returns
        -------
        None.

        """
        try:
            res = requests.post(self.severs_handler_endpoint, json=self.message)
            if res.ok and verbose:
                print("Message sent:")
                print(self.unencrypted_message)
                print()
                print("data sended to server succesfully")
            
            self.store_id = str(res.json()["store_id"])
            if verbose:
                print (self.store_id)
        except requests.exceptions.RequestException:
            print("unnable to connect with server please check wifi connection")
            
    def update_software(self, verbose = False):
        """
        Update software.

        Parameters
        ----------
        verbose : bool, optional
            For further information. The default is False.

        Returns
        -------
        None.

        """
        # Update product information
        with open( os.path.join( os.path.dirname(__file__), "./data/product_data.json"), 'r') as f:
            data = json.load(f)
            f.close()
            self.ean = data["eans"]
            self.ean.remove("-1")
            self.soda_labels = data["labels"]
            self.soda_labels
            self.soda_labels.remove("vacio")
            self.soda_labels.remove("producto no oficial")
            
        if verbose:
            print("Initializer software updated")
            
    def build_return_test_message(self, verbose = False):
        """
        Build test message.

        Parameters
        ----------
        verbose : bool, optional
            For further information. The default is False.

        Returns
        -------
        None.

        """
        self.build_message(store_name = "as", 
                           store_latitude = 1, 
                           store_longitude= 1, 
                           store_state = 1, 
                           store_municipality = 1, 
                           store_zip_code = 1, 
                           store_address = 1, 
                           store_curr_stock = {"7501055365470" : 1}, 
                           store_min_stocks = {"7501055365470" : 1}, 
                           store_max_stocks = {"7501055365470" : 1},
                           fridge_cols = 4,
                           fridge_rows = 2, 
                           verbose = verbose)
    
    def upload_test_mesage(self):
        """
        Upload test message to cloud handler.

        Returns
        -------
        None.

        """
        # Build message
        store_name = input("please write the NAME of the new store: ")
        store_latitude = float(input("please write the LATITUDE of the new sotre: "))
        store_longitude = float(input("please write the LONGITUDE of the new sotre: "))
        store_state = input("please write the STATE of the new sotre: ")
        store_municipality = input("please write the MUNICIPALITY of the new sotre: ")
        store_zip_code = int(input("please write the ZIP_CODE of the new sotre: "))
        store_address = input("please write the ADDRESS of the new sotre: ")

        print("Thanks, please input the store stock details")
        current_stock = {}        
        min_stocks = {}
        max_stocks = {}
        
        i = 0
        for soda in self.soda_labels:
            print('')
            print('-------------- {s} --------------'.format(s=soda))
            current_stock[self.ean[i]] = int(input("how many {s} are on the store right now: ".format(s=soda)))
            min_stocks[self.ean[i]] = int(input("whats the min of {s} to generate an alert: ".format(s=soda)))
            max_stocks[self.ean[i]] = int(input("whats amount of {s} to fill the store: ".format(s=soda)))
            i += 1
            
        print("Finally, input fridge info")
        fridge_cols = int(input("please write the NUMBER of columns in the fridge: "))
        fridge_rows = int(input("please write the NUMBER of rows in the fridge: "))
        
        self.build_message(store_name, store_latitude, store_longitude, store_state, store_municipality, store_zip_code, store_address, current_stock, min_stocks, max_stocks, fridge_cols, fridge_rows)
        
        # Upload message
        self.upload_message(verbose=True)
        
    def run_comms_demo (self):
        """
        Demonstrate de functionality of the class.

        Returns
        -------
        None.

        """
        self.build_return_test_message(verbose = True)
        self.upload_message(verbose = True)
        self.build_data_file(verbose = True)
        
#____________________________________Main______________________________________
if __name__ == "__main__":
    uploader = InitializationMessageUploader()
    uploader.run_comms_demo()
