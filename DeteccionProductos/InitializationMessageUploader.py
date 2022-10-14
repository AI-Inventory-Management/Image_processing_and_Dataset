from datetime import datetime
import cv2 as cv
import numpy as np
import os
from threading import Event
import requests

class InitializationMessageUploader():
    def __init__(self):        
        self.message = {}
        self.severs_handler_endpoint = "http://192.168.0.25:7000/initaialization_messages"
        '''
        self.soda_labels = [
            'fresca lata 355 ml',
            'sidral mundet lata 355 ml',
            'fresca botella de plastico 600 ml',
            'fuze tea durazno 600 ml',
            'power ade mora azul botella de plastico 500 ml',
            'delaware punch lata 355 ml',
            'vacio',
            'del valle durazno botella de vidrio 413 ml',
            'sidral mundet botella de plastico 600 ml',
            'coca cola botella de plastico 600 ml',
            'power ade mora azul lata 453 ml',
            'coca cola lata 355 ml']
        '''    
        self.soda_labels = ['fresca lata 355 ml', 'sidral mundet lata 355 ml']
    
    def build_message(self, 
        store_name:str, 
        store_latitude:float,
        store_longitude:float,
        store_state:str,
        store_municipality:str,
        store_zip_code:int,
        store_adress:str,
        store_curr_stock:dict, 
        store_min_stocks:dict,
        store_max_stocks:dict,
        store_status:int):                
        self.message["store_name"] = store_name
        self.message["store_latitude"] = store_latitude
        self.message["store_longitude"] = store_longitude
        self.message["store_state"] = store_state
        self.message["store_municipality"] = store_municipality
        self.message["store_zip_code"] = store_zip_code
        self.message["store_adress"] = store_adress
        self.message["store_curr_stock"] = store_curr_stock
        self.message["store_min_stocks"] = store_min_stocks
        self.message["store_max_stocks"] = store_max_stocks
        self.message["store_status"] = store_status

    def upload_message(self, verbose = False):
        res = requests.post(self.severs_handler_endpoint, json=self.message)
        if res.ok and verbose:
            print("data sended to server succesfully")
            print(res.json())
            # TODO: add response handling, server response should be store id assigned
            print(self.message)

    def upload_test_mesage(self):
        '''method that will be used to send dummy data to test server connection
        PLEASE DO NOT USE THIS IN PRODUCTION'''
        store_name = input("please write the NAME of the new sotre: ")
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
        for soda in self.soda_labels:
            print('')
            print('-------------- {s} --------------'.format(s=soda))
            current_stock[soda] = int(input("how many {s} are on the store right now: ".format(s=soda)))
            min_stocks[soda] = int(input("whats the min of {s} to generate an alert: ".format(s=soda)))
            max_stocks[soda] = int(input("whats amount of {s} to fill the store: ".format(s=soda)))
        
        store_status = 1

        self.build_message(store_name, store_latitude, store_longitude, store_state, store_municipality, store_zip_code, store_address, current_stock, min_stocks, max_stocks, store_status)
        self.upload_message(verbose=True) 
            
if __name__ == "__main__":
    uploader = InitializationMessageUploader()
    uploader.upload_test_mesage()