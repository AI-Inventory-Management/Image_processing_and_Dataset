#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 17:09:34 2022

@author: lex
"""

import json
import numpy as np

labels = ['fresca lata 355 ml', 
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
          'coca cola lata 355 ml', 
          'producto no oficial']

ean = ["7501055365470", 
       "7501055363162", 
       "7501055303786", 
       "7501055317875", 
       "7501055329267", 
       "7501055365609", 
       "3223905201", 
       "7501055339983", 
       "75007614", 
       "7501055370986", 
       "7501055361540", 
       "-1"]

prev_pred = np.zeros((8, len(labels) - 1))
prev_pred[:, 6] = 1

f = open("./data/product_data.json", 'w')
data = {"labels" : labels, "eans" : ean, "prev_pred" : prev_pred.tolist()}
json.dump(data, f)
f.close()

model_path = './models/sodas_detector_prot'
alfa = 0.4
beta = 0.6 
thresh = 0.65

f = open("./data/model_data.json", 'w')
data = {"model_path" : model_path, "alfa" : alfa, "beta" : beta, "thresh" : thresh}
json.dump(data, f)
f.close()