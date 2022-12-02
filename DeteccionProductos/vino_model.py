"""
Open vino test.

Author:
    Jose Angel del Angel
"""
# Import libraries
import cv2
import matplotlib.pyplot as plt
import numpy as np
from openvino.runtime import Core

ie = Core()
model = ie.read_model(model="saved_model.xml")
compiled_model = ie.compile_model(model=model, device_name="CPU")

output_layer = compiled_model.output(0)

image = cv2.imread(filename="fresca.jpg")
input_image = cv2.resize(src=image, dsize=(150, 420))
input_image = np.expand_dims(input_image, 0)

result_infer = compiled_model([input_image])[output_layer]
result_index = np.argmax(result_infer)

labels = ["fresca lata 355 ml", "sidral mundet lata 355 ml", "fresca botella de plastico 600 ml", "fuze tea durazno 600 ml", "power ade mora azul botella de plastico 500 ml", "delaware punch lata 355 ml", "vacio", "del valle durazno botella de vidrio 413 ml", "sidral mundet botella de plastico 600 ml", "coca cola botella de plastico 600 ml", "power ade mora azul lata 453 ml", "coca cola lata 355 ml", "producto no oficial"]
print(labels[result_index])