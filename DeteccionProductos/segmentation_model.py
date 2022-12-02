"""
Test segmentation model.

Author:
    Jose Angel del Angel
"""
import cv2
import numpy as np
from openvino.runtime import Core

ie = Core()
model = ie.read_model(model="saved_model.xml")
compiled_model = ie.compile_model(model=model, device_name="CPU")

output_layer = compiled_model.output(0)
print(output_layer)
image = cv2.imread(filename="ref.jpg")
input_image = cv2.resize(src=image, dsize=(400, 400))
input_image = np.expand_dims(input_image, axis=0)
result_infer = compiled_model([input_image])[output_layer][0]
print(result_infer)
mask = np.argmax(result_infer, axis=-1)
mask *= 255
print(np.max(mask))
#mask = mask.astype(int)
#mask.dtype = 'uint8'
mask = np.uint8(mask)
print(mask.shape)
mask = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
mask = cv2.resize(mask,(image.shape[1],image.shape[0]))
cv2.imshow("seg",mask)
cv2.waitKey()
cv2.destroyAllWindows()