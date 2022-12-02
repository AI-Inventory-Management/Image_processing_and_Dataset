# Image Proccessing and Dataset
## Installation
To install this proyect download filles form github. Then be sure to have the following files installed in python:
-- Open Vino
-- Tensorflow
-- Numpy
-- OpenCV
-- Serial
-- Time
Then, contact the provider to recieve the Encryption.py file and add it in the ./Image_processing_and_Dataset/DeteccionProductos folder.

The required arduino code is found in ./Image_processing_and_Dataset/Arduino/UltrasonicRead

## Operation
Make sure to connect webcam and arduino module to your system. Then run ./Image_processing_and_Dataset/DeteccionProductos/RIICOMain.py.
System should start to run smoothly. However, installation should include the proper configuration to run the script on boot.

## File Manifest
Files on folder ./Image_processing_and_Dataset/Arduino and subfolders are to be intalled on arduino module. Files on folder ./Image_processing_and_Dataset/DeteccionProductos are fundamental to the proper functionality of the system.
Within this last folder, file DeleteStoreData.py can be run to delete all store data. 
Any other file was used for testing.

## Trubleshooting
To trubleshoot run RIICOMain.py file with verbose = True. Then, most errors will be shown in terminal, if any. Here is a list of errors:
-- Fridge not found -> Check camera connection.
-- Serial disconected -> Check arduino connection.
-- Unnable to connect to server -> Check internet connection.

## Known bugs
Here is a list of known bugs
-- System is stuck at 'message build' -> The number of sensors expected at arduino or system is incorrect -> Check code.
-- Arduino information does not match reallity -> Sensors are not working properly -> Check arduino module and sensors.

## Authors
Main contributors:
-- Jose Angel del Angel
-- Alejandro Dominguez Lugo

