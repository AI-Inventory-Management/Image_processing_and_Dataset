"""
Detect fridge corners coordinates.

    Identify fridge and cut the image keeping only the fridge. Then take 
    samples of the 8 soda cells of the fridge.
    
Classes:
    cornerDetector
    
Author
    Jose Angel del Angel
    
"""
#_________________________________Libraries____________________________________
import numpy as np
import cv2 as cv
import os
import pandas as pd
import time

#__________________________________Classes_____________________________________
class cornerDetector():
    """
    Class to identify fridge by its corners.
    
    ...
    
    Attributes
    ----------
    sweeping_kernel_proportion : float
        Proportion of kernel to image size
        
    haar_like_kernels_size : tuple
        Dimensions of kernel
    
    haar_like_kernels : tuple
        Kernels to be used
        
    dataset_path : string
        Path to dataset folder
    
    dataset_images_path : string
        Path to dataset images folder
    
    raw_images_path : string
        Path to test images
    
    image_id_prefix : string
        Prefix of image name
        
    Methods
    -------
    generate_haar_like_kernels():
        Create the required kernels for analysis
        
    compute_feature(haar_kernel, roi):
        Generate feature computation of the given ROI
        
    get_roi_features(roi):
        Generate the features of the given ROI
    
    sweep_image(image):
        Sweep the image and show ROIs
    
    sweep_image_to_build_dataset(image, images_names_data, features_data, output_data)
        Create dataset form the given image
    
    generate_dataset():
        Create dataset
    
    """
    
    def __init__(self):
        """
        Construct attributes for the class.
        
        Parameters
        ----------
        None.
        
        Returns
        -------
        None.

        """
        self.sweeping_kernel_proportion = 1/10
        self.haar_like_kernels_size = (10, 10)
        self.haar_like_kernels = self.generate_haar_like_kernels()
        self.dataset_path = "./corner_detector_dataset/"
        self.dataset_images_path = os.path.join(self.dataset_path, "images/")
        self.raw_images_path = "./test1_images/"
        self.image_id_prefix = "javier"

    def generate_haar_like_kernels(self):
        """
        Create the kernels required.

        Parameters
        ----------
        None.
        
        Returns
        -------
        kernels : tuple
            Tuple containing kernels generated.

        """
        # Build cast-kernel
        kernel_size_half = int(self.haar_like_kernels_size[0]/2)
        ones_kernel = np.ones(self.haar_like_kernels_size)

        # Build kernels
        kernel_a = np.ones(self.haar_like_kernels_size)*-1
        kernel_a[kernel_size_half:, kernel_size_half:] = ones_kernel[kernel_size_half:, kernel_size_half:]

        kernel_b = np.ones(self.haar_like_kernels_size)*-1
        kernel_b[kernel_size_half:, :kernel_size_half] = ones_kernel[kernel_size_half:, :kernel_size_half]

        kernel_c = np.ones(self.haar_like_kernels_size)*-1
        kernel_c[:kernel_size_half, kernel_size_half:] = ones_kernel[:kernel_size_half, kernel_size_half:]

        kernel_d = np.ones(self.haar_like_kernels_size)*-1
        kernel_d[:kernel_size_half, :kernel_size_half] = ones_kernel[:kernel_size_half, :kernel_size_half]

        kernel_e = np.ones(self.haar_like_kernels_size)*-1
        kernel_e[kernel_size_half:, :] = ones_kernel[kernel_size_half:, :]

        kernel_f =  np.ones(self.haar_like_kernels_size)*-1
        kernel_f[:kernel_size_half, :] = ones_kernel[:kernel_size_half, :]

        kernel_g =  np.ones(self.haar_like_kernels_size)*-1
        kernel_g[:, kernel_size_half:] = ones_kernel[:, kernel_size_half:]

        kernel_h =  np.ones(self.haar_like_kernels_size)*-1
        kernel_h[:, :kernel_size_half] = ones_kernel[:, :kernel_size_half]
        
        #Format
        kernels = (kernel_a, kernel_b, kernel_c, kernel_d, kernel_e, kernel_f, kernel_g, kernel_h)
        return kernels

    def compute_feature(self, haar_kernel, roi):
        """
        Copute a single feature of the given roi with the given kernel.

        Parameters
        ----------
        haar_kernel : np Array
            Kernel to build feature around.
            
        roi : cv2 Image
            Region of Interest tu build feature from.

        Returns
        -------
        np Array
            Feature of ROI.

        """
        # Compute feature
        resized_roi = cv.resize(roi, self.haar_like_kernels_size, interpolation = cv.INTER_AREA)
        conv = haar_kernel*resized_roi
        return conv.sum()

    def get_roi_features(self, roi):
        """
        Create feature list of the given roi.

        Parameters
        ----------
        roi : cv2 Image
            Region of interest to get features from.

        Returns
        -------
        list
            List of features of the given roi.

        """
        return [self.compute_feature(k, roi) for k in self.haar_like_kernels]

    def sweep_image(self, image):
        """
        Go over the image generating ROIs.

        Parameters
        ----------
        image : cv2 Image
            Image to be swept.

        Returns
        -------
        None.

        """
        # Build kernel
        kernel_size = int(min(image.shape[0], image.shape[1])*self.sweeping_kernel_proportion)
        kernel_size_half = int(kernel_size/2)
        
        # Sweep image
        image_copy = image.copy()
        for i in range(kernel_size_half, image.shape[0] - kernel_size_half, kernel_size_half):
            for j in range(kernel_size_half, image.shape[1] - kernel_size_half, kernel_size_half):
                image_copy = image.copy()
                image_copy = cv.rectangle(image_copy, (j-kernel_size_half, i-kernel_size_half), (j+kernel_size_half, i+kernel_size_half), (255, 0, 0), 2)
                
                # Create roi
                roi = image_copy[i-kernel_size_half:i+kernel_size_half,j-kernel_size_half:j+kernel_size_half]
                roi = cv.cvtColor(roi,cv.COLOR_BGR2GRAY)
                
                # Show result
                cv.imshow('sweeping',image_copy)
                cv.imshow('roi',roi)
                cv.waitKey(0)

    def sweep_image_to_build_dataset(self, image, images_names_data, features_data, output_data):
        """
        Sweep image and build dataset from ROIs.

        Parameters
        ----------
        image : cv2 Image
            Image to be swept.
            
        images_names_data : list
            List of names of images.
            
        features_data : list
            List of features of the images.
            
        output_data : list
            List of labels of data.

        Returns
        -------
        None.

        """
        # Build kernel
        kernel_size = int(min(image.shape[0], image.shape[1])*self.sweeping_kernel_proportion)
        kernel_size_half = int(kernel_size/2)
        
        # Sweep image
        for i in range(kernel_size_half, image.shape[0] - kernel_size_half, kernel_size_half):
            for j in range(kernel_size_half, image.shape[1] - kernel_size_half, kernel_size_half):
                # Show ROI
                image_name = self.image_id_prefix + "_" + str(time.time()).replace('.', '_') + ".jpg"
                roi_dir = os.path.join(self.dataset_images_path, image_name)
                image_copy = image.copy()
                image_copy = cv.rectangle(image_copy, (j-kernel_size_half, i-kernel_size_half), (j+kernel_size_half, i+kernel_size_half), (255, 0, 0), 2)
                roi = image_copy[i-kernel_size_half:i+kernel_size_half,j-kernel_size_half:j+kernel_size_half]
                roi = cv.cvtColor(roi,cv.COLOR_BGR2GRAY)
                cv.imshow('sweeping',image_copy)
                cv.imshow('roi',roi)
                pressed_key = cv.waitKey(0)
                
                # Label
                if pressed_key == 0x74 or pressed_key == 0x66:
                    if pressed_key == 0x74:
                        print("pressed key is t")
                        output_val = 1.0
                    elif pressed_key == 0x66:
                        print("pressed key is f")
                        output_val = 0.0
                    cv.imwrite(roi_dir, roi)
                    images_names_data.append(roi_dir)
                    features_data.append(self.get_roi_features(roi))
                    output_data.append(output_val)
                else:
                    print("ERROR data couldn´t be saved, key value is: {key}".format(key= pressed_key))

    def generate_dataset(self):
        """
        Create dataset from images.
        
        Parameters
        ----------
        None.
        
        Returns
        -------
        None.

        """
        # Prepare
        dataset_images_names = []
        dataset_features_data = []
        dataset_output_values_data = []
        
        # Build image dataset
        images_dirs = []
        for image in os.listdir(self.raw_images_path):
            if image.endswith(".jpg"):
                image_path = os.path.join(self.raw_images_path, image)
                images_dirs.append(image_path)
        number_of_images = len(images_dirs)
        one_third_images = int(number_of_images/3)
        if self.image_id_prefix == 'josea':
            images_to_check = images_dirs[1*one_third_images-one_third_images:1*one_third_images]
        elif self.image_id_prefix == 'javier':
            images_to_check = images_dirs[2*one_third_images-one_third_images:2*one_third_images]
        elif self.image_id_prefix == 'alex':
            images_to_check = images_dirs[3*one_third_images-one_third_images:]
        
        # Build dataset
        for image_path in images_to_check:
            image = cv.imread(image_path)
            new_image_size = (int(image.shape[1]*0.3), int(image.shape[0]*0.3) )
            image = cv.resize(image, new_image_size, interpolation = cv.INTER_AREA)
            self.sweep_image_to_build_dataset(image, dataset_images_names, dataset_features_data, dataset_output_values_data)
        features = np.array(dataset_features_data)
        data = {
            "image_dir": dataset_images_names,
            "conv_a": features[:, 0].tolist(),
            "conv_b": features[:, 1].tolist(),
            "conv_c": features[:, 2].tolist(),
            "conv_d": features[:, 3].tolist(),
            "conv_e": features[:, 4].tolist(),
            "conv_f": features[:, 5].tolist(),
            "conv_g": features[:, 6].tolist(),
            "conv_h": features[:, 7].tolist(),
            "is_fridge_corner": dataset_output_values_data,
            }
        df = pd.DataFrame(data)
        try:
            existing_df = pd.read_csv('data.csv')
            df = pd.concat([existing_df, df])
            df.to_csv(os.path.join(self.dataset_path,'data.csv'), index=False)
        except:
            df.to_csv(os.path.join(self.dataset_path,'data.csv'), index=False)

#____________________________________Main______________________________________
if __name__ == "__main__":
    # Generate dataset
    corner_detector = cornerDetector()
    corner_detector.generate_dataset()
