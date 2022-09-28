"""
File to detect the fridge corners cordinates so that we can
cut the image keeping only the fridge and take samples of the
8 soda cells of the fridge
"""
import numpy as np
import cv2 as cv
import os
import pandas as pd
import time

class cornerDetector():
    def __init__(self):
        self.sweeping_kernel_proportion = 1/10
        self.haar_like_kernels_size = (10, 10)
        self.haar_like_kernels = self.generate_haar_like_kernels()
        self.dataset_path = "./corner_detector_dataset/"
        self.dataset_images_path = os.path.join(self.dataset_path, "images/")
        self.raw_images_path = "./test1_images/"
        self.image_id_prefix = "josea"

    def generate_haar_like_kernels(self):
        kernel_size_half = int(self.haar_like_kernels_size[0]/2)
        ones_kernel = np.ones(self.haar_like_kernels_size)

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

        kernels = (kernel_a, kernel_b, kernel_c, kernel_d, kernel_e, kernel_f, kernel_g, kernel_h)

        return kernels

    def compute_feature(self, haar_kernel, roi):
        resized_roi = cv.resize(roi, self.haar_like_kernels_size, interpolation = cv.INTER_AREA)
        conv = haar_kernel*resized_roi
        return conv.sum()

    def get_roi_features(self, roi):
        return [self.compute_feature(k, roi) for k in self.haar_like_kernels]

    def sweep_image(self, image):
        kernel_size = int(min(image.shape[0], image.shape[1])*self.sweeping_kernel_proportion)
        kernel_size_half = int(kernel_size/2)
        image_copy = image.copy()
        for i in range(kernel_size_half, image.shape[0] - kernel_size_half, kernel_size_half):
            for j in range(kernel_size_half, image.shape[1] - kernel_size_half, kernel_size_half):
                image_copy = image.copy()
                image_copy = cv.rectangle(image_copy, (j-kernel_size_half, i-kernel_size_half), (j+kernel_size_half, i+kernel_size_half), (255, 0, 0), 2)
                roi = image_copy[i-kernel_size_half:i+kernel_size_half,j-kernel_size_half:j+kernel_size_half]
                roi = cv.cvtColor(roi,cv.COLOR_BGR2GRAY)
                cv.imshow('sweeping',image_copy)
                cv.imshow('roi',roi)
                cv.waitKey(0)

    def sweep_image_to_build_dataset(self, image, images_names_data, features_data, output_data):
        kernel_size = int(min(image.shape[0], image.shape[1])*self.sweeping_kernel_proportion)
        kernel_size_half = int(kernel_size/2)
        for i in range(kernel_size_half, image.shape[0] - kernel_size_half, kernel_size_half):
            for j in range(kernel_size_half, image.shape[1] - kernel_size_half, kernel_size_half):
                image_name = self.image_id_prefix + "_" + str(time.time()).replace('.', '_') + ".jpg"
                roi_dir = os.path.join(self.dataset_images_path, image_name)
                image_copy = image.copy()
                image_copy = cv.rectangle(image_copy, (j-kernel_size_half, i-kernel_size_half), (j+kernel_size_half, i+kernel_size_half), (255, 0, 0), 2)
                roi = image_copy[i-kernel_size_half:i+kernel_size_half,j-kernel_size_half:j+kernel_size_half]
                roi = cv.cvtColor(roi,cv.COLOR_BGR2GRAY)
                cv.imshow('sweeping',image_copy)
                cv.imshow('roi',roi)
                pressed_key = cv.waitKey(0)
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
        dataset_images_names = []
        dataset_features_data = []
        dataset_output_values_data = []

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


if __name__ == "__main__":
    corner_detector = cornerDetector()
    corner_detector.generate_dataset()
