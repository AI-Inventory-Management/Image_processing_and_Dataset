"""
Fridge content detection usign open vino.

Classes:
    FridgeNotFoundException
    
    OVFridgeContentDetector
    
Author:
    Jose Angel del Angel
    
"""
#_________________________________Libraries____________________________________
import numpy as np
import cv2 as cv
import json
import os

#____________________________________Main______________________________________
class FridgeNotFoundException(Exception):
    """
    Fridge not found exception.
    
        Exception that must be thrown when fridge canot be found using 
        our image processing pipeline.
        
    """
    
    def __init__(self):
        """
        Construct class attributes.
        
        Parameters
        ----------
        None.

        Returns
        -------
        None.

        """
        super().__init__("Fridge couldn't be found, please ensure that camera is pointing to a fridge")

class OVFridgeContentDetector():
    """
    Class to detect contents of a fridge.
    
    Attributes
    ----------
    demo_images_dir : string
        Path to demo images.
        
    segmentation_model : Model
        Model for segmentation.
        
    segmentation_model_path : str
        Path to segmentation model.
        
    ie : Core
        Core.
        
    model : Model
        Model.
        
    model_output_layer : Model Output
        Model output layer.
        
    Methods
    -------
    filter_contours_by_area(contours, hierarchy, area)
        Filter contours beneath given area.
        
    sort_contours_by_area(contours, hierarchy):
        Sort contours by area.
        
    filter_contours_by_only_inner(contours, hierarchy):
        Filter external contours.
        
    find_fridge_content_box(image):
        Get internal area of fridge.
        
    sort_rectangle_cords(rectangle_cords):
        Sort coordinates of rectangle.
        
    rotate_fridge_content(image, content_cords, content_rectangle_data):
        Rotate image of contents of the image to get a perfect rectangle.
        
    get_fridge_content_image(raw_image):
        Obtain image of contents of fridge.
    
    get_fridge_cells(raw_image, fridge_rows_count, fridge_columns_count, output_shape=(80, 200)):
        Obtain cells of content within the fridge.
        
    run_demo():
        Demonstrate functionality of class
    
    """
    
    def __init__(self) -> None:
        """
        Construct attributes of the class.

        Parameters
        ----------
        None.
        
        Returns
        -------
        None.

        """
        self.demo_images_dir = "./test4_images/"
        self.segmentation_model = None
        with open( os.path.join(os.path.dirname(__file__),"./data/model_data.json"), 'r') as f:
            data = json.load(f)
            f.close
            self.segmentation_model_path = os.path.join(os.path.dirname(__file__),data["ov_segmentation_model_path"])        
        from openvino.runtime import Core
        self.ie = Core()     
        model_temp = self.ie.read_model(model = self.segmentation_model_path)
        self.model = self.ie.compile_model(model = model_temp, device_name = "CPU")
        self.model_output_layer = self.model.output(0)
        
    def filter_coutours_by_area(self, contours, hierarchy, area):
        """
        Filter contours by area.
        
            Filter contours with less area than the specified.

        Parameters
        ----------
        contours : list
            List of contours to filter.
            
        hierarchy : list
            List of hierarchies.
            
        area : int
            Minimum area required to be considered.

        Returns
        -------
        new_contours : list
            Filtered contours.
            
        new_hierarchy : list
            Hierarchies of filtered contours.

        """
        # Prepare
        new_contours = []
        new_hierarchy = []
        
        # Filter
        for i in range(len(contours)):
            cnt = contours[i]
            cnt_hierarchy = hierarchy[0][i,:].tolist()
            if cv.contourArea(cnt) > area:
                new_contours.append(cnt)
                new_hierarchy.append(cnt_hierarchy)
                
        # Format
        new_contours = tuple(new_contours)
        new_hierarchy = np.array([new_hierarchy])
        return (new_contours, new_hierarchy)

    def sort_contours_by_area(self, contours, hierarchy):
        """
        Sort contours by area.
        
            Sort contours from biggest area to smallest.

        Parameters
        ----------
        contours : list
            List of contours.
            
        hierarchy : list
            List of hierarchies of contours.

        Returns
        -------
        sorted_contours : list
            Sorted contours.
            
        sorted_hierarchies : list
            Hierarchies of sorted contours.

        """
        # Prepare
        areas = list(map(cv.contourArea, contours))
        array_to_sort = list(zip(areas, contours, hierarchy[0].tolist()))
        
        # Sort
        array_to_sort.sort(key=lambda x:x[0], reverse=True)
        
        # Format
        _, sorted_contours, sorted_hierarchies = zip(*array_to_sort)
        sorted_hierarchies = list(sorted_hierarchies)
        sorted_hierarchies = np.array([sorted_hierarchies])
        return (sorted_contours, sorted_hierarchies)

    def filter_contours_to_only_inner(self, contours, hierarchy):
        """
        Filter contours to inner.
        
            Filter external contours.

        Parameters
        ----------
        contours : list
            List of contours.
            
        hierarchy : list
            List of hierarchies of contours.

        Returns
        -------
        new_contours : list
            Filtered contours.
            
        new_hierarchy : list
            Hierarchies of filtered contours.

        """
        # Prepare
        does_not_have_next_child = hierarchy[0,:,2] == -1
        does_not_have_next_child = does_not_have_next_child.tolist()
        new_contours = []
        new_hierarchy = []
        
        # Filter
        for i in range(len(does_not_have_next_child)):
            cnt = contours[i]
            cnt_hierarchy = hierarchy[0,i,:].tolist()
            if does_not_have_next_child[i]:
                new_contours.append(cnt)
                new_hierarchy.append(cnt_hierarchy)
                
        # Format
        new_contours = tuple(new_contours)
        new_hierarchy = np.array([new_hierarchy])
        return (new_contours, new_hierarchy)

    def find_fridge_content_box(self, image):
        """
        Find fridge content.
        
            Find content box of fridge.

        Parameters
        ----------
        image : cv2 Image
            Image of fridge.

        Raises
        ------
        FridgeNotFoundException
            If fridge not found in image.

        Returns
        -------
        box : array
            Coordinates of box.
            
        min_area_rect : cv2 Contours
            Contour of box.

        """
        # Prepare
        height, width = image.shape[:2]           
        expanded_image = cv.resize(image, (640, 360), interpolation = cv.INTER_AREA)        
        expanded_image = np.expand_dims(expanded_image, axis = 0)
        pred = self.model([expanded_image])[self.model_output_layer][0]
        mask = np.argmax(pred, axis=-1)
        mask *= 255        
        mask = np.uint8(mask)
        fridge_treshold = cv.resize(mask, (width, height), interpolation = cv.INTER_AREA)        
        
        # Filter contours
        contours, hierarchy = cv.findContours(fridge_treshold, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
        contours, hierarchy = self.filter_coutours_by_area(contours, hierarchy, width*height*(1/10))
        if(len(contours)==0):
            raise FridgeNotFoundException()
        contours, hierarchy = self.filter_contours_to_only_inner(contours, hierarchy)
        if(len(contours)==0):
            raise FridgeNotFoundException()
        contours, hierarchy = self.sort_contours_by_area(contours, hierarchy)

        fridge_treshold_copy = fridge_treshold.copy()

        fridge_treshold_copy = cv.merge((fridge_treshold_copy, fridge_treshold_copy, fridge_treshold_copy))
        cv.drawContours(fridge_treshold_copy, contours, -1, (0,255,0), 3)                    
        min_area_rect = cv.minAreaRect(contours[0])
        
        # Get box
        box = cv.boxPoints(min_area_rect)
        box = np.int0(box)            
        return box, min_area_rect

    def sort_rectangle_cords(self, rectangle_cords):
        """
        Sort rectangle coordinates.
        
            This function ensures that:
            - index 0 of rectangle_cords corresponds to upper left corner,
            - index 1 of rectangle_cords corresponds to upper right corner,
            - index 2 of rectangle_cords corresponds to lower left corner,
            - index 3 of rectangle_cords corresponds to lower right corner
            
        Parameters
        ----------
        rectangle_cords : array
            Coordinates of box.

        Returns
        -------
        rectangle_cords_sorted : list
            Sorted coordinates.

        """
        # Prepare
        rectangle_cords_vector_angles = np.rad2deg(np.arctan(rectangle_cords[:,1]/rectangle_cords[:,0]))
        rectangle_cords_vector_lenghts = np.linalg.norm(rectangle_cords, axis=1)
        cords = rectangle_cords.tolist()
        cords_vector_data = list(zip(rectangle_cords_vector_angles, rectangle_cords_vector_lenghts, cords))
        
        # Sort
        cords_vector_data.sort(key=lambda x:x[0])
        
        # Format
        upper_cords = cords_vector_data[:2]
        lower_cords = cords_vector_data[2:4]
        upper_cords.sort(key=lambda x:x[1])
        lower_cords.sort(key=lambda x:x[1])
        cords_vector_data_sorted = upper_cords + lower_cords
        _, _, rectangle_cords_sorted = zip(*cords_vector_data_sorted)
        rectangle_cords_sorted = np.array(rectangle_cords_sorted)
        return rectangle_cords_sorted

    def rotate_fridge_content(self, image, content_cords, content_rectangle_data):
        """
        Rotate image of fridge content.

        Parameters
        ----------
        image : cv2 Image
            Image of fridge.
            
        content_cords : list
            List of coordinates.
            
        content_rectangle_data : cv2 Contours
            Contour of content box.

        Returns
        -------
        rotated_image : cv2 Image
            Rotated image.
            
        rotated_cords : list
            Rotated coordinates.

        """
        # Calculate rotation
        m = (content_cords[1,1]-content_cords[0,1])/(content_cords[1,0]-content_cords[0,0])
        angle = np.rad2deg(np.arctan(m))
        fridge_content_center = content_rectangle_data[0]
        ones = np.ones((content_cords.shape[0], 1))
        rotated_image = image.copy()
        rotated_cords = content_cords.copy()

        rot_mat = cv.getRotationMatrix2D(fridge_content_center, angle, 1.0)
        
        # Rotate
        rotated_image = cv.warpAffine(rotated_image, rot_mat, image.shape[1::-1], flags=cv.INTER_LINEAR)
        rotated_cords = np.concatenate((rotated_cords, ones), axis=1)
        rotated_cords = (rot_mat@(rotated_cords.T)).T
        rotated_cords = np.uint(rotated_cords)
        return (rotated_image, rotated_cords)

    def get_fridge_content_image(self, raw_image):
        """
        Obtain the image of the contents of the fridge.

        Parameters
        ----------
        raw_image : cv2 Image
            Image of fridge.

        Returns
        -------
        content_image : cv2 Image
            Image of the contents of the fridge.

        """
        content_rectangle_cords, content_rectangle = self.find_fridge_content_box(raw_image)
        content_rectangle_cords = self.sort_rectangle_cords(content_rectangle_cords)
        rotated_image, final_content_cords = self.rotate_fridge_content(raw_image, content_rectangle_cords, content_rectangle)
        content_image = rotated_image[final_content_cords[0,1]:final_content_cords[3,1], final_content_cords[0,0]:final_content_cords[3,0]]
        return content_image
    
    def get_fridge_cells(self, raw_image, fridge_rows_count, fridge_columns_count, output_shape=(80, 200)):
        """
        Obtain the cells of the contents of the fridge.

        Parameters
        ----------
        raw_image : cv2 Image
            Image of fridge.
            
        fridge_rows_count : int
            Number of rows of the fridge.
            
        fridge_columns_count : int
            Number of columns of the fridge.
            
        output_shape : tuple, optional
            Shape of the resulting image. The default is (80, 200).

        Returns
        -------
        content_cells : list
            Images of each cell of the fridge.

        """
        fridge_content = self.get_fridge_content_image(raw_image)
        fridge_content_h, fridge_content_w = fridge_content.shape[:2] 
        vertical_boundaries = np.linspace(0,fridge_content_h, fridge_rows_count+1, dtype=np.int0)
        horizontal_boundaries = np.linspace(0,fridge_content_w, fridge_columns_count+1, dtype=np.int0)
        content_cells = []
        
        for i in range(1,len(vertical_boundaries)):
            for j in range(1,len(horizontal_boundaries)):
                cell = fridge_content[vertical_boundaries[i-1]:vertical_boundaries[i], horizontal_boundaries[j-1]:horizontal_boundaries[j]]
                cell = cv.resize(cell, output_shape)
                content_cells.append(cell)

        return content_cells

    def run_demo(self):
        """
        Demonstrate functionality of class.
        
        Parameters
        ----------
        None.
        
        Returns
        -------
        None.

        """
        for image_name in os.listdir(self.demo_images_dir):
            if image_name.endswith(".jpg"):
                image_dir = os.path.join(self.demo_images_dir, image_name)
                image = cv.imread(image_dir)
                height, width = image.shape[:2]
                reduced_dims = ( width//2 , height//2 )
                image = cv.resize(image, reduced_dims)
                try:
                    content_cells = self.get_fridge_cells(image, 2, 4)
                    for i in range(len(content_cells)):
                        cv.imshow("cell " + str(i), content_cells[i])
                    cv.waitKey(0)
                except FridgeNotFoundException:
                    print("fridge not found on image: {im_name}".format(im_name=image_name))
