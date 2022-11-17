import cv2 as cv
import numpy as np
import os

original_base_dir = "./sodas_dataset_raw/session {n}"
segmented_base_dir = "./sodas_dataset_raw_segmented/session_{n}"

def get_num_images_in_files_list(files_list: list[str]):
    res = 0
    for file in files_list:
        if file.endswith(".jpg"):
            res += 1
    return res

for i in range(1,83):
    original_session_dir = original_base_dir.format(n=i)
    segmented_session_dir = segmented_base_dir.format(n=i)
    segmented_image_names = os.listdir(segmented_session_dir)
    num_candidate_images = get_num_images_in_files_list(segmented_image_names)
    
    paired_segmented_images = []
    
    for file_name in os.listdir(original_session_dir):                
        if file_name.endswith(".jpg"):  
            candidate_index = 0            
            while True:        
                candidate_mask = cv.imread( os.path.join(segmented_session_dir, segmented_image_names[candidate_index]) )

                original_image = cv.imread( os.path.join( original_session_dir, file_name) )
                original_image = original_image*1.0
                
                candidate_mask = cv.cvtColor(candidate_mask, cv.COLOR_BGR2GRAY)
                candidate_mask = candidate_mask/255.0            
                
                original_times_masked = cv.merge([ original_image[:,:,0]*candidate_mask, original_image[:,:,1]*candidate_mask, original_image[:,:,2]*candidate_mask ])
                original_times_masked = np.uint8(original_times_masked)
                original_times_masked = cv.resize(original_times_masked, (int(original_times_masked.shape[1]/2), int(original_times_masked.shape[0]/2)))            
                
                cv.imshow("test", original_times_masked)
                pressed_key = cv.waitKey(0)
                print(pressed_key)
                if pressed_key == 0x79:
                    # pressed key is 'y'
                    cv.imwrite( os.path.join("./raw_and_segmented_") )                                        
                    segmented_image_names.pop(candidate_index)
                    num_candidate_images -= 1                    
                    break
                else:                    
                    candidate_index = (candidate_index + 1)%num_candidate_images                                
    break