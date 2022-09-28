import cv2
import numpy as np
import os

vid = cv2.VideoCapture(0)

fotoNum = 0

while(True):
    ret, image = vid.read()
    
    if type(image) != type(None):
        h, l, d = image.shape
        thresh = 10
        
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
         
        # Use canny edge detection
        edges = cv2.Canny(gray,50,150,apertureSize=3)
         
        # Apply HoughLinesP method to
        # to directly obtain line end points
        lines_list =[]
        lines = cv2.HoughLinesP(
                    edges, # Input edge image
                    1, # Distance resolution in pixels
                    np.pi/180, # Angle resolution in radians
                    threshold=100, # Min number of votes for valid line
                    minLineLength=450, # Min allowed length of line
                    maxLineGap=100 # Max allowed gap between line for joining them
                    )
        
        highest = [(0, h), (0, h)]
        
        if type(lines) != type(None):
            for points in lines:
                # Extracted points nested in the list
                x1,y1,x2,y2=points[0]
                
                if (abs(y2 - y1) < thresh) and (abs(x2 - x1) < int(l * 0.25)):
                    cv2.line(image,(x1,y1),(x2,y2),(0,255,255),2)
                    # Maintain a simples lookup list for points
                    lines_list.append([(x1,y1),(x2,y2)])
                    if (y1 < highest[0][1]):
                        highest = [(x1,y1),(x2,y2)]
        
        cv2.line(image, highest[0], highest[1], (255, 0, 0), 3)
        
        side = abs(highest[0][0] - highest[1][0])
        
        alt = [highest[0][1], highest[1][1]]
        
        p1 = (0, 0)
        p2 = (0, 0)
        
        if(alt[0] - side < 0) or (alt[1] - side < 0):
            p1 = (highest[0][0], highest[0][1])
            p2 = (highest[1][0], highest[0][1] + 2 * side)
    
        elif(alt[0] - 2 * side < 0) or (alt[1] - 2 * side < 0):
            p1 = (highest[0][0], highest[0][1] - side)
            p2 = (highest[1][0], highest[0][1] + side)
            
        else:
            p1 = (highest[0][0], highest[0][1] - 2 * side)
            p2 = (highest[1][0], highest[0][1] - 2 * side)
        
        cv2.line(image, (p1[0], p1[1]),(p1[0], p2[1]), (255, 0, 0), 3)
        
        cv2.line(image, (p1[0], p1[1]),(p2[0], p1[1]), (255, 0, 0), 3)
        
        cv2.line(image, (p1[0], p2[1]),(p2[0], p2[1]), (255, 0, 0), 3)
        
        cv2.line(image, (p2[0], p1[1]),(p2[0], p2[1]), (255, 0, 0), 3)
        
        divH = int(abs(p1[0] - p2[0]) / 4)
        divV = int(abs(p1[1] - p2[1]) / 2)
        
        cv2.line(image, (p1[0], p1[1] + divV), (p2[0], p1[1] + divV), (255, 0, 0), 2)
        
        cv2.line(image, (p1[0] + 1 * divH, p1[1]), (p1[0] + 1 * divH, p2[1]), (255, 0, 0), 2)
        cv2.line(image, (p1[0] + 2 * divH, p1[1]), (p1[0] + 2 * divH, p2[1]), (255, 0, 0), 2)
        cv2.line(image, (p1[0] + 3 * divH, p1[1]), (p1[0] + 3 * divH, p2[1]), (255, 0, 0), 2)
        
        image = cv2.resize(image, (int(l/2), int(h/2)))
        
        cv2.imshow('image', image[:l, :h])
        
        key = cv2.waitKey(1)
        
        if key & 0xFF == ord('q'):
            break
        
        if key & 0xFF == ord('f'):
            roi1 = image[p1[0] : p1[0] + 1 * divH, p1[1] : p1[1] + divV]
            roi2 = image[p1[0] + 1 * divH : p1[0] + 2 * divH, p1[1] : p1[1] + divV]
            roi3 = image[p1[0] + 2 * divH : p1[0] + 3 * divH, p1[1] : p1[1] + divV]
            roi4 = image[p1[0] + 3 * divH : p1[0] + 4 * divH, p1[1] : p1[1] + divV]
            
            roi5 = image[p1[0] : p1[0] + 1 * divH, p1[1] + divV : p1[1] + 2 * divV]
            roi6 = image[p1[0] + 1 * divH : p1[0] + 2 * divH, p1[1] + divV : p1[1] + 2 * divV]
            roi7 = image[p1[0] + 2 * divH : p1[0] + 3 * divH, p1[1] + divV : p1[1] + 2 * divV]
            roi8 = image[p1[0] + 3 * divH : p1[0] + 4 * divH, p1[1] + divV : p1[1] + 2 * divV]
            
            rois = [roi1, roi2, roi3, roi4, roi5, roi6, roi7, roi8]
            
            for i in range(8):
                name = "./Data/roi" + str(i) + "_foto" + str(fotoNum) + ".jpg"
                cv2.imwrite(name, rois[i])
                
            fotoNum += 1
                
            
                
    
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()