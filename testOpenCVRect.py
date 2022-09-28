import cv2
import numpy as np
import os


image_path = ""
raw_images_path = "./test1_images"

for image in os.listdir(raw_images_path):
    if image.endswith(".jpg"):
        image_path = os.path.join(raw_images_path, image)
    
    # Best
    image = cv2.imread(image_path)
    
    # Worst
    #image = cv2.imread('./test1_images/WIN_20220927_17_13_23_Pro.jpg')
    
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
     
    # Iterate over points
    #lowest = [(0,0), (0,0)]
    
    highest = [(0, h), (0, h)]
    
    if type(lines) != type(None):
        for points in lines:
            # Extracted points nested in the list
            x1,y1,x2,y2=points[0]
            # Draw the lines joing the points
            # On the original image
            
            
            if (abs(y2 - y1) < thresh) and (abs(x2 - x1) < int(l * 0.25)):
                cv2.line(image,(x1,y1),(x2,y2),(0,255,255),2)
                # Maintain a simples lookup list for points
                lines_list.append([(x1,y1),(x2,y2)])
                if (y1 < highest[0][1]):
                    highest = [(x1,y1),(x2,y2)]
                # if (y1 > lowest[0][1]):
                #     lowest = [(x1,y1),(x2,y2)]
        
    
    #cv2.line(image, lowest[0], lowest[1], (255, 0, 0), 3)
    
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
    
    
    """
    minArea = int(h * l * 0.15) 
    
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    
    cnts = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    
    new_cnts = []
    for cnt in cnts:
        approx = cv2.contourArea(cnt)
        if approx > minArea:
            new_cnts.append(cnt)
        #print(approx)
    
    for i in cnts:
            epsilon = 0.05*cv2.arcLength(i,False)
            approx = cv2.approxPolyDP(i,epsilon,False)
            if len(approx) == 4:
                cv2.drawContours(img,cnts,-1,(0,255,0),2)
    
    #cv2.drawContours(img, new_cnts, -1, (0,255,255), 3)"""
    image = cv2.resize(image, (int(l/2), int(h/2)))
    
    cv2.imshow('image', image)
    #cv2.imshow('Binary',thresh_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()