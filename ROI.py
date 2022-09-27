import cv2 #opencv
import numpy as np
#Seleccionamos la camara a usar conectada por usb
vc = cv2.VideoCapture(0)
folder = "0"
contador = 0
while True:
    next, frame = vc.read()
    frame2 = frame.copy()
    #primero
    start_point = (20, 0)
    end_point = (20, 480)
    #segundo
    start_point2 = (170, 0)
    end_point2 = (170, 480)
    #tercero
    start_point3 = (320, 0)
    end_point3 = (320, 480)
    #cuarto
    start_point4 = (470, 0)
    end_point4 = (470, 480)
    #quinto
    start_point5 = (620, 0)
    end_point5 = (620, 480)
    #medio
    start_point6 = (20, 240)
    end_point6 = (620, 240)

    # Green color in BGR
    color = (0, 255, 0)

    # Line thickness of 9 px
    thickness = 1

    # Using cv2.line() method
    # Draw a diagonal green line with thickness of 9 px
    cv2.line(frame, start_point, end_point, color, thickness)
    cv2.line(frame, start_point2, end_point2, color, thickness)
    cv2.line(frame, start_point3, end_point3, color, thickness)
    cv2.line(frame, start_point4, end_point4, color, thickness)
    cv2.line(frame, start_point5, end_point5, color, thickness)
    cv2.line(frame, start_point6, end_point6, color, thickness)

    cv2.imshow("frame",frame)
    if cv2.waitKey(50) >= 0:
        img1 = frame2[0:240,20:170]
        img2 = frame2[0:240,170:320]
        img3 = frame2[0:240,320:470]
        img4 = frame2[0:240,470:620]
        img5 = frame2[240:480,20:170]
        img6 = frame2[240:480,170:320]
        img7 = frame2[240:480,320:470]
        img8 = frame2[240:480,470:620]
        cv2.imwrite(folder+"/"+str(contador)+".jpg",img1)
        cv2.imwrite(folder+"/"+str(contador+1)+".jpg",img2)
        cv2.imwrite(folder+"/"+str(contador+2)+".jpg",img3)
        cv2.imwrite(folder+"/"+str(contador+3)+".jpg",img4)
        cv2.imwrite(folder+"/"+str(contador+4)+".jpg",img5)
        cv2.imwrite(folder+"/"+str(contador+5)+".jpg",img6)
        cv2.imwrite(folder+"/"+str(contador+6)+".jpg",img7)
        cv2.imwrite(folder+"/"+str(contador+7)+".jpg",img8)
        contador += 8
        print("Imagenes capturadas!!!!! Total: ",contador)