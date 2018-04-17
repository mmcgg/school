import numpy as np
import cv2

cap = cv2.VideoCapture(0)


while(True):
    # Capture frame by frame
    ret, frame = cap.read()
    
    if(ret==True):
        
        # For the left hand
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        minHSV = np.array([0, 150, 0])
        maxHSV = np.array([180, 255, 255])
        maskHSV = cv2.inRange(hsv, minHSV, maxHSV)

        cv2.imshow("High saturation",maskHSV)


        minHSV = np.array([0, 0, 150])
        maxHSV = np.array([180, 255, 255])
        maskHSV = cv2.inRange(hsv, minHSV, maxHSV)

        cv2.imshow("High Value",maskHSV)

        minHSV = np.array([0, 0, 0])
        maxHSV = np.array([180, 100, 255])
        maskHSV = cv2.inRange(hsv, minHSV, maxHSV)

        cv2.imshow("Low Saturation",maskHSV)


        minHSV = np.array([0, 0, 0])
        maxHSV = np.array([180, 255, 100])
        maskHSV = cv2.inRange(hsv, minHSV, maxHSV)

        cv2.imshow("Low Value",maskHSV)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        


