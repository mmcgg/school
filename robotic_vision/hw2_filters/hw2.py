import numpy as np
import cv2

cap = cv2.VideoCapture(0)

# Define the video codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
out = cv2.VideoWriter('output.avi', fourcc, 30.0, (640,480))

while True:
    # Capture frame by frame
    ret, frame = cap.read()

    if ret==True:

        # Display the original frame
        cv2.imshow('Original', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    
        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Edge detection
        edges = cv2.Canny(frame, 150, 300)

        # Display the canny edge detection image
        cv2.imshow('Canny', edges)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        

        #frame = gray
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        kernel = np.array([[1,0,-1],
                           [1,0,-1],
                           [1,0,-1]])
        dst_hor1 = cv2.filter2D(frame,-1,kernel)
        dst_hor2 = cv2.filter2D(frame,-1,kernel*-1.0)        

        kernel = np.array([[1,1,1],
                           [0,0,0],
                           [-1,-1,-1]])
        dst_vert1 = cv2.filter2D(frame,-1,kernel)
        dst_vert2 = cv2.filter2D(frame,-1,kernel*-1.0)        

        dst = dst_hor1 + dst_hor2 + dst_vert1 + dst_vert2

        #ret,dst = cv2.threshold(dst,100,255,cv2.THRESH_BINARY)

        # Display the custom kernel filtered image
        cv2.imshow('Custom Filter', dst)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        

        # Write the frame to file
        # out.write(frame)
        
    else:
            break

# When everythin is done, release the capture
cap.release()
out.release()
cv2.destroyAllWindows()
