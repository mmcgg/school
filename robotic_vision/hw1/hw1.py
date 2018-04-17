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
    
        # Our operations on the frame come here
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Write the frame to file
        out.write(frame)

        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
            break

# When everythin is done, release the capture
cap.release()
out.release()
cv2.destroyAllWindows()
