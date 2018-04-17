import numpy as np
import cv2
import time
from scipy import stats
from copy import copy

cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture('./mv2_001.avi')
fgbg = cv2.createBackgroundSubtractorMOG2()
first = 1

# Set up the detector with default parameters.

# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()

# Change thresholds
params.minThreshold = 0
params.maxThreshold = 255


# Filter by Area.
params.filterByArea = True
params.minArea = 1500

# Filter by Circularity
params.filterByCircularity = False
params.minCircularity = 0.1

# Filter by Convexity
params.filterByConvexity = False
params.minConvexity = 0.87

# Filter by Inertia
params.filterByInertia = False
params.minInertiaRatio = 0.01

blob_detector = cv2.SimpleBlobDetector_create(params) 

# Let the camera get its white balance adjusted
start = time.time()
while time.time()-start<1:
    # Capture frame by frame
    ret, frame = cap.read()

    if ret==True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # Display the original frame
        cv2.imshow('Original', frame)



while True:
    # Capture frame by frame
    ret, frame = cap.read()

    if ret==True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_mod = copy(frame)

        if first==1:
            first=0
            last_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Select a region of interest
            c,r,w,h = cv2.selectROI("Image",frame_mod,False,False)
            camshift_track_window = (c,r,w,h)
            meanshift_track_window = (c,r,w,h)

            # set up the ROI for tracking
            roi = frame_mod[r:r+h, c:c+w]
            hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            hue_mode = np.array(stats.mode(hsv_roi[:,:,0])).flatten()[0]
        
            mask = cv2.inRange(hsv_roi, np.array((hue_mode-1, 0.,0.)), np.array((hue_mode+1,255.,255.)))
            roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
            cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
            
            # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
            term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

        
        hsv = cv2.cvtColor(frame_mod, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
        
        ret, meanshift_track_window = cv2.meanShift(dst, meanshift_track_window, term_crit)
        
        # Draw meanshift on image
        x,y,w,h = meanshift_track_window
        cv2.rectangle(frame_mod, (x,y), (x+w,y+h), (0,255,0),2)

        # Get a measurement from camshift
        ret, camshift_track_window = cv2.CamShift(dst, camshift_track_window, term_crit)
        
        # Draw camshift on image
        pts = cv2.boxPoints(ret)
        pts = np.int0(pts)
        cv2.polylines(frame_mod,[pts],True, (0,0,255),2)
        #cv2.imshow('Mean and Cam shift',frame_mod)

        # Do Background subtraction
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        subtracted = cv2.absdiff(frame,last_frame)*2
        ret,thresh = cv2.threshold(subtracted,10,255,cv2.THRESH_BINARY)
        thresh = cv2.erode(thresh,np.ones([3,3]),iterations=1)
        thresh = cv2.blur(thresh,(3,3))
        thresh = cv2.dilate(thresh,np.ones([3,3]),iterations=1)
        # thresh = cv2.blur(thresh,(3,3))
        ret,thresh = cv2.threshold(subtracted,10,255,cv2.THRESH_BINARY)
        thresh = cv2.erode(thresh,np.ones([3,3]),iterations=1)
        thresh = cv2.blur(thresh,(3,3))
        thresh = cv2.dilate(thresh,np.ones([3,3]),iterations=1)
        
        cv2.imshow('Thresholded',thresh)

 
        # Detect blobs.
        keypoints = blob_detector.detect(thresh)

        # Draw detected blobs as red circles.
        # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
        frame_mod = cv2.drawKeypoints(frame_mod, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
 
        # Show keypoints
        cv2.imshow("Object Trackers", frame_mod)

        # save the frame as the previous frame
        last_frame = frame
        

    else:
            break

# When everythin is done, release the capture
cap.release()
cv2.destroyAllWindows()
