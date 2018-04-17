import numpy as np
import cv2
import time
from scipy import stats

cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture('./mv2_001.avi')
fgbg = cv2.createBackgroundSubtractorMOG2()
first = 1
delta_t = 1.0/30.0

meanshift_kalman = cv2.KalmanFilter(4,2)
meanshift_kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
meanshift_kalman.transitionMatrix = np.array([[1,0,delta_t,0],[0,1,0,delta_t],[0,0,1,0],[0,0,0,1]],np.float32)
meanshift_kalman.processNoiseCov = np.array([[1.0/3.0*delta_t**3.0,0,1.0/2.0*delta_t**2.0,0],[0,1.0/3.0*delta_t**3.0,0,1.0/2.0*delta_t**2.0],[1.0/2.0*delta_t**2.0,0,delta_t,0],[0,1.0/2.0*delta_t**2.0,0,delta_t]],np.float32) * 100.1

camshift_kalman = cv2.KalmanFilter(4,2)
camshift_kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
camshift_kalman.transitionMatrix = np.array([[1,0,delta_t,0],[0,1,0,delta_t],[0,0,1,0],[0,0,0,1]],np.float32)
camshift_kalman.processNoiseCov = np.array([[1.0/3.0*delta_t**3.0,0,1.0/2.0*delta_t**2.0,0],[0,1.0/3.0*delta_t**3.0,0,1.0/2.0*delta_t**2.0],[1.0/2.0*delta_t**2.0,0,delta_t,0],[0,1.0/2.0*delta_t**2.0,0,delta_t]],np.float32) * 100.1

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

        if first==1:
            first=0
            last_frame = frame

            # Select a region of interest
            c,r,w,h = cv2.selectROI("Image",frame,False,False)
            camshift_track_window = [c,r,w,h]
            meanshift_track_window = [c,r,w,h]

            camshift_tracked_point = np.array([c+w/2,r+h/2])
            meanshift_tracked_point = np.array([c+w/2,r+h/2])

            # Set up the Kalman filter
            meanshift_kalman.statePost = np.array([[meanshift_tracked_point[0]],
                                                   [meanshift_tracked_point[1]],
                                                   [0.0],
                                                   [0.0]],dtype=np.float32)
            camshift_kalman.statePost = np.array([[camshift_tracked_point[0]],
                                                   [camshift_tracked_point[1]],
                                                   [0.0],
                                                   [0.0]],dtype=np.float32)

            meanshift_kalman.errorCovPost = 1.1*np.eye(4,dtype=np.float32)
            camshift_kalman.errorCovPost = 1.1*np.eye(4,dtype=np.float32)

            # set up the ROI for tracking
            roi = frame[r:r+h, c:c+w]
            hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            hue_mode = np.array(stats.mode(hsv_roi[:,:,0])).flatten()[0]
            mask = cv2.inRange(hsv_roi, np.array((hue_mode-1, 10.,10.)), np.array((hue_mode+1,255.,255.)))
            roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
            cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
            
            # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
            term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
        
        # Predict the state
        meanshift_state = meanshift_kalman.predict()
        meanshift_track_window[0] = meanshift_state[0] - meanshift_track_window[2]/2
        meanshift_track_window[1] = meanshift_state[1] - meanshift_track_window[3]/2
        meanshift_track_window[2] = meanshift_track_window[2]
        meanshift_track_window[3] = meanshift_track_window[3]

        # Get a measurement from meanshift
        meanshift_track_window = tuple(meanshift_track_window)
        ret, meanshift_track_window = cv2.meanShift(dst, meanshift_track_window, term_crit)
        meanshift_track_window = list(meanshift_track_window)
        measurement = np.array([[meanshift_track_window[0] + meanshift_track_window[2]/2],
                                [meanshift_track_window[1] + meanshift_track_window[3]/2]],dtype=np.float32)
        
        # Correct the state based on a measurement
        meanshift_state = meanshift_kalman.correct(measurement)
        meanshift_track_window[0] = meanshift_state[0] - meanshift_track_window[2]/2
        meanshift_track_window[1] = meanshift_state[1] - meanshift_track_window[3]/2
        meanshift_track_window[2] = meanshift_track_window[2]
        meanshift_track_window[3] = meanshift_track_window[3]

        
        # Draw meanshift on image
        x,y,w,h = meanshift_track_window
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0),2)

        # Predict the state
        camshift_state = camshift_kalman.predict()
        camshift_track_window[0] = camshift_state[0] - camshift_track_window[2]/2
        camshift_track_window[1] = camshift_state[1] - camshift_track_window[3]/2
        camshift_track_window[2] = camshift_track_window[2]
        camshift_track_window[3] = camshift_track_window[3]

        # Get a measurement from camshift
        camshift_track_window = tuple(camshift_track_window)
        ret, camshift_track_window = cv2.CamShift(dst, camshift_track_window, term_crit)
        camshift_track_window = list(camshift_track_window)
        measurement = np.array([[camshift_track_window[0] + camshift_track_window[2]/2],
                                [camshift_track_window[1] + camshift_track_window[3]/2]],dtype=np.float32)
        
        # Correct the state based on a measurement
        camshift_state = camshift_kalman.correct(measurement)
        camshift_track_window[0] = camshift_state[0] - camshift_track_window[2]/2
        camshift_track_window[1] = camshift_state[1] - camshift_track_window[3]/2
        camshift_track_window[2] = camshift_track_window[2]
        camshift_track_window[3] = camshift_track_window[3]
        
        
        # Draw camshift on image
        pts = cv2.boxPoints(ret)
        pts = np.int0(pts)
        cv2.polylines(frame,[pts],True, (0,0,255),2)
        cv2.imshow('Mean and Cam shift',frame)

    else:
            break

# When everythin is done, release the capture
cap.release()
cv2.destroyAllWindows()
