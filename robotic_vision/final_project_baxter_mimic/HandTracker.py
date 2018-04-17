import numpy as np
import cv2
import time
from scipy import stats

#Blue ball Hue: 100
#Orange ball Hue: 5
#Pink ball Hue: 170

class HandTracker():

    def __init__(self,cap,use_spheres=True):
        
        self.cap = cap
        ret, frame = self.cap.read()
        if(ret==True):
            self.last_frame = frame

        delta_t = 1/30.0

        # Left hand kalman filter
        # Constant Velocity Model
        self.left_kalman = cv2.KalmanFilter(4,2)
        self.left_kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
        self.left_kalman.transitionMatrix = np.array([[1,0,delta_t,0],[0,1,0,delta_t],[0,0,1,0],[0,0,0,1]],np.float32)
        self.left_kalman.processNoiseCov = np.array([[1.0/3.0*delta_t**3.0,0,1.0/2.0*delta_t**2.0,0],[0,1.0/3.0*delta_t**3.0,0,1.0/2.0*delta_t**2.0],[1.0/2.0*delta_t**2.0,0,delta_t,0],[0,1.0/2.0*delta_t**2.0,0,delta_t]],np.float32) * 5000.0

        # Right hand kalman filter
        # Constant Velocity Model
        self.right_kalman = cv2.KalmanFilter(4,2)
        self.right_kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
        self.right_kalman.transitionMatrix = np.array([[1,0,delta_t,0],[0,1,0,delta_t],[0,0,1,0],[0,0,0,1]],np.float32)
        self.right_kalman.processNoiseCov = np.array([[1.0/3.0*delta_t**3.0,0,1.0/2.0*delta_t**2.0,0],[0,1.0/3.0*delta_t**3.0,0,1.0/2.0*delta_t**2.0],[1.0/2.0*delta_t**2.0,0,delta_t,0],[0,1.0/2.0*delta_t**2.0,0,delta_t]],np.float32) * 500.0

        # Setup termination criteria for cam/mean shift, either 10 iteration or move by atleast 1 pt
        self.term_crit = ( cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 0, 1000000 )
        
        # Let the camera get its white balance adjusted
        start = time.time()
        while time.time()-start<1:
            # Capture frame by frame
            ret, frame = self.cap.read()


    def select_hands(self):
        ret, frame = self.cap.read()
        
        # Select a region of interest for left hand
        c,r,w,h = cv2.selectROI("Left hand",frame,False,False)
        self.left_track_window = [c,r,w,h]
        self.left_tracked_point = np.array([c+w/2,r+h/2])

        # Set up the Kalman filter
        self.left_kalman.statePost = np.array([[self.left_tracked_point[0]],
                                               [self.left_tracked_point[1]],
                                               [0.0],
                                               [0.0]],dtype=np.float32)
        self.left_kalman.errorCovPost = 1.1*np.eye(4,dtype=np.float32)
        
        # set up the ROI for tracking
        roi = frame[r:r+h, c:c+w]
        hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hue_mode = np.array(stats.mode(hsv_roi[:,:,0])).flatten()[0]
        mask = cv2.inRange(hsv_roi, np.array((hue_mode-1, 10.,10.)), np.array((hue_mode+1,255.,255.)))
        self.left_roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
        cv2.normalize(self.left_roi_hist,self.left_roi_hist,0,255,cv2.NORM_MINMAX)

        # Find the area0 for the left hand
        left_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Pick out the red gloves
        minHSV = np.array([160, 100, 0])
        maxHSV = np.array([180, 255, 255])
        maskHSV = cv2.inRange(left_hsv, minHSV, maxHSV)
        
        # Get a measurement from self.left_kalman
        self.left_track_window = tuple(self.left_track_window)
        ret, self.left_track_window = cv2.CamShift(maskHSV, self.left_track_window, self.term_crit)
        self.left_track_window = list(self.left_track_window)
        self.left_area0 = ret[1][0]*ret[1][1]

        # left_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # left_dst = cv2.calcBackProject([left_hsv],[0],self.left_roi_hist,[0,180],1)
        # self.left_track_window = tuple(self.left_track_window)
        # ret, self.left_track_window = cv2.CamShift(left_dst, self.left_track_window, self.term_crit)
        # self.left_track_window = list(self.left_track_window)        
        # self.left_area0 = ret[1][0]*ret[1][1]

        
        # Select a region of interest for right hand
        c,r,w,h = cv2.selectROI("Right hand",frame,False,False)
        self.right_track_window = [c,r,w,h]
        self.right_tracked_point = np.array([c+w/2,r+h/2])

        # Set up the Kalman filter
        self.right_kalman.statePost = np.array([[self.right_tracked_point[0]],
                                                    [self.right_tracked_point[1]],
                                                    [0.0],
                                                    [0.0]],dtype=np.float32)
        self.right_kalman.errorCovPost = 1.1*np.eye(4,dtype=np.float32)
        
        # set up the ROI for tracking
        roi = frame[r:r+h, c:c+w]
        hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hue_mode = np.array(stats.mode(hsv_roi[:,:,0])).flatten()[0]
        mask = cv2.inRange(hsv_roi, np.array((hue_mode-1, 10.,10.)), np.array((hue_mode+1,255.,255.)))
        self.right_roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
        cv2.normalize(self.right_roi_hist,self.right_roi_hist,0,255,cv2.NORM_MINMAX)

        # Find the area0 for the right hand
        right_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        right_dst = cv2.calcBackProject([right_hsv],[0],self.right_roi_hist,[0,180],1)
        self.right_track_window = tuple(self.right_track_window)
        ret, self.right_track_window = cv2.CamShift(right_dst, self.right_track_window, self.term_crit)
        self.right_track_window = list(self.right_track_window)        
        self.right_area0 = ret[1][0]*ret[1][1]


        # Draw left area on image
        pts = cv2.boxPoints(ret)
        pts = np.int0(pts)
        cv2.polylines(frame,[pts],True,(0,255,0),2)        
        
        # Draw right area on image
        pts = cv2.boxPoints(ret)
        pts = np.int0(pts)
        cv2.polylines(frame,[pts],True,(255,0,0),2)
        cv2.imshow('Original Areas',frame)        
        

    def track(self):
        while(True):
            # Capture frame by frame
            ret, frame = cap.read()

            if(ret==True):

                # For the left hand
                left_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                # left_dst = cv2.calcBackProject([left_hsv],[0],self.left_roi_hist,[0,180],1)

                # Pick out the pink ball
                minHSV = np.array([160, 0, 0])
                maxHSV = np.array([180, 255, 255])
                maskHSV = cv2.inRange(left_hsv, minHSV, maxHSV)

                kernel = np.ones((3,3),np.uint8)
                maskHSV = cv2.erode(maskHSV,kernel,iterations = 1)
                maskHSV = cv2.dilate(maskHSV,kernel,iterations = 1)                

                # Predict the left hand state
                self.left_state = self.left_kalman.predict()
                self.left_track_window[0] = self.left_state[0] - self.left_track_window[2]/2
                self.left_track_window[1] = self.left_state[1] - self.left_track_window[3]/2
                self.left_track_window[2] = self.left_track_window[2]
                self.left_track_window[3] = self.left_track_window[3]
                
                # Get a measurement from self.left_kalman
                self.left_track_window = tuple(self.left_track_window)
                # ret, self.left_track_window = cv2.CamShift(left_dst, self.left_track_window, self.term_crit)
                ret, self.left_track_window = cv2.CamShift(maskHSV, self.left_track_window, self.term_crit)
                area = ret[1][0]*ret[1][1]

                try:
                    d = 1e9*.004*.02*(1.0/self.left_area0 - 1.0/area)
                except:
                    print "Lost the glove"

                print d
                    
                
                self.left_track_window = list(self.left_track_window)
                measurement = np.array([[self.left_track_window[0] + self.left_track_window[2]/2],
                                        [self.left_track_window[1] + self.left_track_window[3]/2]],dtype=np.float32)
                
                # Correct the state based on a measurement
                self.left_state = self.left_kalman.correct(measurement)
                self.left_track_window[0] = self.left_state[0] - self.left_track_window[2]/2
                self.left_track_window[1] = self.left_state[1] - self.left_track_window[3]/2
                self.left_track_window[2] = self.left_track_window[2]
                self.left_track_window[3] = self.left_track_window[3]
                
                
                # Draw left on image
                pts = cv2.boxPoints(ret)
                pts = np.int0(pts)
                x,y,w,h = self.left_track_window
                # cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0),2)
                cv2.polylines(frame,[pts],True,(0,255,0),2)


                # For the right hand
                right_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                right_dst = cv2.calcBackProject([right_hsv],[0],self.right_roi_hist,[0,180],1)
        
                # Predict the right hand state
                self.right_state = self.right_kalman.predict()
                self.right_track_window[0] = self.right_state[0] - self.right_track_window[2]/2
                self.right_track_window[1] = self.right_state[1] - self.right_track_window[3]/2
                self.right_track_window[2] = self.right_track_window[2]
                self.right_track_window[3] = self.right_track_window[3]
                
                # Get a measurement from self.right_kalman
                self.right_track_window = tuple(self.right_track_window)
                ret, self.right_track_window = cv2.CamShift(right_dst, self.right_track_window, self.term_crit)


                
                self.right_track_window = list(self.right_track_window)
                measurement = np.array([[self.right_track_window[0] + self.right_track_window[2]/2],
                                        [self.right_track_window[1] + self.right_track_window[3]/2]],dtype=np.float32)
                
                # Correct the state based on a measurement
                self.right_state = self.right_kalman.correct(measurement)
                self.right_track_window[0] = self.right_state[0] - self.right_track_window[2]/2
                self.right_track_window[1] = self.right_state[1] - self.right_track_window[3]/2
                self.right_track_window[2] = self.right_track_window[2]
                self.right_track_window[3] = self.right_track_window[3]
                
                
                # Draw right on image
                pts = cv2.boxPoints(ret)
                pts = np.int0(pts)
                cv2.polylines(frame,[pts],True,(255,0,0),2)
                x,y,w,h = self.right_track_window
                
                cv2.imshow('Hands',frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break





if __name__ == '__main__':
        
    cap = cv2.VideoCapture(0)

    ht = HandTracker(cap,use_spheres=True)
    #ht.select_hands()
    time.sleep(1)
    ht.track()
    
