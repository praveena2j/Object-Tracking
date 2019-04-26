import cv2
import numpy as np
from numpy import *
cap = cv2.VideoCapture("343.mp4")

ret, frame = cap.read()
height, width, channels = frame.shape
frame1 = cv2.resize(frame,(width/2, height/2))
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255

gFrCascadePath = "haarcascade_frontalface.xml"
frCC = cv2.CascadeClassifier(gFrCascadePath)

# face detection for deciding the tracking zone 
faces = frCC.detectMultiScale(frame1, 1.2, 3, 100)
(r, c, w, h) = faces[0]

# mask for motion in tracking zone
HSV_test = hsv[c-2*h : c + 3*h, r-2*w : r+3*w]

# face detection for mean shift window
face_frame = frame1[c-2*h : c + 3*h, r-2*w : r+3*w]
faces_frame = frCC.detectMultiScale(face_frame, 1.2, 3, 100)
(fr,fc, fw, fh) = faces_frame[0]

# initialization for tracking window for meanshift
trackingwindow = (fr-fw, fc-fh, 5*fw, 5*fh)

# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

# cropping area for trackingzone
trackingzone_height = r - 2*w
trackingzone_height_ = trackingzone_height + 5*w
trackingzone_width = c - 2*h
trackingzone_width_ = trackingzone_width + 5*h

while(1):
    ret, frame3 = cap.read()
    frame2 = cv2.resize(frame3,(width/2, height/2))
    nextfm = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

    next_frame = nextfm[trackingzone_width : trackingzone_width_, trackingzone_height : trackingzone_height_]
    previous_frame = prvs[trackingzone_width : trackingzone_width_, trackingzone_height : trackingzone_height_]

    # Optical Flow Computation
    flow = cv2.calcOpticalFlowFarneback(previous_frame, next_frame, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])

    HSV_test[...,0] = ang*180/np.pi/2
    HSV_test[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(HSV_test,cv2.COLOR_HSV2BGR)
   
    #  Motion mask in the tracking zone
    dst = cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY)
    ret1,thresh1 = cv2.threshold(dst,17,255,cv2.THRESH_BINARY)    

    # apply meanshift to get the new location
    ret, ms_track_window = cv2.meanShift(thresh1, trackingwindow, term_crit)

    # mean center of the mean shift window
    meancenter_x = ms_track_window[0] + (ms_track_window[2]/2)
    meancenter_y = ms_track_window[1] + (ms_track_window[3]/2)
    
    sx, sy, sw, sh = ms_track_window

    mshift_frame = frame2[trackingzone_width : trackingzone_width_, trackingzone_height : trackingzone_height_]
    img2 = cv2.rectangle(mshift_frame,(sx, sy), (sx+sw, sy+sh), (255,255,0),1)
    # img2 = cv2.rectangle(mshift_frame,(meancenter_x, meancenter_y), (meancenter_x+1, meancenter_y+1), (255,255,0),1)
    img3 = mshift_frame[sy : sy + sh, sx : sx + sw] 

    cv2.rectangle(frame2, (meancenter_x+trackingzone_height, meancenter_y+trackingzone_width), (meancenter_x+trackingzone_height+1, meancenter_y+trackingzone_width+1), (255,255,0),1)
    cv2.rectangle(frame2,(sx + trackingzone_height, sy + trackingzone_width), (sx+sw+trackingzone_height, sy+sh+ trackingzone_width), (255,255,0),1)
    size =  img3.shape
    print size
    cv2.imshow("img_track", img3)
    cv2.imshow("imgtra", frame2)
    # cv2.waitKey(100)
    ms_track_window_list = list(ms_track_window)
    trackingwindow_list = list(trackingwindow)
    
    # Updation of the tracking zone
    trackingzone_width = meancenter_y + trackingzone_width - (355/2) 
    trackingzone_height = meancenter_x + trackingzone_height - (355/2)
    trackingzone_width_ = trackingzone_width + 355 
    trackingzone_height_ = trackingzone_height + 355
    
    #trackingzone_width = trackingzone_width + (ms_track_window_list[1] - trackingwindow_list[1]) 
    #trackingzone_height = trackingzone_height + (ms_track_window_list[0] - trackingwindow_list[0])
    #trackingzone_width_ = trackingzone_width + 5*h 
    #trackingzone_height_ = trackingzone_height + 5*w 

    size_next = nextfm.shape

    if trackingzone_width < 0:
    	trackingzone_width = 0
    if trackingzone_height < 0:
	trackingzone_height = 0
    if trackingzone_width_ > size_next[0]:
  	trackingzone_width_ = size_next[0]
	trackingzone_width = size_next[0] - 355
    if trackingzone_height_ > size_next[1]:
        trackingzone_height_ = size_next[1]
        trackingzone_height = size_next[1] - 355
 

    # trackingwindow[0] = meancenter_x
    trackingwindow = (100, 100, 5*fw, 5*fh)
 
    # trackingwindow = tuple(ms_track_window_list)
    k = cv2.waitKey(60) & 0xff
    if k == 27:
        break
    else:
        cv2.imwrite(chr(k)+".jpg",img3)  
    prvs = nextfm.copy()

cv2.destroyAllWindows()
cap.release()
