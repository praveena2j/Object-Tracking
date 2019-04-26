import cv2
import numpy as np
from numpy import *

text_file = open("trajectory.txt","r")
lines = text_file.readlines()
text_file.close()
trajectory = []
for num in range(1, len(lines)):
    trajectory.append(eval(lines[num][:-1]))

reference_x = 630
reference_y = 364
sw = 216
sh = 338

smoothing_radius = 10
smoothed_trajectory = []
tr = 0

fourcc = cv2.cv.CV_FOURCC(*'XVID')		
video = cv2.VideoWriter('output_short.avi',fourcc,25.0,(200,256))

for traj in trajectory:
    sum_x = 0
    sum_y = 0
    count = 0
    for num in range(-smoothing_radius, smoothing_radius):
        if ((tr + num) >=0) and ((tr + num) < len(trajectory)):	    
	    sum_x = sum_x + trajectory[tr+num][1]
	    sum_y = sum_y + trajectory[tr+num][2]
	    count = count + 1
    avg_x = sum_x/count
    avg_y = sum_y/count
    smooth_traj_tuple = (tr, avg_x, avg_y)
    smoothed_trajectory.append(smooth_traj_tuple)
    tr = tr + 1

with open ('smooth_trajectory.txt','w') as f:
    for smth_traject in smoothed_trajectory:
        sys = smth_traject
	f.write(str(sys)+'\n')

prof_window_x = reference_x - (sw/2)
prof_window_y = reference_y - (sh/2)

cnt = 0
out_cap = cv2.VideoCapture("cut_short.mp4")
while True:
    outret, outframe = out_cap.read()
    height, width, channels = outframe.shape
    
    if outret:
	out_frame = cv2.resize(outframe,(width/2, height/2))
	prof_out = out_frame[prof_window_y:prof_window_y + sh, prof_window_x:prof_window_x + sw]

	prof_im = cv2.resize(prof_out, (200, 256), interpolation = cv2.INTER_CUBIC)	
	video.write(prof_im)

	cv2.imshow("prof", prof_out)
	cv2.waitKey(100)
	#print smoothed_trajectory[cnt][1]
	prof_window_x = (reference_x + trajectory[cnt][1]) - (sw/2)
	prof_window_y = (reference_y + trajectory[cnt][2]) - (sh/2)
        cnt = cnt + 1
	print cnt
