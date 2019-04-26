import cv2
import numpy as np
from numpy import *

text_file = open("smooth_trajectory.txt","r")
lines = text_file.readlines()
text_file.close()
trajectory = []
for num in range(1, len(lines)):
    trajectory.append(eval(lines[num][:-1]))

with open ('smooth_trajectory.txt','r') as f:
    for smth_traject in smoothed_trajectory:
	sys = smth_traject
	f.write(str(sys)+'\n')

prof_window_x = reference_x
prof_window_y = reference_y

cnt = 0
out_cap = cv2.VideoCapture("cut_short2.mp4")
while True:
    outret, outframe = out_cap.read()
    height, width, channels = outframe.shape
	    
    if outret:
	out_frame = cv2.resize(outframe,(width/2, height/2))
	height_, width_, channels_ = out_frame.shape

	prof_out = out_frame[prof_window_y:prof_window_y + sh, prof_window_x:prof_window_x + sw]

	prof_im = cv2.resize(prof_out, (200, 256), interpolation = cv2.INTER_CUBIC)	
	video2.write(prof_im)

	#print smoothed_trajectory[cnt][1]
	prof_window_x = (reference_x + traj_list[cnt][1])
	prof_window_y = (reference_y + traj_list[cnt][2])
	cnt = cnt + 1
	print cnt
	#prof = im_color[CMT.tl[1] : CMT.br[1], CMT.tl[0] : CMT.br[0]]
