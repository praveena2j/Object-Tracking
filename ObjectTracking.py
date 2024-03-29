#!/usr/bin/env python

import argparse
import cv2
from numpy import empty, nan
import os
import sys
import time

import CMT
import numpy as np
import util
from scipy.misc import imread, imsave, imresize

CMT = CMT.CMT()

parser = argparse.ArgumentParser(description='Track an object.')

parser.add_argument('inputpath', nargs='?', help='The input path.')
parser.add_argument('--challenge', dest='challenge', action='store_true', help='Enter challenge mode.')
parser.add_argument('--preview', dest='preview', action='store_const', const=True, default=None, help='Force preview')
parser.add_argument('--no-preview', dest='preview', action='store_const', const=False, default=None, help='Disable preview')
parser.add_argument('--no-scale', dest='estimate_scale', action='store_false', help='Disable scale estimation')
parser.add_argument('--with-rotation', dest='estimate_rotation', action='store_true', help='Enable rotation estimation')
parser.add_argument('--bbox', dest='bbox', help='Specify initial bounding box.')
parser.add_argument('--pause', dest='pause', action='store_true', help='Pause after every frame and wait for any key.')
parser.add_argument('--output-dir', dest='output', help='Specify a directory for output data.')
parser.add_argument('--quiet', dest='quiet', action='store_true', help='Do not show graphical output (Useful in combination with --output-dir ).')
parser.add_argument('--skip', dest='skip', action='store', default=None, help='Skip the first n frames', type=int)

args = parser.parse_args()

CMT.estimate_scale = args.estimate_scale
CMT.estimate_rotation = args.estimate_rotation

if args.pause:
	pause_time = 0
else:
	pause_time = 10

if args.output is not None:
	if not os.path.exists(args.output):
		os.mkdir(args.output)
	elif not os.path.isdir(args.output):
		raise Exception(args.output + ' exists, but is not a directory')

if args.challenge:
	with open('images.txt') as f:
		images = [line.strip() for line in f]

	init_region = np.genfromtxt('region.txt', delimiter=',')
	num_frames = len(images)

	results = empty((num_frames, 4))
	results[:] = nan

	results[0, :] = init_region

	frame = 0

	im0 = cv2.imread(images[frame])
	im_gray_ = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)
	height, width, channels = im_gray_.shape
	im_gray0 = cv2.resize(im_gray,(width/2, height/2))
	im_draw = np.copy(im0)

	tl, br = (util.array_to_int_tuple(init_region[:2]), util.array_to_int_tuple(init_region[:2] + init_region[2:4] - 1))

	try:
		CMT.initialise(im_gray0, tl, br)
		while frame < num_frames:
			im = cv2.imread(images[frame])
			im_gray_2 = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
			im_gray = cv2.resize(im_gray_2,(width/2, height/2))
			CMT.process_frame(im_gray_2)
			results[frame, :] = CMT.bb

			# Advance frame number
			frame += 1
	except:
		pass  # Swallow errors

	np.savetxt('output.txt', results, delimiter=',')
else:
	# Clean up
	cv2.destroyAllWindows()

	preview = args.preview
	if args.inputpath is not None:

	        # If a path to a file was given, assume it is a single video file
		if os.path.isfile(args.inputpath):
			cap = cv2.VideoCapture(args.inputpath)

			#Skip first frames if required
			if args.skip is not None:
				cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, args.skip)

		# Otherwise assume it is a format string for reading images
		else:
			cap = util.FileVideoCapture(args.inputpath)

			#Skip first frames if required
			if args.skip is not None:
				cap.frame = 1 + args.skip

		# By default do not show preview in both cases
		if preview is None:
			preview = False

	else:
		# If no input path was specified, open camera device
		cap = cv2.VideoCapture(0)
		if preview is None:
			preview = True

	# Check if videocapture is working
	if not cap.isOpened():
		print 'Unable to open video input.'
		sys.exit(1)

	while preview:
		status, im = cap.read()
		cv2.imshow('Preview', im)
		k = cv2.waitKey(10)
		if not k == -1:
			break

	# Read first frame
	status, im0 = cap.read()
	height, width, channels = im0.shape
	
	if height > 1000:	
		im_gray = cv2.resize(im0,(width/2, height/2))
	else:
		im_gray = np.copy(im0)
	im_gray0 = cv2.cvtColor(im_gray, cv2.COLOR_BGR2GRAY)
	im_draw = np.copy(im_gray)

	if args.bbox is not None:
		# Try to disassemble user specified bounding box
		values = args.bbox.split(',')
		try:
			values = [int(v) for v in values]
		except:
			raise Exception('Unable to parse bounding box')
		if len(values) != 4:
			raise Exception('Bounding box must have exactly 4 elements')
		bbox = np.array(values)

		# Convert to point representation, adding singleton dimension
		bbox = util.bb2pts(bbox[None, :])

		# Squeeze
		bbox = bbox[0, :]

		tl = bbox[:2]
		br = bbox[2:4]
	else:
		# Get rectangle input from user
		(tl, br) = util.get_rect(im_draw)

	print 'using', tl, br, 'as init bb'

	sh = (br[1] - tl[1])
	sw = (br[0] - tl[0])

	CMT.initialise(im_gray, tl, br)

	frame = 1
	i = 0

	fourcc = cv2.cv.CV_FOURCC(*'XVID')		
	video1 = cv2.VideoWriter('output.avi',fourcc,25.0,(100,156))
	im_prvs	= np.copy(im_gray0)
	traj_list = []
	
	while True:
		# Read image
		status, im = cap.read()
		if not status:
			break
		if height > 1000:
			im_color = cv2.resize(im,(width/2, height/2))		
		else:
			im_color = np.copy(im)
		im_gray_5 = cv2.cvtColor(im_color, cv2.COLOR_BGR2GRAY)
		#im_draw = np.copy(im)
		tic = time.time()
		CMT.process_frame(im_color, im_prvs, frame)
		toc = time.time()

		# Display results

		# Draw updated estimate
		if CMT.has_result:
			
			cv2.line(im_draw, CMT.tl, CMT.tr, (255, 0, 0), 4)
			cv2.line(im_draw, CMT.tr, CMT.br, (255, 0, 0), 4)
			cv2.line(im_draw, CMT.br, CMT.bl, (255, 0, 0), 4)
			cv2.line(im_draw, CMT.bl, CMT.tl, (255, 0, 0), 4)
			
			if CMT.tl[0] < 0:
    			    CMT.tl = (0, CMT.tl[1])
			if CMT.tl[1] < 0:
    			    CMT.tl = (CMT.tl[0], 0)
			
			prof = im_color[CMT.tl[1] : CMT.br[1], CMT.tl[0] : CMT.br[0]]
			
			cv2.imshow("colorview", im_color)
			print prof.shape
			cv2.imshow("prof_view", prof)			

			prof_im = cv2.resize(prof, (100, 156), interpolation = cv2.INTER_CUBIC)
			
			video1.write(prof_im)			
			if (frame-1 == 0):		
			    reference_y = CMT.tl[0]					
			    reference_x = CMT.tl[1]
			    traj_list.append([frame -1, reference_x, reference_y])
			else:
			    traj_list.append([frame-1, reference_x - CMT.tl[1], reference_y - CMT.tl[0]])
			
		util.draw_keypoints(CMT.tracked_keypoints, im_draw, (255, 255, 255))
		# this is from simplescale
		util.draw_keypoints(CMT.votes[:, :2], im_draw)  # blue
		util.draw_keypoints(CMT.outliers[:, :2], im_draw, (0, 0, 255))

		if args.output is not None:
			# Original image
			cv2.imwrite('{0}/input_{1:08d}.png'.format(args.output, frame), im)
			# Output image
			cv2.imwrite('{0}/output_{1:08d}.png'.format(args.output, frame), im_draw)

			# Keypoints
			with open('{0}/keypoints_{1:08d}.csv'.format(args.output, frame), 'w') as f:
				f.write('x y\n')
				np.savetxt(f, CMT.tracked_keypoints[:, :2], fmt='%.2f')

			# Outlier
			with open('{0}/outliers_{1:08d}.csv'.format(args.output, frame), 'w') as f:
				f.write('x y\n')
				np.savetxt(f, CMT.outliers, fmt='%.2f')

			# Votes
			with open('{0}/votes_{1:08d}.csv'.format(args.output, frame), 'w') as f:
				f.write('x y\n')
				np.savetxt(f, CMT.votes, fmt='%.2f')

			# Bounding box
			with open('{0}/bbox_{1:08d}.csv'.format(args.output, frame), 'w') as f:
				f.write('x y\n')
				# Duplicate entry tl is not a mistake, as it is used as a drawing instruction
				np.savetxt(f, np.array((CMT.tl, CMT.tr, CMT.br, CMT.bl, CMT.tl)), fmt='%.2f')

		if not args.quiet:
			cv2.imshow('main', im_draw)
			# Check key input
			k = cv2.waitKey(pause_time)
			key = chr(k & 255)
			if key == 'q':
				break
			if key == 'd':
				import ipdb; ipdb.set_trace()

		# Remember image
		im_prev = im_gray

		# Advance frame number
		frame += 1

		print '{5:04d}: center: {0:.2f},{1:.2f} scale: {2:.2f}, active: {3:03d}, {4:04.0f}ms'.format(CMT.center[0], CMT.center[1], CMT.scale_estimate, CMT.active_keypoints.shape[0], 1000 * (toc - tic), frame)
	video1.release()
	text_file = open("raw_trajectory.txt","r")
	lines = text_file.readlines()
	text_file.close()
	trajectory = []
	for num in range(1, len(lines)):
    	    trajectory.append(eval(lines[num][:-1]))

	reference = eval(lines[0][:-1])

	reference_y = reference[2]
	reference_x = reference[1]
	
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

	prof_window_x = reference_x
	prof_window_y = reference_y

	cnt = 0
	out_cap = cv2.VideoCapture("cut_short2.mp4")
	while True:
	    outret, outframe = out_cap.read()
	    height, width, channels = outframe.shape
	    
	    if outret:
		out_frame = cv2.resize(outframe,(width/2, height/2))
		prof_out = out_frame[prof_window_x:prof_window_x + sh, prof_window_y:prof_window_y + sw]

		prof_im = cv2.resize(prof_out, (200, 256), interpolation = cv2.INTER_CUBIC)	
		video.write(prof_im)

		cv2.imshow("prof", prof_out)
		cv2.waitKey(100)
		#print smoothed_trajectory[cnt][1]
		prof_window_x = (reference_x - smoothed_trajectory[cnt][2])
		prof_window_y = (reference_y - smoothed_trajectory[cnt][1])
		cnt = cnt + 1
		print cnt
