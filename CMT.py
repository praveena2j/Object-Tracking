import cv2
import itertools
from numpy import array, zeros, vstack, hstack, math, nan, argsort, median, argmax, isnan, append
import scipy.cluster
import scipy.spatial
import time
import pybgs

import numpy as np
import util

class CMT(object):

	DETECTOR = 'BRISK'
	DESCRIPTOR = 'BRISK'
	DESC_LENGTH = 512
	MATCHER = 'BruteForce-Hamming'
	THR_OUTLIER = 20
	THR_CONF = 0.75
	THR_RATIO = 0.8

	estimate_scale = True
	estimate_rotation = True

	def initialise(self, im_gray, tl, br):

		tr = (br[0], tl[1])
		bl = (tl[0], br[1])

		min_x = min((tl[0], tr[0], br[0], bl[0]))
		min_y = min((tl[1], tr[1], br[1], bl[1]))
		max_x = max((tl[0], tr[0], br[0], bl[0]))
		max_y = max((tl[1], tr[1], br[1], bl[1]))

		reference_x = min_x + ((max_x - min_x)/2)
		reference_y = min_y + ((max_y - min_y)/2)

		# Initialise detector, descriptor, matcher
		self.detector = cv2.FeatureDetector_create(self.DETECTOR)
		self.descriptor = cv2.DescriptorExtractor_create(self.DESCRIPTOR)
		self.matcher = cv2.DescriptorMatcher_create(self.MATCHER)

		# Get initial keypoints in whole image
		im_gray0 = cv2.cvtColor(im_gray, cv2.COLOR_BGR2GRAY)
		keypoints_cv = self.detector.detect(im_gray0)

		# Remember keypoints that are in the rectangle as selected keypoints
		ind = util.in_rect(keypoints_cv, tl, br)
		selected_keypoints_cv = list(itertools.compress(keypoints_cv, ind))
		selected_keypoints_cv, self.selected_features = self.descriptor.compute(im_gray0, selected_keypoints_cv)
		selected_keypoints = util.keypoints_cv_to_np(selected_keypoints_cv)
		num_selected_keypoints = len(selected_keypoints_cv)

		if num_selected_keypoints == 0:
			raise Exception('No keypoints found in selection')

		# Remember keypoints that are not in the rectangle as background keypoints
		background_keypoints_cv = list(itertools.compress(keypoints_cv, ~ind))
		background_keypoints_cv, background_features = self.descriptor.compute(im_gray0, background_keypoints_cv)
		_ = util.keypoints_cv_to_np(background_keypoints_cv)

		# Assign each keypoint a class starting from 1, background is 0
		self.selected_classes = array(range(num_selected_keypoints)) + 1
		background_classes = zeros(len(background_keypoints_cv))

		# Stack background features and selected features into database
		self.features_database = vstack((background_features, self.selected_features))

		# Same for classes
		self.database_classes = hstack((background_classes, self.selected_classes))

		# Get all distances between selected keypoints in squareform
		pdist = scipy.spatial.distance.pdist(selected_keypoints)
		self.squareform = scipy.spatial.distance.squareform(pdist)

		# Get all angles between selected keypoints
		angles = np.empty((num_selected_keypoints, num_selected_keypoints))
		for k1, i1 in zip(selected_keypoints, range(num_selected_keypoints)):
			for k2, i2 in zip(selected_keypoints, range(num_selected_keypoints)):

				# Compute vector from k1 to k2
				v = k2 - k1

				# Compute angle of this vector with respect to x axis
				angle = math.atan2(v[1], v[0])

				# Store angle
				angles[i1, i2] = angle

		self.angles = angles

		# Find the center of selected keypoints
		center = np.mean(selected_keypoints, axis=0)

		# Remember the rectangle coordinates relative to the center
		self.center_to_tl = np.array(tl) - center
		self.center_to_tr = np.array([br[0], tl[1]]) - center
		self.center_to_br = np.array(br) - center
		self.center_to_bl = np.array([tl[0], br[1]]) - center

		# Calculate springs of each keypoint
		self.springs = selected_keypoints - center

		# Set start image for tracking
		self.im_prev = im_gray0

		# Make keypoints 'active' keypoints
		self.active_keypoints = np.copy(selected_keypoints)

		# Attach class information to active keypoints
		self.active_keypoints = hstack((selected_keypoints, self.selected_classes[:, None]))

		# Remember number of initial keypoints
		self.num_initial_keypoints = len(selected_keypoints_cv)

		i = 1
		self.term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
		
		self.trajectory = []

		self.traj_tuple = (0, reference_x, reference_y)
		self.trajectory.append(self.traj_tuple)

	def estimate(self, keypoints):

		center = array((nan, nan))
		scale_estimate = nan
		med_rot = nan

		# At least 2 keypoints are needed for scale
		if keypoints.size > 1:

			# Extract the keypoint classes
			keypoint_classes = keypoints[:, 2].squeeze().astype(np.int) 

			# Retain singular dimension
			if keypoint_classes.size == 1:
				keypoint_classes = keypoint_classes[None]

			# Sort
			ind_sort = argsort(keypoint_classes)
			keypoints = keypoints[ind_sort]
			keypoint_classes = keypoint_classes[ind_sort]

			# Get all combinations of keypoints
			all_combs = array([val for val in itertools.product(range(keypoints.shape[0]), repeat=2)])	

			# But exclude comparison with itself
			all_combs = all_combs[all_combs[:, 0] != all_combs[:, 1], :]

			# Measure distance between allcombs[0] and allcombs[1]
			ind1 = all_combs[:, 0] 
			ind2 = all_combs[:, 1]

			class_ind1 = keypoint_classes[ind1] - 1
			class_ind2 = keypoint_classes[ind2] - 1

			duplicate_classes = class_ind1 == class_ind2

			if not all(duplicate_classes):
				ind1 = ind1[~duplicate_classes]
				ind2 = ind2[~duplicate_classes]

				class_ind1 = class_ind1[~duplicate_classes]
				class_ind2 = class_ind2[~duplicate_classes]

				pts_allcombs0 = keypoints[ind1, :2]
				pts_allcombs1 = keypoints[ind2, :2]

				# This distance might be 0 for some combinations,
				# as it can happen that there is more than one keypoint at a single location
				dists = util.L2norm(pts_allcombs0 - pts_allcombs1)

				original_dists = self.squareform[class_ind1, class_ind2]

				scalechange = dists / original_dists

				# Compute angles
				angles = np.empty((pts_allcombs0.shape[0]))

				v = pts_allcombs1 - pts_allcombs0
				angles = np.arctan2(v[:, 1], v[:, 0])
				
				original_angles = self.angles[class_ind1, class_ind2]

				angle_diffs = angles - original_angles

				# Fix long way angles
				long_way_angles = np.abs(angle_diffs) > math.pi

				angle_diffs[long_way_angles] = angle_diffs[long_way_angles] - np.sign(angle_diffs[long_way_angles]) * 2 * math.pi

				scale_estimate = median(scalechange)
				if not self.estimate_scale:
					scale_estimate = 1;

				med_rot = median(angle_diffs)
				if not self.estimate_rotation:
					med_rot = 0;

				keypoint_class = keypoints[:, 2].astype(np.int)
				votes = keypoints[:, :2] - scale_estimate * (util.rotate(self.springs[keypoint_class - 1], med_rot))

				# Remember all votes including outliers
				self.votes = votes

				# Compute pairwise distance between votes
				pdist = scipy.spatial.distance.pdist(votes)

				# Compute linkage between pairwise distances
				linkage = scipy.cluster.hierarchy.linkage(pdist)

				# Perform hierarchical distance-based clustering
				T = scipy.cluster.hierarchy.fcluster(linkage, self.THR_OUTLIER, criterion='distance')

				# Count votes for each cluster
				cnt = np.bincount(T)  # Dummy 0 label remains
				
				# Get largest class
				Cmax = argmax(cnt)

				# Identify inliers (=members of largest class)
				inliers = T == Cmax
				# inliers = med_dists < THR_OUTLIER

				# Remember outliers
				self.outliers = keypoints[~inliers, :]

				# Stop tracking outliers
				keypoints = keypoints[inliers, :]

				# Remove outlier votes
				votes = votes[inliers, :]

				# Compute object center
				center = np.mean(votes, axis=0)

		return (center, scale_estimate, med_rot, keypoints)

	def process_frame(self, im_color, im_prvs, frame):
		params = { 
		 	'algorithm': 'grimson_gmm', 
		 	'low': 5.0 * 5.0,
 			'high': 5.0 * 5.0 * 2,
 			'alpha': 0.05,
 			'max_modes': 3 }

		bg_sub = pybgs.BackgroundSubtraction()	

		high_threshold_mask = np.zeros(shape=im_color.shape[0:2], dtype=np.uint8)
		low_threshold_mask = np.zeros_like(high_threshold_mask)
		bg_sub.init_model(im_color, params)

		im_gray = cv2.cvtColor(im_color, cv2.COLOR_BGR2GRAY)
		tracked_keypoints, _ = util.track(self.im_prev, im_gray, self.active_keypoints)
		(center, scale_estimate, rotation_estimate, tracked_keypoints) = self.estimate(tracked_keypoints)

		# Detect keypoints, compute descriptors
		keypoints_cv = self.detector.detect(im_gray) 
		keypoints_cv, features = self.descriptor.compute(im_gray, keypoints_cv)

		# Create list of active keypoints
		active_keypoints = zeros((0, 3)) 

		# Get the best two matches for each feature
		matches_all = self.matcher.knnMatch(features, self.features_database, 2)
		# Get all matches for selected features
		if not any(isnan(center)):
			selected_matches_all = self.matcher.knnMatch(features, self.selected_features, len(self.selected_features))

		# For each keypoint and its descriptor
		if len(keypoints_cv) > 0:
			transformed_springs = scale_estimate * util.rotate(self.springs, -rotation_estimate)
			for i in range(len(keypoints_cv)):

				# Retrieve keypoint location
				location = np.array(keypoints_cv[i].pt)

				# First: Match over whole image
				# Compute distances to all descriptors
				matches = matches_all[i]
				distances = np.array([m.distance for m in matches])

				# Convert distances to confidences, do not weight
				combined = 1 - distances / self.DESC_LENGTH

				classes = self.database_classes

				# Get best and second best index
				bestInd = matches[0].trainIdx
				secondBestInd = matches[1].trainIdx

				# Compute distance ratio according to Lowe
				ratio = (1 - combined[0]) / (1 - combined[1])

				# Extract class of best match
				keypoint_class = classes[bestInd]

				# If distance ratio is ok and absolute distance is ok and keypoint class is not background
				if ratio < self.THR_RATIO and combined[0] > self.THR_CONF and keypoint_class != 0:

					# Add keypoint to active keypoints
					new_kpt = append(location, keypoint_class)
					active_keypoints = append(active_keypoints, array([new_kpt]), axis=0)

				# In a second step, try to match difficult keypoints
				# If structural constraints are applicable
				if not any(isnan(center)):

					# Compute distances to initial descriptors
					matches = selected_matches_all[i]				
					distances = np.array([m.distance for m in matches])
					# Re-order the distances based on indexing
					idxs = np.argsort(np.array([m.trainIdx for m in matches]))
					distances = distances[idxs]					

					# Convert distances to confidences
					confidences = 1 - distances / self.DESC_LENGTH

					# Compute the keypoint location relative to the object center
					relative_location = location - center

					# Compute the distances to all springs
					displacements = util.L2norm(transformed_springs - relative_location)

					# For each spring, calculate weight
					weight = displacements < self.THR_OUTLIER  # Could be smooth function

					combined = weight * confidences

					classes = self.selected_classes

					# Sort in descending order
					sorted_conf = argsort(combined)[::-1]  # reverse

					# Get best and second best index
					bestInd = sorted_conf[0]
					secondBestInd = sorted_conf[1]

					# Compute distance ratio according to Lowe
					ratio = (1 - combined[bestInd]) / (1 - combined[secondBestInd])

					# Extract class of best match
					keypoint_class = classes[bestInd]

					# If distance ratio is ok and absolute distance is ok and keypoint class is not background
					if ratio < self.THR_RATIO and combined[bestInd] > self.THR_CONF and keypoint_class != 0:

						# Add keypoint to active keypoints
						new_kpt = append(location, keypoint_class)

						# Check whether same class already exists
						if active_keypoints.size > 0:
							same_class = np.nonzero(active_keypoints[:, 2] == keypoint_class)
							active_keypoints = np.delete(active_keypoints, same_class, axis=0)

						active_keypoints = append(active_keypoints, array([new_kpt]), axis=0)

		# If some keypoints have been tracked
		if tracked_keypoints.size > 0:

			# Extract the keypoint classes
			tracked_classes = tracked_keypoints[:, 2]

			# If there already are some active keypoints
			if active_keypoints.size > 0:

				# Add all tracked keypoints that have not been matched
				associated_classes = active_keypoints[:, 2]
				missing = ~np.in1d(tracked_classes, associated_classes)
				active_keypoints = append(active_keypoints, tracked_keypoints[missing, :], axis=0)

			# Else use all tracked keypoints
			else:
				active_keypoints = tracked_keypoints

		# Update object state estimate
		_ = active_keypoints
		self.center = center
		self.scale_estimate = scale_estimate
		self.rotation_estimate = rotation_estimate
		self.tracked_keypoints = tracked_keypoints
		self.active_keypoints = active_keypoints
		
		self.keypoints_cv = keypoints_cv
		_ = time.time()

		#self.tl = (nan, nan)
		#self.tr = (nan, nan)
		#self.br = (nan, nan)
		#self.bl = (nan, nan)

		#self.bb = array([nan, nan, nan, nan])

		self.has_result = False

		#reference_x = tr_r + (tr_w/2) + trackingzone_height 
		#reference_y = tr_c + (tr_h/2) + trackingzone_width 

		# cv2.imshow("prevroi", im_prvs)
		# cv2.imshow("prevroi", im_prvs)
		# cv2.imshow("nextroi", im_gray)

		# diffmask = cv2.absdiff(im_prvs, im_gray)
    		# ret,thresh = cv2.threshold(diffmask,25,255,cv2.THRESH_BINARY)
		# cv2.imshow("thresh", thresh)
		
		
		if not any(isnan(self.center)) and self.active_keypoints.shape[0] > self.num_initial_keypoints / 10:
			self.has_result = True

			tl = util.array_to_int_tuple(center + scale_estimate * util.rotate(self.center_to_tl[None, :], rotation_estimate).squeeze())
			tr = util.array_to_int_tuple(center + scale_estimate * util.rotate(self.center_to_tr[None, :], rotation_estimate).squeeze())
			br = util.array_to_int_tuple(center + scale_estimate * util.rotate(self.center_to_br[None, :], rotation_estimate).squeeze())
			bl = util.array_to_int_tuple(center + scale_estimate * util.rotate(self.center_to_bl[None, :], rotation_estimate).squeeze())

			min_x = min((tl[0], tr[0], br[0], bl[0]))
			min_y = min((tl[1], tr[1], br[1], bl[1]))
			max_x = max((tl[0], tr[0], br[0], bl[0]))
			max_y = max((tl[1], tr[1], br[1], bl[1]))

			self.tl = tl
			self.tr = tr
			self.bl = bl
			self.br = br

			self.bb = np.array([min_x, min_y, max_x - min_x, max_y - min_y])
		
		else:
			self.has_result = True
			# Tracking Window for meanshift
			print self.bb
			(r,c,w,h) = self.bb
			# Tracking zone for tracking

			trackingzone_height = r - 20
			trackingzone_height_ = trackingzone_height + w + 40
			trackingzone_width = c - 20
			trackingzone_width_ = trackingzone_width + h + 40

			size_next = im_gray.shape

			if trackingzone_width < 0:
    			    trackingzone_width = 0
			if trackingzone_height < 0:
    			    trackingzone_height = 0
			if trackingzone_width_ > size_next[0]:
    			    trackingzone_width_ = size_next[0]
    			    trackingzone_width = size_next[0] - (40+h)
			if trackingzone_height_ > size_next[1]:
    			    trackingzone_height_ = size_next[1]
    			    trackingzone_height = size_next[1] - (40+w)

			# Trackingwindow for meanshift
			tr_r = r - trackingzone_height
			tr_c = c - trackingzone_width
			tr_w = w
			tr_h = h
                        trackingwindow = (tr_r,tr_c,tr_w,tr_h)

			# Extracting consecutive frames for difference
			framegray = cv2.cvtColor(im_color, cv2.COLOR_BGR2GRAY)
			
			kernel = np.ones((3,3),np.uint8)
			bg_sub.subtract(frame, im_color, low_threshold_mask, high_threshold_mask)
			bg_sub.update(frame, im_color, high_threshold_mask)
			
			cv2.imshow('foreground', low_threshold_mask)
			cv2.waitKey(100)

        		blur = cv2.bilateralFilter(low_threshold_mask,9,100,100)

        		closing = cv2.morphologyEx(blur, cv2.MORPH_CLOSE, kernel)
    
        		mean_mask = closing[trackingzone_width : trackingzone_width_, trackingzone_height : trackingzone_height_]

			color_mask = im_color[trackingzone_width : trackingzone_width_, trackingzone_height : trackingzone_height_]
		
        		dilation = cv2.dilate(mean_mask,kernel,iterations = 2)
			im_floodfill = dilation.copy()
		
			mh, mw = dilation.shape[:2]
			mask = np.zeros((mh+2, mw+2), np.uint8)
			cv2.floodFill(im_floodfill, mask, (0,0), 255);
			im_floodfill_inv = cv2.bitwise_not(im_floodfill)

			meansh_mask = dilation | im_floodfill_inv

			meanshift_copy = meansh_mask.copy()
			cnts, hierarchy = cv2.findContours(meanshift_copy, 1,2)
			mask = np.ones(meansh_mask.shape[:2], dtype="uint8") * 255
			for c in cnts:
		    	    if (cv2.contourArea(c) < 10000):
		    	        cv2.drawContours(mask, [c], -1, 0, -1)
			meanshift_mask = cv2.bitwise_and(mask, meansh_mask, mask = None)
					
			track_x, track_y, track_w, track_h = trackingwindow
    
			img_ms = meanshift_mask[track_y : track_y + track_h, track_x : track_x + track_w]

			im_msk = color_mask[track_y : track_y + track_h, track_x : track_x + track_w]

			print meanshift_mask.shape
			print im_color.shape
			
			cv2.imshow('imgms', img_ms)			
			cv2.imshow("img_ms", im_msk)
			cv2.waitKey(10000)

			if (cv2.countNonZero(img_ms) < 0.4*(track_w*track_h)):
			    ms_track_window = trackingwindow
	
			else:
			    # Meanshift Tracking
			    ret, ms_track_window = cv2.meanShift(meanshift_mask, trackingwindow, self.term_crit)
				 
			# mean center of the mean shift window
			meancenter_x = ms_track_window[0] + (ms_track_window[2]/2)
			meancenter_y = ms_track_window[1] + (ms_track_window[3]/2)

			# Updation of the tracking zone
    			trackingzone_width = meancenter_y + trackingzone_width - ((40+h)/2)
    			trackingzone_height = meancenter_x + trackingzone_height - ((40+w)/2)
    			trackingzone_width_ = trackingzone_width + (40+h)
    			trackingzone_height_ = trackingzone_height + (40+w)
				
			if trackingzone_width < 0:
    			    trackingzone_width = 0
			if trackingzone_height < 0:
    			    trackingzone_height = 0
			if trackingzone_width_ > size_next[0]:
    			    trackingzone_width_ = size_next[0]
    			    trackingzone_width = size_next[0] - (40+h)
			if trackingzone_height_ > size_next[1]:
    			    trackingzone_height_ = size_next[1]
    			    trackingzone_height = size_next[1] - (40+w)

		        (sx, sy, sw, sh) = ms_track_window

			# Updation of the trackingwindow for meanshift
    			#trackingwindow = ms_track_window			
			#print trackingwindow
			self.tl = (int(sx + trackingzone_height), int(sy + trackingzone_width))
			self.tr = (int(sx + trackingzone_height + sw), int(sy + trackingzone_width))
			self.bl = (int(sx + trackingzone_height), int(sy + trackingzone_width + sh))
			self.br = (int(sx + trackingzone_height + sw), int(sy + trackingzone_width + sh))			

			self.bb = np.array([sx + trackingzone_height, sy + trackingzone_width, sw, sh])
			
			# print "final bounding box"
			# 	print self.bb

			# self.bb = np.array([min_x, min_y, max_x - min_x, max_y - min_y])
		self.im_prev = im_gray

		#print i, meancenter_x - ref_x, meancenter_y - ref_y 
		#traj_tuple = (i, meancenter_x + trackingzone_height - reference_x, meancenter_y + trackingzone_width - reference_y)
		#trajectory.append(traj_tuple)
	
		#with open ('trajectory.txt','w') as f:
		#    for traject in trajectory:
		#	system = traject
		#        f.write(str(system)+'\n')
		
