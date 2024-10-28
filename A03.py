## Information on Code
## Assignment 03 of CS 490 Video Processing and Vision
## Purpose: Given a subset of videos from the Large-scale
## Single Object Tracking (LaSOT) dataset, successfully track
## the object bounding box accross the frames of the video
## Writer: Cristian Duque
## Based off material from Professor Reale & assistance from Eric Hoang 

###############################################################################
# IMPORTS
###############################################################################

import sys
import numpy as np
import cv2


def track_doggo(video_frames, first_box):
    
    # Initial bounding box setup (ymin, xmin, ymax, xmax)
    ymin, xmin, ymax, xmax = first_box
    box = (xmin, ymin, xmax - xmin, ymax - ymin)
    
    # Initialize tracking list with the first bounding box
    tracked_boxes = [first_box]
    
    # Initial model histogram from the first bounding box with H & S channels
    initial_frame = video_frames[0]
    hsv_frame = cv2.cvtColor(initial_frame, cv2.COLOR_BGR2HSV)
    object_region = hsv_frame[ymin:ymax, xmin:xmax]
    #object_region = object_region[10:-50, 40:-20]
    # kmean = color cluster 
    # take all pixels inside initial bounding box and take 50% box 
    # is most represented to create and apply mask
    # Simplified: Find color most part of the dog and segment it out into groups
    # First try: Cut the bounding box in half and find histogram 
    # Create heat map to scale based on what is dog (1), maybe dog(.5), and unsure (.1)
    
    cv2.imshow("DOG", object_region)
    cv2.waitKey(-1)
    model_hist = cv2.calcHist([object_region], [0, 1], None, [180, 256], [0, 180, 0, 256])
    #model_hist = cv2.calcHist([object_region], [0], None, [180], [0, 180])
    
    model_hist = cv2.normalize(model_hist, model_hist, 0, 255, cv2.NORM_MINMAX)
    
    # Initialize Kalman Filter with state for x, y, width, height and velocities
    kalman = cv2.KalmanFilter(6, 4)
    kalman.measurementMatrix = np.eye(4,6, dtype=np.float32)
    kalman.transitionMatrix = np.eye(6, dtype=np.float32)
    kalman.processNoiseCov = np.eye(6, dtype=np.float32) * 0.03  # Smaller to reduce sudden changes

    # Parameters to control maximum allowed growth in box dimensions
    max_scale_factor = 1.05  # Further reduced to prevent rapid growth

    # Update histogram dynamically every n frames
    n = 5
    
    # Track frames
    for frame_idx, frame in enumerate(video_frames[1:]):
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        back_proj = cv2.calcBackProject([hsv_frame], [0,1], model_hist, [0, 180, 0, 256], 1)
        #back_proj = cv2.calcBackProject([hsv_frame], [0], model_hist, [0, 180], 1)
        
        # Apply Gaussian blur to smooth back projection and reduce noise
        back_proj = cv2.GaussianBlur(back_proj, (5, 5), 0)
        
        # Add edge detection for refined contour support
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_frame, 100,200)
        #back_proj = cv2.bitwise_and(back_proj, back_proj, mask=edges)
        
        #print(np.amin(back_proj), np.amax(back_proj))
        
        #print(back_proj.shape)
        cv2.imshow("BACKPROJ", back_proj)
        cv2.waitKey(-1)
        
        # Apply CamShift to get the new location
        ret, box = cv2.CamShift(back_proj, box, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1))
        pts = cv2.boxPoints(ret)
        
        # Get bounding box parameters from CamShift results
        xmin, ymin, w, h = cv2.boundingRect(np.int0(pts))
        xmax, ymax = xmin + w, ymin + h

        # Ensure box stays within frame boundaries
        height, width = frame.shape[:2]
        xmin, ymin, xmax, ymax = max(0, xmin), max(0, ymin), min(width, xmax), min(height, ymax)

        '''
        # Limit size growth and maintain shape
        last_box = tracked_boxes[-1]
        last_w, last_h = last_box[3] - last_box[1], last_box[2] - last_box[0]
        
        if w > last_w * max_scale_factor or h > last_h * max_scale_factor:
            w = int(last_w * max_scale_factor)
            h = int(last_h * max_scale_factor)
            xmax, ymax = xmin + w, ymin + h
        '''
            
        
        # Kalman filter prediction and correction with bounding box center and size
        '''
        measurement = np.array([[np.float32(xmin + w / 2)], [np.float32(ymin + h / 2)], [np.float32(w)], [np.float32(h)]])
        kalman.correct(measurement)
        xmin = int(measurement[0] - w/2)
        ymin = int(measurement[1] - h/2)
        xmax = int(measurement[0] + w/2)
        ymax = int(measurement[1] + h/2
        '''

        '''
        # Update bounding box only if it's not significantly outside previous area
        if abs(xmin - last_box[1]) < width // 5 and abs(ymin - last_box[0]) < height // 5:
            tracked_boxes.append((ymin, xmin, ymax, xmax))
        else:
            # Reset to initial box if large drift detected
            tracked_boxes.append(first_box)
            box = (first_box[1], first_box[0], first_box[3] - first_box[1], first_box[2] - first_box[0])
        '''
        
        tracked_boxes.append((ymin, xmin, ymax, xmax))
                
        '''
        # Update the histogram periodically to adapt to color
        if frame_idx % n == 0:
            object_region = hsv_frame[ymin:ymax, xmin:xmax]
            model_hist = cv2.calcHist([object_region], [0,1], None, [180,256], [0,180,0,256])
            #model_hist = cv2.calcHist([object_region], [0], None, [180], [0,180])
            
            cv2.normalize(model_hist, model_hist, 0, 255, cv2.NORM_MINMAX)
        '''
    
    cv2.destroyAllWindows()
    return tracked_boxes