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
    # Initial list/tuple bounding box (ymin, xmin, ymax, xmax)
    ymin, xmin, ymax, xmax = first_box
    box = (xmin, ymin, xmax-xmin, ymax-ymin)
    
    # Initialize tracking list with the first bounding box
    tracked_boxes = [first_box]
    
    # Create initial model histogram from the first bounding box with H & S channels
    initial_frame = video_frames[0]
    hsv_frame = cv2.cvtColor(initial_frame,cv2.COLOR_BGR2HSV)
    object_region = hsv_frame[ymin:ymax, xmin:xmax]
    model_hist = cv2.calcHist([object_region], [0,1], None, [180,256], [0,180,0,256])
    cv2.normalize(model_hist, model_hist, 0, 255, cv2.NORM_MINMAX)
    
    # Create kernels for optical flow
    kfx = np.array([[1,0,-1], [2,0,-2], [1,0,-1]])
    kfy= np.array([[1,2,1], [0,0,0], [-1,-2,-1]])
    kft1 = np.zeros_like(kfx)
    kft2 = np.zeros_like(kfy)

    
    # Tracking each frame
    for frame in video_frames[1:]:
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        back_proj = cv2.calcBackProject([hsv_frame], [0,1], model_hist, [0, 180,0,256], 1)
        
        # Apply Gaussian blur to smooth back projection and reduce noise
        back_proj = cv2.GaussianBlur(back_proj, (5,5), 0)
        
        # Apply a mask to reduce background
        mask = np.zeros(back_proj.shape, dtype="uint8")
        mask[ymin:ymax, xmin:xmax] = 255
        back_proj = cv2.bitwise_and(back_proj, back_proj, mask=mask)
        
        # Apply CamShift to get the new location
        ret, box = cv2.CamShift(back_proj, box, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1,))
        pts = cv2.boxPoints(ret)
        
        # Calculate bounding box new constraints: object center
        xmin, ymin, w, h, = cv2.boundingRect(np.int0(pts))
        xmax, ymax = xmin+w, ymin+h
        
        # Ensure bounding box stays within frame boundaries
        height, width = frame.shape[:2]
        xmin, ymin, xmax, ymax = max(0, xmin), max(0, ymin), min(width, xmax), min(height, ymax)
        
        # Update bounding box and add to the list
        tracked_boxes.append((ymin, xmin, ymax, xmax))
    
    return tracked_boxes