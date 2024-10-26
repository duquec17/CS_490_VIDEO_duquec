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
import torch
import cv2
import pandas
import sklearn

def track_doggo(video_frames, first_box):
    # Initial list/tuple bounding box (ymin, xmin, ymax, xmax)
    ymin, xmin, ymax, xmax = box
    box = (xmin, ymin, xmax-xmin, ymax-ymin)
    
    # Initialize tracking list with the first bounding box
    tracked_boxes = [first_box]
    
    # Create initial model histogram from the first bounding box
    initial_frame = video_frames[0]
    hsv_frame = cv2.cvtColor(initial_frame,cv2.COLOR_BGR2HSV)
    object_region = hsv_frame[ymin:ymax, xmin:xmax]
    model_hist = cv2.calcHist([object_region], [0], None, [180], [0,180])
    cv2.normalize(model_hist, model_hist, 0, 255, cv2.NORM_MINMAX)
    
    # Tracking each frame
    for frame in video_frames[1:]:
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        back_proj = cv2.calcBackProject([hsv_frame], [0], model_hist, [0, 180], 1)
        
        # Apply CamShift to get the new location
        ret, box = cv2.CamShift(back_proj, box, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1,))
        x,y,w,h = box
        
        # Update bounding box and add to the list
        updated_box = (int(y), int(x), int(y+h), int(x+w))
        tracked_boxes.append(updated_box)
    
    return