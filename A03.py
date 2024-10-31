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

# Function to find center of bounding box and pass new bounding box corner points
def shrink_box(ymax,ymin,xmax,xmin):

    # Calculate half size from the center
    original_height = ymax - ymin
    original_width = xmax - xmin

    # Calculate a center point for the original bounding box
    # Vertical center is offset by one-third of the height from ymin
    # Horizontal center is at the midpoint between xmin and xmax
    center_y = ymin + original_height // 3
    center_x = xmin + original_width // 2

    # Define the new bounding box size as a smaller region within the original
    # New height is one-third of the original height
    # New width is half of the original width
    new_height = original_height // 3
    new_width = original_width // 2

    # Calculate the new bounding box coordinates based on the center point
    # The region is centered around (center_x, center_y)
    ymin_new = center_y - new_height // 3
    ymax_new = center_y + new_height // 3
    xmin_new = center_x - new_width // 2
    xmax_new = center_x + new_width // 2
    
    # Return the coordinates of the smaller, centered bounding box
    return ymin_new, ymax_new, xmin_new,xmax_new


def track_doggo(video_frames, first_box):

    #1. Initial bounding box setup (ymin, xmin, ymax, xmax)
    ymin, xmin, ymax, xmax = first_box
    box = (xmin, ymin, xmax - xmin, ymax - ymin)

    #2. Initialize tracking list with the first bounding box
    tracked_boxes = [first_box]

    #3. Initial model histogram from the first bounding box with H & S channels
    initial_frame = video_frames[0]
    hsv_frame = cv2.cvtColor(initial_frame, cv2.COLOR_BGR2HSV)
    object_region = hsv_frame[ymin:ymax, xmin:xmax]

    #4. Assign new bounding box values
    ymin_new, ymax_new, xmin_new,xmax_new = shrink_box(ymax, ymin, xmax, xmin)

    # Extract the half-size object region
    object_region = hsv_frame[ymin_new:ymax_new, xmin_new:xmax_new]

    #object_region = object_region[10:-50, 40:-20]
    # kmean = color cluster
    # take all pixels inside initial bounding box and take 50% box
    # is most represented to create and apply mask
    # Simplified: Find color most part of the dog and segment it out into groups
    # First try: Cut the bounding box in half and find histogram
    # Create heat map to scale based on what is dog (1), maybe dog(0.5), and unsure (.1)

    cv2.imshow("DOG", object_region)
    cv2.waitKey(-1)
    model_hist = cv2.calcHist([object_region], [0, 1], None, [180, 256], [0, 180, 0, 256])

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

        # Apply Gaussian blur to smooth back projection and reduce noise
        back_proj = cv2.GaussianBlur(back_proj, (5, 5), 0)

        #print(back_proj.shape)
        cv2.imshow("BACKPROJ", back_proj)
        cv2.waitKey(-1)

        # Apply CamShift to get the new location
        ret, box = cv2.CamShift(back_proj, box, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1))
        pts = cv2.boxPoints(ret)

        # Get bounding box parameters from CamShift results
        xmin, ymin, w, h = cv2.boundingRect(np.int0(pts))
        # Scale bounding box by 2 to increase detection chance
        xmax, ymax = xmin + w * 12//10 , ymin + h * 18//10

        # Ensure box stays within frame boundaries
        height, width = frame.shape[:2]
        xmin, ymin, xmax, ymax = max(0, xmin), max(0, ymin), min(width, xmax), min(height, ymax)

        # Add current version of box to list to later calculate
        tracked_boxes.append((ymin, xmin, ymax, xmax))

    cv2.destroyAllWindows()
    return tracked_boxes