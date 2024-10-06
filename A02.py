## Information on Code
## Assignment 02 of CS 490 Video Processing and Vision
## Purpose: Compute optical flow. Works in tandem with A01.py
## Top-Down Approach
## Writer: Cristian Duque
## Based off material from Professor Reale

###############################################################################
# IMPORTS
###############################################################################

from pathlib import Path
import sys
import cv2
import numpy as np
import os 
import shutil

###############################################################################
# Additional enum
###############################################################################

from enum import Enum

import A01
class OPTICAL_FLOW(Enum):
    HORN_SHUNCK="horn_shunck"
    LUCAS_KANADE="lucas_kanade"
    
###############################################################################
# Definitions of functions
###############################################################################


def compute_video_derivatives(video_frames, size):
    # Check size of video frame enter branch based on size
    
    if size == 2:
        # Applies following filter for size of 2
        kfx = np.array([[-1,1],
                        [-1,1]], dtype="float64")
        kfy = np.array([[-1,-1],
                        [1,1]], dtype="float64")
        kft1 = np.array([[-1,-1],
                         [-1,-1]], dtype="float64")
        kft2 = np.array([[1,1],
                         [1,1]], dtype="float64")
    elif size == 3:
        # Applies following filter for size of 3
        kfx = np.array([[-1,0,1],
                        [-2,0,2],
                        [-1,0,1]], dtype="float64")
        kfy = np.array([[-1,-2,-1],
                        [0,0,0],
                        [1,2,1]], dtype="float64")
        kft1 = np.array([[-1,-2,-1],
                         [-2,-4,-2],
                         [-1,-2,-1]], dtype="float64")
        kft2 = np.array([[1,2,1],
                         [2,4,2],
                         [1,2,1]], dtype="float64")
    else:
        return None
    
    # Lists that hold all results
    all_fx = []
    all_fy = []
    all_ft = []
    
    # Previous frame tracker
    prev_frame = None
    
    # Loop through each pair of frames in video
    for frame in video_frames:
        # Convert the iamge to a grayscale, float64 image with range [0, 1]
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = gray_frame.astype(np.float64) / 255.0
        
        # If previous frame is not set, set it to the current frame (repeating the first frame)
        if prev_frame is None:
            prev_frame = gray_frame.copy() 
            
        # Apply Filters
        fx = cv2.filter2D(gray_frame, -1, kfx)
        fy = cv2.filter2D(gray_frame, -1, kfy)
        ft1 = cv2.filter2D(prev_frame, -1, kft1)
        ft2 = cv2.filter2D(gray_frame, -1, kft2)
        
        # Calculate ft through difference
        ft = ft1 - ft2
        
        # Scale the results
        if size == 2:
            fx /= 4.0
            fy /= 4.0
            ft /= 4.0
        elif size == 3:
            fx /= 8.0
            fy /= 8.0
            ft /= 16.0
            
        # Append results to lists
        all_fx.append(fx)
        all_fy.append(fy)
        all_ft.append(ft)
        
        # Set previous frame to current frame
        prev_frame = gray_frame.copy()
    
    # Return three lists
    return all_fx, all_fy, all_ft

def compute_one_optical_flow_horn_shunck(fx, fy, ft, max_iter, max_error, weight=1.0):
    return

def compute_optical_flow(video_frames, method=OPTICAL_FLOW.HORN_SHUNCK, max_iter=10, max_error=1e-4, horn_weight=1.0, kanade_win_size=10):
    return

###############################################################################
# Main function for debugging purposes
###############################################################################

def main():
    # Load video frames
    video_filepath = "assign02/input/simple/image_%07d.png"
    #video_filepath = "assign02/input/noice/image_%07d.png"
    video_frames = A01.load_video_as_frames(video_filepath)
    
    # Check if data is invalid
    if video_frames is None:
        print("ERROR: Could not open or find the video!")
        exit(1)
    
    # OPTIONAL: Only grab the first five frames
    video_frames = video_frames[0:5]
    
    # Calculate optical flow
    flow_frames = compute_optical_flow(video_frames, method=OPTICAL_FLOW.HORN_SHUNCK)

    # While not closed...
    key = -1
    ESC_KEY = 27
    SPACE_KEY = 32
    index = 0

    while key != ESC_KEY:
        # Get the current image and flow image
        image = video_frames[index]
        flow = flow_frames[index]
        
        flow = np.absolute(flow)

        # Show the images
        cv2.imshow("ORIGINAL", image)
        cv2.imshow("FLOW", flow)

        # Wait 30 milliseconds, and grab any key presses
        key = cv2.waitKey(30)
        
        # If space, move forward
        if key == SPACE_KEY:
            index += 1
            if index >= len(video_frames):
                index = 0
    
    # Destroy the windows
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()