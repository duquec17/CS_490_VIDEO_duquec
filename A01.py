## Information on Code
## Assignment 01 of CS 490 Video Processing and Vision
## Purpose: Open a video file, play it, and resave it as individual frames in the python language.
## Top-Down Approach
## Writer: Cristian Duque
## Based off material from Professor Reale

###############################################################################
# IMPORTS
###############################################################################

import sys
import numpy as np
import torch
import cv2
import pandas
import sklearn

def load_video_as_frames(video_filepath):
    
    # Load video 
    capture = cv2.VideoCapture(video_filepath)
    
    # Response to failed video load
    if not capture.isOpened():
        print("ERROR: Cannot open")
        return 
    
    return 

def compute_wait(fps):
    return

def display_frames(all_frames, title, fps=30):
    
    # Call to get wait time
    wait_time = compute_wait(fps)
    
    # Wait 30 milliseconds, and grab any key presses
    key = cv2.waitKey(wait_time)
    
    # Destory the window
    cv2.destroyAllWindows
    
    return

def save_frames(all_frames, output_dir, basename, fps=30):
    return 

###############################################################################
# MAIN
###############################################################################

def main():
    
   # 
   if len(sys.argv) < 3:
       print("ERROR: Insufficient Command Lines")
       exit(1)
   else:
        # Trying to load video from argument
        filename = sys.argv[1]
        
        # Close down
        print("Closing application...")
        
    
if __name__ == "__main__":
    main()
