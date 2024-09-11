## Information on Code
## Assignment 01 of CS 490 Video Processing and Vision
## Purpose: Open a video file, play it, and resave it as individual frames in the python language.
## Top-Down Approach
## Writer: Cristian Duque
## Based off material from Professor Reale

###############################################################################
# IMPORTS
###############################################################################

from pathlib import Path
import sys
import cv2
import os 
import shutil


###############################################################################
# Definitions of functions
###############################################################################

def load_video_as_frames(video_filepath):
    
    # Load video 
    capture = cv2.VideoCapture(video_filepath)
    
    # Response to failed video load
    if not capture.isOpened():
        print("ERROR: Cannot open")
        return 
    
    # Creation of all frames list
    all_frames = []
    
    # Loop through the video once and add each frame to list
    while True:
        ret, image = capture.read()
        if not ret:
            break
        all_frames.append(image)
    
    # Returns all frames to be later used 
    return all_frames

def compute_wait(fps):
    
    # Compute the wait in miliseconds as int
    computed_delay = int(1000.0/fps)
     
    # Returns computed delay to be later used
    return computed_delay

def display_frames(all_frames, title, fps=30):
    
    # Call to get wait time
    wait_time = compute_wait(fps)
    
    # Loop through frames once and display using cv2
    for frame in all_frames:
        cv2.imshow(title, frame) # Display
        key = cv2.waitKey(wait_time) # Wait
    
    # Destory the window
    cv2.destroyAllWindows()
    
    return

def save_frames(all_frames, output_dir, basename, fps=30):
    
    # Naming convention for video folder
    video_folder = basename + "_" + str(fps)
    
    # Make  the full output path
    output_path = os.path.join(output_dir, video_folder)
    
    # Remove already existing output paths
    if (os.path.exists(output_path)):
        shutil.rmtree(output_path)
       
    # Remake output path
    os.makedirs(output_path)
    
    # Get zero-padded filename, full path, and save the frame
    for index,frame in enumerate(all_frames):
        filename = "image_%07d.png" % index
        full_path = os.path.join(output_path, filename)
        cv2.imwrite(full_path, frame)
    
    return 

###############################################################################
# MAIN
###############################################################################

def main():
    
   # Check length of entered input and fail if less than 3
   if len(sys.argv) < 3:
       print("ERROR: Insufficient Command Lines")
       exit(1)
   else:
        # Read the input video path
        video_filepath = sys.argv[1]
        
        # Read the output directory
        output_dir = sys.argv[2]
        
        # Get the core filename from video path
        core_filename = Path(video_filepath).stem
        
        # Load all frames; if it returns none then output an error
        all_frames = load_video_as_frames(video_filepath)
        if all_frames is None:
            print("Error with video frames")
            exit(1)
        
        # Display all frames with title "input video" and fps 30
        display_frames(all_frames, "Input Video", fps=30)
        
        # Save the (output) frames to the output folder
        save_frames(all_frames, output_dir, core_filename, fps=30)
        
        # Close down
        print("Closing application...")
        
    
if __name__ == "__main__":
    main()
