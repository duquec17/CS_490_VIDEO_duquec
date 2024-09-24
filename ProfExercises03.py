# MIT LICENSE
#
# Copyright 2024 Michael J. Reale
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
# to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

###############################################################################
# IMPORTS
###############################################################################

import sys
import numpy as np
import torch
import cv2
import pandas
import sklearn

def make_bounce_video(image_shape=(480,640,3),
                      frame_cnt=30,
                      start_pos=(100,300),
                      radius=40,
                      speed_factor=1):
    
    all_frames = []
    
    pos = np.array(start_pos)
    velocity = np.array([2,0])
    accel = np.array([0,1])
    
    velocity *= speed_factor
    accel *= speed_factor
    
    for i in range(frame_cnt):
        image = np.zeros(image_shape, dtype="uint8")
        cv2.circle(image, pos, radius, (0,0,255), -1)
        all_frames.append(image)
        
        pos += velocity
        velocity += accel
        
        if (pos[1] + radius) >= image_shape[0]:
            pos[1] = image_shape[0] - 1 - radius
            velocity[1] = -0.6*velocity[1]
    
    return all_frames  

'''
def check_track(image, window):
    sr = window[1]
    er = sr + window[3]
    #sc = 
    subimage = image[]       
'''   

###############################################################################
# MAIN
###############################################################################

def main():        
    ###############################################################################
    # PYTORCH
    ###############################################################################
    
    b = torch.rand(5,3)
    print(b)
    print("Torch CUDA?:", torch.cuda.is_available())
    
    ###############################################################################
    # PRINT OUT VERSIONS
    ###############################################################################

    print("Torch:", torch.__version__)
    print("Numpy:", np.__version__)
    print("OpenCV:", cv2.__version__)
    print("Pandas:", pandas.__version__)
    print("Scikit-Learn:", sklearn.__version__)
        
    ###############################################################################
    # OPENCV
    ###############################################################################
    if len(sys.argv) <= 1:
        # Webcam
        print("Opening webcam...")

        # Linux/Mac (or native Windows) with direct webcam connection
        capture = cv2.VideoCapture(0, cv2.CAP_DSHOW) # CAP_DSHOW recommended on Windows 
        # WSL: Use Yawcam to stream webcam on webserver
        # https://www.yawcam.com/download.php
        # Get local IP address and replace
        #IP_ADDRESS = "192.168.0.7"    
        #capture = cv2.VideoCapture("http://" + IP_ADDRESS + ":8081/video.mjpg")
        
        # Did we get it?
        if not capture.isOpened():
            print("ERROR: Cannot open capture!")
            exit(1)
            
        # Set window name
        windowName = "Webcam"
            
    else:
        # Trying to load video from argument

        # Get filename
        filename = sys.argv[1]
        
        # Load video
        capture = cv2.VideoCapture(filename)
        
        # Check if data is invalid
        if not capture.isOpened():
            print("ERROR: Could not open or find the video!")
            exit(1)

        # Set window name
        windowName = "Video"
        
    # Create window ahead of time
    cv2.namedWindow(windowName)
    
    # While not closed...
    key = -1
    while key == -1:
        # Get next frame from capture
        ret, frame = capture.read()
        
        if ret == True:        
            # Show the image
            cv2.imshow(windowName, frame)
        else:
            break

        # Wait 30 milliseconds, and grab any key presses
        key = cv2.waitKey(30)

    # Release the capture and destroy the window
    capture.release()
    cv2.destroyAllWindows()
        
    key = -1
    ESC_KEY = 27
    index = 0
    
    # OVERRIDE
    pos = (100,100)
    radius = 40
    frame_cnt = 60
    speed_factor = 20
    video_frames = make_bounce_video(start_pos=pos, 
                                     frame_cnt=frame_cnt,
                                     radius=radius,
                                     speed_factor=speed_factor)
    
    # x,y,width,height
    track_window = (pos[0]-radius, pos[1]-radius, 2*radius, 2*radius)
    
    criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 10, 1)
       
    while key != ESC_KEY:
        cur_frame = video_frames[index]
        
        input_image = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)
        ret, track_window = cv2.meanShift(input_image, track_window, criteria)
        
        track_image = np.copy(cur_frame)
        
        cv2.rectangle(track_image, track_window[0:2],
                      (track_window[0]+track_window[2], 
                       track_window[1]+track_window[3]),
                      (0,255,0), 2)
        
        cv2.imshow("ORIGINAL", cur_frame) 
        cv2.imshow("TRACKED", track_image)      
        key = cv2.waitKey(33)
                
        index += 1
        if index >= len(video_frames):
            index = 0
            track_window = (pos[0]-radius, pos[1]-radius, 2*radius, 2*radius)

    cv2.destroyAllWindows()

    # Close down...
    print("Closing application...")

if __name__ == "__main__": 
    main()
    # The end