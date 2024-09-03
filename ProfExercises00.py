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

###############################################################################
# MAIN
###############################################################################

def main(): 
    
    myimage = np.zeros((480, 640, 3), dtype="uint8")
    
    myimage[:240,...,0] = 255
        
    cv2.imshow("Test", myimage)
    cv2.waitKey(-1)
    cv2.destroyAllWindows()
           
           
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
        capture = cv2.VideoCapture(1) #, cv2.CAP_DSHOW) # CAP_DSHOW recommended on Windows 
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
    
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
    
    my_frames = []
    
    # While not closed...
    key = -1
    while key == -1:
        # Get next frame from capture
        ret, frame = capture.read()
        
        my_frames.append(frame)
        
        if ret == True:        
            # Show the image
            cv2.imshow(windowName, frame)
            
            frame_cnt = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_index = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
            
            if(frame_cnt != -1 and frame_cnt == frame_index):
                capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        else:
            break

        # Wait 30 milliseconds, and grab any key presses
        key = cv2.waitKey(30)
        
    my_video = np.array(my_frames)
    print("VIDEO:", my_video.shape)

    # Release the capture and destroy the window
    capture.release()
    cv2.destroyAllWindows()

    # Close down...
    print("Closing application...")

if __name__ == "__main__": 
    main()
    # The end