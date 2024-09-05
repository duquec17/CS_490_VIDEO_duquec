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

def burn_eyes(image):
    output = np.copy(image)
    output = cv2.resize(output, dsize=(0,0), fx=0.1, fy=0.1)
    output = cv2.resize(output, dsize=(0,0), fx=10.0, fy=10.0, 
                        interpolation=cv2.INTER_NEAREST)
    return output

def blur_across_buffer(frame_buffer, time_index, frame):
    frame = frame.astype(np.float64)
    frame /= 255.0
    
    frame_buffer[time_index] = frame
    ave_image = np.mean(frame_buffer, axis=0)
    
    time_index += 1
    time_index %= len(frame_buffer) # frame_buffer.shape[0]
    
    return frame_buffer, time_index, ave_image
    

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
    
    frame_buffer = None
    time_index = 0
    
    # While not closed...
    key = -1
    while key == -1:
        # Get next frame from capture
        ret, frame = capture.read()
        
        my_frames.append(frame)
        
        if ret == True:
            if frame_buffer is None:
                frame_cnt = 10
                video_shape = (10,) + frame.shape
                frame_buffer = np.zeros(video_shape, dtype=np.float64)
                    
            # Show the image
            cv2.imshow(windowName, frame)
            
            proc_frame = burn_eyes(frame)
            
            cv2.imshow("UNSPEAKABLE HORRORS", proc_frame)
            
            frame_buffer, time_index, ave_image = blur_across_buffer(frame_buffer,
                                                                     time_index,
                                                                     frame)
            
            cv2.imshow("AVERAGE", ave_image)
            
            fimage = frame.astype(np.float64)/255.0
            fimage = np.absolute(fimage - ave_image)
            
            cv2.imshow("GHOST", fimage)
            
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