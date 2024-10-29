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

def filterNeighborhood2D(image, kernel, crow, ccol):
    halfH = kernel.shape[0]//2
    halfW = kernel.shape[1]//2
    
    startOffH = (1 - kernel.shape[0]%2)
    startOffW = (1 - kernel.shape[1]%2)
    
    endRow = crow + halfH
    endCol = ccol + halfW
    
    startRow = crow - halfH + startOffH
    startCol = ccol - halfW + startOffW
    
    clamp_startRow = max(0, startRow)
    clamp_startCol = max(0, startCol)
    neighborhood = image[clamp_startRow:(endRow+1), clamp_startCol:(endCol+1)]
    
    if startRow < 0:
        kernel = kernel[-startRow:]
    elif endRow > (image.shape[0]-1):
        off = image.shape[0] - 1 - endRow
        kernel = kernel[0:(kernel.shape[0]+off)]
            
    if startCol < 0:
        kernel = kernel[:, -startCol:]
    elif endCol > (image.shape[1]-1):
        off = image.shape[1] - 1 - endCol
        kernel = kernel[:, 0:(kernel.shape[1]+off)]
        
    #print("NEIGHBORHOOD:", neighborhood.shape)
    #print("KERNEL:", kernel.shape) 
    
    value = kernel * neighborhood
    value = np.sum(value)  
    
    return value

def filterNeighborhood3D(video, kernel, ctime, crow, ccol):
    halfT = kernel.shape[0]//2
    halfH = kernel.shape[1]//2
    halfW = kernel.shape[2]//2
    
    startOffT = (1 - kernel.shape[0]%2)
    startOffH = (1 - kernel.shape[1]%2)
    startOffW = (1 - kernel.shape[2]%2)
    
    endTime = ctime + halfT
    endRow = crow + halfH
    endCol = ccol + halfW
    
    startTime = ctime - halfT + startOffT
    startRow = crow - halfH + startOffH
    startCol = ccol - halfW + startOffW
    
    clamp_startTime = max(0, startTime)
    clamp_startRow = max(0, startRow)
    clamp_startCol = max(0, startCol)
    neighborhood = video[   clamp_startTime:(endTime+1),
                            clamp_startRow:(endRow+1), 
                            clamp_startCol:(endCol+1)]
    
    def get_bounds(start, end, max_image_size, max_kernel_size):
        if start < 0:
            s = -start
            e = max_kernel_size
        elif end > (max_image_size-1):
            off = max_image_size - 1 - end
            s = 0
            e = max_kernel_size + off
        else:
            s = 0
            e = max_kernel_size
            
        return s,e
    
    st, et = get_bounds(startTime, endTime, video.shape[0], kernel.shape[0])
    sr, er = get_bounds(startRow, endRow, video.shape[1], kernel.shape[1])
    sc, ec = get_bounds(startCol, endCol, video.shape[2], kernel.shape[2])
       
    kernel = kernel[st:et, sr:er, sc:ec]
      
    print("NEIGHBORHOOD:", neighborhood.shape)
    print("KERNEL:", kernel.shape) 
    
    value = kernel * neighborhood
    value = np.sum(value)  
    
    return value

def filter2D(image, kernel):
    output = np.copy(image)
    
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            output[row,col] = filterNeighborhood2D(image, kernel, row, col)
            
    return output 

def filter3D(video, kernel):
    output = np.copy(video)
    
    for t in range(video.shape[0]):
        for row in range(video.shape[1]):
            for col in range(video.shape[2]):
                output[t,row,col] = filterNeighborhood3D(video, kernel, t, row, col)           

    return output

def compute_one_optical_flow_horn_shunck(prev_frame, cur_frame,
                                        kfx, kfy, kft1, kft2,
                                        max_iter=20):
    
    fx = (cv2.filter2D(prev_frame, cv2.CV_64F, kfx) 
          + cv2.filter2D(cur_frame, cv2.CV_64F, kfx))
    
    fy = (cv2.filter2D(prev_frame, cv2.CV_64F, kfy) 
          + cv2.filter2D(cur_frame, cv2.CV_64F, kfy))
    
    ft = (cv2.filter2D(prev_frame, cv2.CV_64F, kft1) 
          + cv2.filter2D(cur_frame, cv2.CV_64F, kft2))
    
    fx /= 4.0
    fy /= 4.0
    ft /= 4.0
    
    u = np.zeros(fx.shape, dtype="float64")
    v = np.zeros(fx.shape, dtype="float64")
    
    lap_filter = np.array([[0, 0.25, 0],
                          [0.25,0,0.25],
                          [0,0.25,0]], dtype="float64")
    
    converged = False
    iter_cnt = 0
    lamb = 0.1
    print_inc = 5
    
    while not converged:
        # MAGIC
        uav = cv2.filter2D(u, cv2.CV_64F, lap_filter)
        vav = cv2.filter2D(v, cv2.CV_64F, lap_filter)
        
        P = fx*uav + fy*vav + ft
        D = lamb + fx*fx + fy*fy
        
        PD = P/D
        
        u = uav - fx*PD
        v = vav - fy*PD
                
        iter_cnt += 1
        
        if iter_cnt % print_inc == 0:
            print("ITERATION", iter_cnt, "DONE...")
                
        if iter_cnt >= max_iter:
            converged = True
            
    
    extra = np.zeros_like(u)
    combo = np.stack([u,v,extra], axis=-1)

    
    return combo  

def convert_to_hsv_flow(flow):   
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype="uint8")
    #print("MIN MAX:", np.amin(ang), np.amax(ang))
    
    hsv[...,1] = 255
    hsv[...,0] = cv2.normalize(ang, None, 0, 255, cv2.NORM_MINMAX) # 255*ang/(2.0*np.pi) # ang*180.0/np.pi
    hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return flow
          
    
def compute_optical_flow_horn_shunck(video_frames, kfx, kfy, kft1, kft2,
                                     max_iter=20):
    all_flow = []
    prev_frame = None
    #index = 0
        
    for index, frame in enumerate(video_frames):
        print("** FRAME", index, "************************")
        if prev_frame is None:
            prev_frame = frame
            
        flow = compute_one_optical_flow_horn_shunck(prev_frame, frame,
                                                    kfx, kfy, kft1, kft2,
                                                    max_iter=max_iter)
        
        
        #flow = convert_to_hsv_flow(flow)
        flow *= 10
        
        all_flow.append(flow)
        prev_frame = frame
        #index += 1
        
    return all_flow

def compute_optical_flow_farneback(video_frames):
    all_flow = []
    prev_frame = None
    #index = 0
        
    for index, frame in enumerate(video_frames):
        print("** FRAME", index, "************************")
        if prev_frame is None:
            prev_frame = frame
            
        flow = cv2.calcOpticalFlowFarneback(prev_frame, frame,
                                            None, 0.5, 3, 
                                            winsize=31, #15,
                                            iterations=3,
                                            poly_n=5, 
                                            poly_sigma=1.2, 
                                            flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
                
        #flow = convert_to_hsv_flow(flow)
        extra = np.zeros_like(flow[:,:,0])
        flow = np.stack([flow[:,:,0],flow[:,:,1],extra], axis=-1)
        flow *= 10000
        #print(flow.shape)
        #print(flow.dtype)
        print(np.amin(flow), np.amax(flow))
                
        all_flow.append(flow)
        prev_frame = frame
        #index += 1
        
    return all_flow


def make_test_video(image_size=(480,640), frame_cnt=30, inc_x=5, inc_y=0):
    video_frames = []
    
    start_pos = [100,100]
    end_pos = [200,200]
    
    for index in range(frame_cnt):
        frame = np.zeros(image_size, dtype="float64")
        cv2.rectangle(frame, start_pos, end_pos, (1.0,), -1)
        video_frames.append(frame)
        start_pos[0] += inc_x
        end_pos[0] += inc_x
        
        start_pos[1] += inc_y
        end_pos[1] += inc_y
        
    return video_frames  

def detect_motion(flow_frames):
    
    motion_frames = []
    for flow in flow_frames:
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        motion_frames.append(mag)
    
    return motion_frames  

def get_block_averages(all_frames, win_size):
    block_images = []
    
    for frame in all_frames:
        one_block_image = np.zeros_like(frame)
        for row in range(0, frame.shape[0], win_size):
            for col in range(0, frame.shape[1], win_size):
                subimage = frame[row:(row+win_size), col:(col+win_size)]
                ave_val = np.mean(subimage)
                one_block_image[row:(row+win_size), col:(col+win_size)] = ave_val
        block_images.append(one_block_image)
        
    return block_images

def get_bound_box_image(image, box):
    # (ymin, xmin, ymax, xmax)
    ymin, xmin, ymax, xmax = box
    subimage = image[ymin:ymax, xmin:xmax, :]
    return subimage

def get_color_similarity(image, target):
    image = image - target
    image = image*image
    image = np.sum(image, axis=2, keepdims=True)
    image = np.sqrt(image)
    image /= 255.0
    image = 1.0 - image
    return image

def get_hue_mask(hsv):
    return cv2.inRange(hsv, (0.0, 60.0, 32.0), (180.0, 255.0, 255.0))

def get_model_hue_histogram(subimage):    
    hsv = cv2.cvtColor(subimage, cv2.COLOR_BGR2HSV)
    mask = get_hue_mask(hsv)
    hist = cv2.calcHist([hsv], [0], mask, [180], [0,180])
    cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
    return hist

def cluster_colors(image):
    samples = image.astype("float32")
    samples = np.reshape(samples, [-1, 3])
    # https://docs.opencv.org/4.x/d1/d5c/tutorial_py_kmeans_opencv.html
    ret, labels, centers = cv2.kmeans(samples, 4, None, 
                                      (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
                                      10,cv2.KMEANS_RANDOM_CENTERS)
    print(labels)
    print(centers)
    recolor = centers[labels.flatten()]
    recolor = np.reshape(recolor, image.shape)
    recolor /= 255.0
    return recolor

###############################################################################
# MAIN
###############################################################################

def main(): 
    
    dummy_video = np.zeros((4,4,4), dtype="float64")
    dummy_filter = np.zeros((2,2,2), dtype="float64")
    dummy_output = filter3D(dummy_video, dummy_filter)
           
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
    
    # While not closed...
    key = -1
    prev_frame = None
    
    kfx = np.array([[-1, 1],
                    [-1, 1]], dtype="float64")
    kfy = np.array([[-1,-1],
                    [1,1]], dtype="float64")
    kft1 = np.array([[-1,-1],
                     [-1,-1]], dtype="float64")
    kft2 = np.array([[1,1],
                     [1,1]], dtype="float64")
    
    video_frames = []   
    
    box = (210, 150, 310, 230) 
    
    while key == -1:
        # Get next frame from capture
        ret, frame = capture.read()
        
        if ret == True:        
            # Show the image
            
            frame = cluster_colors(frame)
            cv2.imshow(windowName, frame)
            
            
            subimage = get_bound_box_image(frame, box)
            cv2.imshow("ITEM", subimage)
            model_hist = get_model_hue_histogram(subimage)
            
            ave_color = np.mean(subimage, axis=(0,1))
            print("AVE:", ave_color)
            
            color_heat = get_color_similarity(frame, ave_color)
            cv2.imshow("COLOR", color_heat)
            
            hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = get_hue_mask(hsv_image)
            back_image = cv2.calcBackProject([hsv_image], [0], model_hist, [0,180],1)
            back_image = back_image.astype("float64")
            mask = mask.astype("float64")
            mask /= 255.0
            back_image *= mask
            back_image /= 255.0
            
            print(np.amax(back_image))
            cv2.imshow("BACK PROJECT", back_image) #*255)
            
            gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype("float64")
            gray_image /= 255.0
            
            kernel_size = 17
            gray_image = cv2.GaussianBlur(gray_image, 
                                          ksize=(kernel_size, kernel_size),
                                          sigmaX=0)
            
            cv2.imshow("GRAY", gray_image)
                        
            video_frames.append(gray_image)
                        
            prev_frame = np.copy(gray_image)              
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
    video_frames = make_test_video(inc_x=0, inc_y=5)
          
    flow_frames = compute_optical_flow_horn_shunck(video_frames, 
                                                    kfx, kfy,
                                                    kft1, kft2)
    
    #flow_frames = detect_motion(flow_frames)
    #flow_frames = get_block_averages(flow_frames, win_size=30)
    
    #flow_frames = compute_optical_flow_farneback(video_frames)
        
    while key != ESC_KEY:
        cur_frame = video_frames[index]
        flow_frame = np.absolute(flow_frames[index])
        
        subimage = get_bound_box_image(flow_frames[index], box)
        cv2.imshow("ITEM", subimage)
        
        ave_flow = np.mean(subimage, axis=(0,1))
        print("AVE:", ave_flow)
                    
        cv2.imshow("ORIGINAL", cur_frame)      
        cv2.imshow("FLOW", flow_frame)  
        key = cv2.waitKey(33)
                
        index += 1
        if index >= len(video_frames):
            index = 0

    cv2.destroyAllWindows()

    # Close down...
    print("Closing application...")

if __name__ == "__main__": 
    main()
    # The end