import unittest
from unittest.mock import patch
import shutil
from pathlib import Path

import sys
import os
import subprocess as sub
import cv2
import numpy as np
import pandas as pd
import General_Testing as GT
import A02
from Test_A02 import *

class Test_A02_Grad(Test_A02):  
    ###########################################################################
    # TEST: compute_one_optical_flow_lucas_kanade
    ###########################################################################
    def test_compute_one_optical_flow_lucas_kanade(self):        
        # For each video...
        for index in range(len(video_files)):
            # Load the video file as individual frames
            video_frames = A01.load_video_as_frames(os.path.join(input_dir, video_files[index]))
            
            # Try two different combinations of window sizes
            window_sizes = [19, 41]
            
            for win_size in window_sizes:                            
                # For each frame pair...
                for frame_index in range(len(video_frames)):                
                    test_name = video_folders[index] + "_" + get_base_lucas_kanade_name(win_size, frame_index)                    
                    with self.subTest(test_name):
                        # Read in derivatives
                        size = 3
                        fx = np.load(os.path.join(ground_dir, video_folders[index],
                                                get_base_derivative_name("fx", size, frame_index) + ".npy")) 
                                    
                        fy = np.load(os.path.join(ground_dir, video_folders[index], 
                                                get_base_derivative_name("fy", size, frame_index) + ".npy"))
                        
                        ft = np.load(os.path.join(ground_dir, video_folders[index], 
                                                get_base_derivative_name("ft", size, frame_index) + ".npy"))
                        
                        # Load ground data for horn
                        ground_data = np.load(os.path.join(ground_dir, video_folders[index], 
                                            get_base_lucas_kanade_name(win_size, frame_index) + ".npy"))
                                            
                        # Compute...
                        pred_data = A02.compute_one_optical_flow_lucas_kanade(fx, fy, ft, win_size=win_size)
                                               
                        # Test
                        GT.check_for_unequal_no_diff("Failed on frame", 
                                test_name, 
                                pred_data, ground_data)
                                                                
    ###########################################################################
    # TEST: compute_optical_flow
    ###########################################################################
    def test_compute_optical_flow_grad(self):
        # For each video...
        for index in range(len(video_files)):
            # Load the video file as individual frames
            video_frames = A01.load_video_as_frames(os.path.join(input_dir, video_files[index]))
            
            # Try two different combinations of window sizes
            window_sizes = [19, 41]
            
            for win_size in window_sizes:          
                # Compute the flow
                flow_frames = A02.compute_optical_flow(video_frames, A02.OPTICAL_FLOW.LUCAS_KANADE, 
                                                       kanade_win_size=win_size)
                
                # For each frame pair...
                for frame_index in range(len(video_frames)):
                    test_name = video_folders[index] + "_" + get_base_lucas_kanade_name(win_size, frame_index)     
                    with self.subTest(test_name):                    
                        # Load ground data for horn
                        ground_data = np.load(os.path.join(ground_dir, video_folders[index], 
                                            get_base_lucas_kanade_name(win_size, frame_index) + ".npy"))
                        
                        # Test
                        GT.check_for_unequal_no_diff("Failed on frame", 
                                test_name, 
                                flow_frames[frame_index], ground_data)    
            
            
def main():
    runner = unittest.TextTestRunner()
    test_cases = unittest.TestLoader().loadTestsFromTestCase(Test_A02_Grad)    
    runner.run(test_cases)

if __name__ == '__main__':    
    main()
