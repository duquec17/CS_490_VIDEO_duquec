import unittest
from unittest.mock import patch
import io
import shutil
import multiprocessing
from pathlib import Path
from threading import Thread
from time import sleep, perf_counter
import sys
import os
import subprocess as sub
import cv2
import numpy as np
import pandas as pd
import General_Testing as GT
import A02
import A01

base_dir = "assign02"
input_dir = base_dir + "/" + "input"
ground_dir = "../../Data/assign02/ground"
output_dir = base_dir + "/" + "output"

video_files = [
    "simple/image_%07d.png",
    "noice/image_%07d.png"
]

video_folders = [
    "simple",
    "noice"
]


def get_base_horn_shunck_name(itercnt, weight, index):
    return "horn_i%03d_w%0.2f_%07d" % (itercnt, weight, index)
  
def get_base_lucas_kanade_name(win_size, index):
    return "lucas_w%02d_%07d" % (win_size, index)

def get_base_derivative_name(prefix, size, index):
    return "%s_s%02d_%07d" % (prefix, size, index)

class Test_A02(unittest.TestCase):  
      
    ###########################################################################
    # TEST: compute_video_derivatives
    ###########################################################################
    def test_compute_video_derivatives(self):
        # For each video...
        for index in range(len(video_files)):
            # Load the video file as individual frames
            video_frames = A01.load_video_as_frames(os.path.join(input_dir, video_files[index]))
            
            # Trivially try with bad size
            self.assertEqual(A02.compute_video_derivatives(video_frames, size=1), None, "Bad size (1) should return None.")
            self.assertEqual(A02.compute_video_derivatives(video_frames, size=4), None, "Bad size (4) should return None.")
            
            # For each REAL size
            for size in [2,3]:
                # Compute derivatives            
                video_fx, video_fy, video_ft = A02.compute_video_derivatives(video_frames, size=size)
                derivs = {
                    "fx": video_fx,
                    "fy": video_fy,
                    "ft": video_ft
                }
                
                # Loop through and check which ones match
                for prefix in ["fx", "fy", "ft"]:
                    for frame_index in range(len(video_frames)):
                        test_name = "%s_%s_s%d_%04d" % (video_folders[index], prefix, size, frame_index)
                        with self.subTest(test_name):
                            # Load ground data
                            ground_path = os.path.join(ground_dir, video_folders[index], 
                                                       get_base_derivative_name(prefix, size, frame_index) + ".npy")                                                       
                            ground_data = np.load(ground_path) 
                            
                            GT.check_for_unequal_no_diff("Failed on frame", 
                                test_name, 
                                derivs[prefix][frame_index], ground_data) 
                            
    ###########################################################################
    # TEST: compute_one_optical_flow_horn_shunck
    ###########################################################################
    def test_compute_one_optical_flow_horn_shunck(self):        
        # For each video...
        for index in range(len(video_files)):
            # Load the video file as individual frames
            video_frames = A01.load_video_as_frames(os.path.join(input_dir, video_files[index]))
            
            # Try two different combinations of iterations and weights
            iter_weight_combos = [[15, 1.0], [5, 0.1]]
            
            for iter_cnt, weight in iter_weight_combos:                            
                # For each frame pair...
                for frame_index in range(len(video_frames)):                
                    test_name = video_folders[index] + "_" + get_base_horn_shunck_name(iter_cnt, weight, frame_index)     
                    with self.subTest(test_name):
                        # Read in derivatives
                        size = 2
                        fx = np.load(os.path.join(ground_dir, video_folders[index],
                                                get_base_derivative_name("fx", size, frame_index) + ".npy")) 
                                    
                        fy = np.load(os.path.join(ground_dir, video_folders[index], 
                                                get_base_derivative_name("fy", size, frame_index) + ".npy"))
                        
                        ft = np.load(os.path.join(ground_dir, video_folders[index], 
                                                get_base_derivative_name("ft", size, frame_index) + ".npy"))
                        
                        # Load ground data for horn
                        ground_data = np.load(os.path.join(ground_dir, video_folders[index], 
                                            get_base_horn_shunck_name(iter_cnt, weight, frame_index) + ".npy"))
                                            
                        # Compute...
                        pred_data, pred_cost, used_iter = A02.compute_one_optical_flow_horn_shunck(fx, fy, ft, 
                                                                                        max_iter=iter_cnt, 
                                                                                        max_error=0,
                                                                                        weight=weight)
                                                                        
                        # Test
                        GT.check_for_unequal_no_diff("Failed on frame", 
                                test_name, 
                                pred_data, ground_data)
                        
                        # If the error check is working, then we should have an iteration count of 1
                        # on the first pair
                        if frame_index == 0:
                            self.assertEqual(used_iter, 1, "Not checking error threshold!")
                        else:
                            self.assertEqual(used_iter, iter_cnt, "Wrong iteration count!")
                                        
    ###########################################################################
    # TEST: compute_optical_flow
    ###########################################################################
    def test_compute_optical_flow(self):
        # For each video...
        for index in range(len(video_files)):
            # Load the video file as individual frames
            video_frames = A01.load_video_as_frames(os.path.join(input_dir, video_files[index]))
            
            # Try two different combinations of iterations and weights
            iter_weight_combos = [[15, 1.0], [5, 0.1]]
            
            for iter_cnt, weight in iter_weight_combos:     
                # Compute the flow
                flow_frames = A02.compute_optical_flow(video_frames, A02.OPTICAL_FLOW.HORN_SHUNCK, 
                                                       max_iter=iter_cnt, 
                                                       max_error=0, horn_weight=weight)
                
                # For each frame pair...
                for frame_index in range(len(video_frames)):
                    test_name = video_folders[index] + "_" + get_base_horn_shunck_name(iter_cnt, weight, frame_index)     
                    with self.subTest(test_name):                    
                        # Load ground data for horn
                        ground_data = np.load(os.path.join(ground_dir, video_folders[index], 
                                            get_base_horn_shunck_name(iter_cnt, weight, frame_index) + ".npy"))
                        
                        # Test
                        GT.check_for_unequal_no_diff("Failed on frame", 
                                test_name, 
                                flow_frames[frame_index], ground_data)    
            
def main():
    runner = unittest.TextTestRunner()
    test_cases = unittest.TestLoader().loadTestsFromTestCase(Test_A02)    
    runner.run(test_cases)
        
if __name__ == '__main__':    
    main()
