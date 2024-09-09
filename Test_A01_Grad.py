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
import A01
from Test_A01 import *

class Test_A01_Grad(Test_A01):  
    ###########################################################################
    # TEST: change_fps
    ###########################################################################      
    def test_change_fps(self):        
        # Load ground files
        basename = "noice"
        fps_cnts = [8, 15, 30, 60]
        ground_frames = {}
        for fps in fps_cnts:
            _, ground_frames[fps], _ = self.load_one_ground_images(basename, fps)
        
        # Do every combination of 30 to fps
        test_frames = {}
        for target_fps in fps_cnts:
            # Convert
            test_frames[target_fps] = A01.change_fps(input_frames=ground_frames[30], old_fps=30, new_fps=target_fps)
            # Check
            with self.subTest("Convert 30 to " + str(target_fps)):
                # Check counts
                self.assertEqual(len(test_frames[target_fps]), 
                                 len(ground_frames[target_fps]), 
                                 "Wrong number of frames.")
                
                # Check actual images
                for i in range(len(ground_frames[target_fps])):
                    GT.check_for_unequal_no_diff("Failed on frame", str(i), 
                                                 test_frames[target_fps][i],
                                                 ground_frames[target_fps][i])   
      
    ###########################################################################
    # TEST: main (grad)
    ###########################################################################                
    def test_main_grad(self):
        # Save old input args
        old_args = sys.argv
        
        # Define thread for running main program
        def run_main_program(input_args):
            sys.argv = input_args
            cap_stream = GT.redirect_std_out()   
            try:
                A01.main()
                exitcode = 0
            except SystemExit as se:
                exitcode = 1
            GT.restore_std_out()
            captured = GT.get_captured_output(cap_stream)
            captured = captured.rstrip().lstrip()   
            return exitcode, captured
        
        # CORRECT RUN with new fps
        
        def check_one_video(video_filename, fps):
            basename = Path(video_filename).stem
            ground_filenames, ground_frames, out_video_folder = self.load_one_ground_images(basename, fps)
                      
            real_output_dir = os.path.join(output_dir, "test_main_grad")              
            desired_args = ["program", 
                            os.path.join(input_dir, video_filename), 
                            real_output_dir,
                            str(fps)]
            desired_title = "Input Video"
            second_desired_title = "Output Video"
            full_output_path = os.path.join(real_output_dir, out_video_folder)  
            
            # Remove the path just in case
            if os.path.exists(real_output_dir):
                shutil.rmtree(real_output_dir)                              
            
            # Test for display window
            with self.subTest(video_filename + ": Display Window"):      
                quick_thread = Thread(target=run_main_program,
                                        args=(desired_args,))            
                quick_thread.start()
                sleep(1.0)
                found_window = cv2.getWindowProperty(desired_title, cv2.WND_PROP_VISIBLE)
                sleep(3.0)
                found_second_window = cv2.getWindowProperty(second_desired_title, cv2.WND_PROP_VISIBLE)
                
                quick_thread.join()
            
                # Check for window now
                self.assertTrue(found_window > 0, "Wrong input window title!")
                self.assertTrue(found_second_window > 0, "Wrong output window title!")
                        
            # Check for saved frames         
            self.do_test_after_save_frames(full_output_path, ground_frames, ground_filenames, video_filename)
            
            # Remove the path just in case
            if os.path.exists(real_output_dir):
                shutil.rmtree(real_output_dir) 
          
        # Call each video file
        all_videos = self.get_videos_in_input_dir()        
        fps = 60 # DIFFERENT FPS   
        for video_filename in all_videos:     
            check_one_video(video_filename, fps)    
        
        # Restore old args
        sys.argv = old_args
            
def main():
    runner = unittest.TextTestRunner()
    test_cases = unittest.TestLoader().loadTestsFromTestCase(Test_A01_Grad)    
    runner.run(test_cases)

if __name__ == '__main__':    
    main()
