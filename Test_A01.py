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
import A01

base_dir = "assign01"
input_dir = base_dir + "/" + "input"
ground_dir = base_dir + "/" + "ground"
output_dir = base_dir + "/" + "output"

class Test_A01(unittest.TestCase):   
    
    ###########################################################################
    # load_video_as_frames
    ###########################################################################    
    def get_videos_in_input_dir(self):
        return os.listdir(input_dir)
    
    ###########################################################################
    # load_video_as_frames
    ###########################################################################    
    def load_one_ground_images(self, basename, fps):            
        out_video_folder = basename + "_" + str(fps)
                    
        # Load up ground frames
        base_ground_dir = os.path.join(ground_dir, out_video_folder)
        ground_filenames = os.listdir(base_ground_dir)
        ground_filenames.sort()
        ground_frames = []
        for filename in ground_filenames:
            ground_frames.append(cv2.imread(os.path.join(base_ground_dir, filename)))
            
        return ground_filenames, ground_frames, out_video_folder   
    
    ###########################################################################
    # do_test_after_save_frames
    ###########################################################################  
    def do_test_after_save_frames(self, full_output_path, ground_frames, ground_filenames, video_filename):
        # Check that folder path exists
        with self.subTest(video_filename + ": Output Path"):
            self.assertTrue(os.path.exists(full_output_path), "Output path not correctly created: " + full_output_path)
            
        # Check for correct number of images
        filenames = os.listdir(full_output_path)
        filenames.sort()
        with self.subTest(video_filename + ": Number of Saved Images"):
            self.assertEqual(len(filenames), len(ground_frames), "Wrong number of frames saved.")
        
        # Check for correct image names
        with self.subTest(video_filename + ": Image Names"):
            for i in range(len(ground_filenames)):
                self.assertEqual(filenames[i], ground_filenames[i], "Wrong image filename.")
            
        # Check that images are saved properly
        with self.subTest(video_filename + ": Image Contents"):
            for i in range(len(ground_frames)):
                image = cv2.imread(os.path.join(full_output_path, filenames[i]))
                GT.check_for_unequal_no_diff("Failed on frame", filenames[i], image, ground_frames[i])   
    
    ###########################################################################
    # TEST: load_video_as_frames
    ###########################################################################
    def test_load_video_as_frames(self):
        # Testing where non-existent video returns a None
        with self.subTest("Bad Filename"):
            cap_stream = GT.redirect_std_out()
            bad_frames = A01.load_video_as_frames(video_filepath="nonexistent.mp4")
            GT.restore_std_out()
            self.assertEqual(bad_frames, None, "Should return None when video cannot be loaded.")
            captured = GT.get_captured_output(cap_stream)
            captured = captured.rstrip().lstrip()
            self.assertNotEqual(captured, "", "Should print error message on bad load.")
            
        # Test each video
        video_list = self.get_videos_in_input_dir()
        fps = 30 # default
        
        for video_filename in video_list:            
            # Get basename
            basename = Path(video_filename).stem
            
            # Load images using function
            all_frames = A01.load_video_as_frames(video_filepath=os.path.join(input_dir, video_filename))
                                
            # Get ground truth images
            _, ground_frames, _ = self.load_one_ground_images(basename, fps)
                    
            # Check number of video frames loaded        
            with self.subTest(video_filename + ": Frame Count Check"):
                self.assertEqual(len(all_frames), len(ground_frames), "Incorrect number of frames added to list.")
                
            # Check each frame is correct
            with self.subTest(video_filename + ": Individual Frame Check"):            
                for index in range(len(ground_frames)):                        
                    # Compare with frame
                    GT.check_for_unequal_no_diff("Failed on frame", str(index), all_frames[index], ground_frames[index])   
        
    ###########################################################################
    # TEST: compute_wait
    ###########################################################################       
    def test_compute_wait(self):
        fps_list = {
            8: 125, 
            15: 66,
            30: 33, 
            60: 16, 
            101: 9
        }
                
        for fps in fps_list:           
            ground_wait = fps_list[fps]
            with self.subTest("FPS " + str(fps)):
                wait = A01.compute_wait(fps)
                self.assertEqual(wait, ground_wait, "Wrong wait time")
    
    ###########################################################################
    # TEST: display_frames
    ###########################################################################       
    '''
    def test_display_frames(self):        
        # Load up some ground frames
        _, ground_frames, _ = self.load_one_ground_images("noice", fps=30)
        
        # Make functions for calling
        def check_display_frames(desired_fps):    
            with self.subTest("FPS = " + str(desired_fps)):        
                desired_title = "Beans at " + str(desired_fps)            
                quick_thread = Thread(target=A01.display_frames,
                                    args=(ground_frames, desired_title, desired_fps))
                quick_thread.start()
                sleep(0.5)
                video_visible = cv2.getWindowProperty(desired_title, cv2.WND_PROP_VISIBLE)
                quick_thread.join()
                
                # Was it there?            
                self.assertTrue(video_visible > 0, "Wrong window title!")
                
                # IS it still there?     
                video_visible = cv2.getWindowProperty(desired_title, cv2.WND_PROP_VISIBLE)               
                self.assertTrue(video_visible == 0, "Window not destroyed!")     
                        
        # Check for correct video title  
        # WARNING: Does NOT check timing!!!!   
        check_display_frames(30)
        check_display_frames(60)  
    '''
    
    ###########################################################################
    # TEST: save_frames
    ########################################################################### 
    def test_save_frames(self):
        # Make some dummy frames
        dummy_frames = [
            np.zeros((256,256,3)), np.zeros((256,256,3)), np.zeros((256,256,3))
        ]
        dummy_frames[0][:,:] = (255,0,0)
        dummy_frames[1][:,:] = (0,255,0)
        dummy_frames[2][:,:] = (0,0,255)
        
        # Use a dummy output directory        
        real_output_dir = os.path.join(output_dir, "test_save_frames")
        basename = "dummy"
        fps = 78
        full_output_path = os.path.join(real_output_dir, basename + "_" + str(fps))
        
        # Remove the path just in case
        if os.path.exists(real_output_dir):
            shutil.rmtree(real_output_dir)
        
        # Save the frames
        A01.save_frames(all_frames=dummy_frames, output_dir=real_output_dir, basename=basename, fps=fps)
        
        # Do all the checks
        self.do_test_after_save_frames(full_output_path, ground_frames = dummy_frames,
                                       ground_filenames = ["image_0000000.png", "image_0000001.png", "image_0000002.png"],
                                       video_filename="dummy")
        
        # Check that directory is destroyed and recreated
        dummy_frames = dummy_frames[0:2]
        A01.save_frames(all_frames=dummy_frames, output_dir=real_output_dir, basename=basename, fps=fps)
        with self.subTest("Recreation of Output Directory"):
            filenames = os.listdir(full_output_path)
            self.assertEqual(len(filenames), len(dummy_frames), "Directory not recreated properly.")
                
        # Check for default value
        A01.save_frames(all_frames=dummy_frames, output_dir=real_output_dir, basename=basename)
        full_output_path = os.path.join(real_output_dir, basename + "_30")
        with self.subTest("Checking Default FPS"):
            self.assertTrue(os.path.exists(full_output_path), "Wrong default value for FPS.")
           
         # Cleanup after
        if os.path.exists(real_output_dir):
            shutil.rmtree(real_output_dir)          
    
    ###########################################################################
    # TEST: main
    ###########################################################################     
    def test_main(self):
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
              
        # BAD RUN
        
        # Test for argument check  
        with self.subTest("0 Arguments"):        
            exitcode, captured = run_main_program(["program"],)
            self.assertNotEqual(captured, "", "Should print error message.")
            self.assertEqual(exitcode, 1, "Should use exit(1).")
            
        with self.subTest("1 Argument"):        
            exitcode, captured = run_main_program(["program", "potato"],)
            self.assertNotEqual(captured, "", "Should print error message.")
            self.assertEqual(exitcode, 1, "Should use exit(1).")
                      
        # Test for bad video load (error message and exit)        
        with self.subTest("Bad Video Path"):        
            exitcode, captured = run_main_program(["program", "potato.mp4", output_dir],)
            self.assertNotEqual(captured, "", "Should print error message.")
            self.assertEqual(exitcode, 1, "Should use exit(1).")
        
        # CORRECT RUN
          
        def check_one_video(video_filename, fps):
            basename = Path(video_filename).stem
            ground_filenames, ground_frames, out_video_folder = self.load_one_ground_images(basename, fps)
                           
            real_output_dir = os.path.join(output_dir, "test_main")
            desired_args = ["program", 
                            os.path.join(input_dir, video_filename), 
                            real_output_dir]
            desired_title = "Input Video"
            full_output_path = os.path.join(real_output_dir, out_video_folder)
                        
            # Remove the path just in case
            if os.path.exists(real_output_dir):
                shutil.rmtree(real_output_dir)                                       
            
            # Test for display window
            with self.subTest(video_filename + ": Display Window"):      
                '''
                quick_thread = Thread(target=run_main_program,
                                        args=(desired_args,))            
                quick_thread.start()
                sleep(1.0)
                found_window = cv2.getWindowProperty(desired_title, cv2.WND_PROP_VISIBLE)
                quick_thread.join()
            
                # Check for window now
                self.assertTrue(found_window > 0, "Wrong window title!")
                '''
                
                run_main_program(desired_args)
                
            # Check for saved frames         
            self.do_test_after_save_frames(full_output_path, ground_frames, ground_filenames, video_filename)
          
            # Remove the path just in case
            if os.path.exists(real_output_dir):
                shutil.rmtree(real_output_dir) 
                
        # Call each video file
        all_videos = self.get_videos_in_input_dir()        
        fps = 30   
        for video_filename in all_videos:     
            check_one_video(video_filename, fps)    
        
        # Restore old args
        sys.argv = old_args
        
def main():
    runner = unittest.TextTestRunner()
    test_cases = unittest.TestLoader().loadTestsFromTestCase(Test_A01)    
    runner.run(test_cases)
        
if __name__ == '__main__':    
    main()

