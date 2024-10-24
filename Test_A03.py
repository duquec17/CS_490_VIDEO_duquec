from General_A03 import *
import A03
import shutil

UG_DOG_LIST = [7,8,11,9]
START_DOG_FRAME = 30
MAX_DOG_FRAMES = 60 # 120
show_dog_videos = True

###############################################################################
# TRACKING
###############################################################################

def track_doggos(dog_video_list, show_dog_videos):
    # Setup all metrics
    all_metrics = {}
        
    # For each dog video...
    for dog_index in dog_video_list:
        # Load data
        dog_images, dog_boxes, video_name = load_dog_video(dog_index, max_images_to_load=MAX_DOG_FRAMES,
                                                           starting_frame_index=START_DOG_FRAME)
        
        # Make specific output directory
        dog_output = os.path.join(BASE_OUT_DIR, video_name)
    
        # Re-create output directory
        if os.path.exists(dog_output):
            shutil.rmtree(dog_output)       
        os.makedirs(dog_output)
        
        # What the dog doing?
        one_metrics = compute_track_metrics(video_name, dog_images, dog_boxes, 
                                            A03.track_doggo, dog_output)
        
        # Add to metrics
        all_metrics[video_name] = one_metrics
        
        # If requested, show video
        if show_dog_videos:
            show_predicted_video(dog_index)
                
    # Save metrics
    print_metrics(all_metrics)
    with open(BASE_OUT_DIR + "/RESULTS.txt", "w") as f:
        print_metrics(all_metrics, f)
        
def main():
    track_doggos(UG_DOG_LIST, show_dog_videos=show_dog_videos)
    
if __name__ == "__main__": 
    main()
    