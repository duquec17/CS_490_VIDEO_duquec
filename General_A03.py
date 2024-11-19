import os
import numpy as np
import cv2
import sys

BASE_DOG_DIR = os.path.join(".", "data", "dog")
BASE_OUT_DIR = os.path.join(".", "assign03", "output")

###############################################################################
# LOAD DATA
###############################################################################
def get_dog_folder_name(dog_index):
    return "dog-" + str(dog_index)

def load_dog_video(dog_index, max_images_to_load=None, starting_frame_index=None):
    # Get dog folder name
    folder_name = get_dog_folder_name(dog_index)
    full_dog_folder = os.path.join(BASE_DOG_DIR, folder_name)
    
    # Load the ground truth info
    ground_filepath = os.path.join(full_dog_folder, "groundtruth.txt")
    with open(ground_filepath, "r") as f:
        all_ground_lines = f.readlines()
     
    # Parse it 
    dog_boxes = []
    for line in all_ground_lines:
        # Clean it up
        line = line.strip()
        # Split by commas
        tokens = line.split(",")
        # Orig: sx, sy, width, height
        # Get sy, sx, ey, ex
        sx = int(tokens[0])
        sy = int(tokens[1])
        ex = sx + int(tokens[2])
        ey = sy + int(tokens[3])
        # Add box
        dog_boxes.append([sy, sx, ey, ex])
    
    # Get image folder name
    full_image_folder = os.path.join(full_dog_folder, "img")
    
    # Load the images
    dog_images = []
    all_dog_filenames = list(os.listdir(full_image_folder))
    all_dog_filenames.sort()
    
    # Do we want to get only a certain number of frames?
    if starting_frame_index is not None:
        all_dog_filenames = all_dog_filenames[starting_frame_index:]
        dog_boxes = dog_boxes[starting_frame_index:]
        
    if max_images_to_load is not None:
        all_dog_filenames = all_dog_filenames[:max_images_to_load]
        dog_boxes = dog_boxes[:max_images_to_load]
    
    for filename in all_dog_filenames:
        # Get image filepath
        image_filepath = os.path.join(full_image_folder, filename)
        # Load image
        image = cv2.imread(image_filepath)
        # Add to list
        dog_images.append(image)
     
    # Return results
    return dog_images, dog_boxes, folder_name

###############################################################################
# COMPUTES INTERSECTION OVER UNION FOR BOUNDING BOXES
###############################################################################
def compute_one_IOU(predicted, ground):
    # Bounding box stored as (y1, x1, y2, x2)   
    def compute_area(left, right, top, bottom):
        width = right - left
        height = bottom - top
        width = max(0, width)
        height = max(0, height)        
        area = width * height
        return area

    # Get intersection
    left = max(predicted[1], ground[1])
    right = min(predicted[3], ground[3])
    top = max(predicted[0], ground[0])
    bottom = min(predicted[2], ground[2])    
    intersection = compute_area(left, right, top, bottom)
    
    # Get union
    area_pred = compute_area(predicted[1], predicted[3], predicted[0], predicted[2])
    area_ground = compute_area(ground[1], ground[3], ground[0], ground[2])
    union = area_pred + area_ground - intersection     

    # Get IOU
    iou = intersection / union
            
    return iou

###############################################################################
# DRAWS BOUNDING BOXES ON IMAGES
###############################################################################
def draw_dog_box(image, bb, color):
    cv2.rectangle(image, (bb[1], bb[0]), (bb[3], bb[2]), color, thickness=2)     

###############################################################################
# PREDICTS BOUNDING BOXES ON VIDEO AND COMPUTES METRICS
###############################################################################
def compute_track_metrics(video_name, video_frames, all_ground_boxes, 
                          track_dog_func, out_dir):
    # Prepare metric dictionary
    metrics = {}
    metrics["Accuracy"] = 0.0
    metrics["IOU"] = 0.0
    
    # Get total count
    total_cnt = len(video_frames)
    
    # Print startup
    print("Tracking doggo in video", video_name)
    all_pred_boxes = track_dog_func(video_frames, all_ground_boxes[0])
    
    # For each frame...
    for index in range(len(video_frames)):
        # Get image, ground box, and predicted box
        image = np.copy(video_frames[index])
        ground_box = all_ground_boxes[index]
        pred_box = all_pred_boxes[index]

        # Draw bounding boxes on image
        draw_dog_box(image, ground_box, (0,0,0))
        draw_dog_box(image, pred_box, (0,255,0))

        # Show images (DEBUG)                
        #cv2.imshow("IMAGE", image)        
        #cv2.waitKey(-1)

        # Save image
        cv2.imwrite(out_dir + "/%s_%05d.png" % (video_name, index), image)
        
        # Compute IOU
        one_iou = compute_one_IOU(pred_box, ground_box)
        metrics["IOU"] += one_iou

        # Good overlap?
        if one_iou >= 0.5:
            metrics["Accuracy"] += 1.0

        # Print progress
        percent = 100.0*index / total_cnt
        print("%.1f%% complete...       " % percent, end="\r", flush=True)

    # Print complete    
    print(video_name, "complete!                                 ")

    # Average out metrics
    metrics["Accuracy"] /= total_cnt
    metrics["IOU"] /= total_cnt

    # Return metrics
    return metrics

###############################################################################
# PRINTS METRICS (to STDOUT or file)
###############################################################################
def print_metrics(all_metrics, stream=sys.stdout):
    # Compute average
    ave_acc = 0
    ave_iou = 0
    
    # Print header
    print("VIDEO\tACC\tIOU", file=stream)    
    for name in all_metrics:
        print(name, end="\t", file=stream)    
        print("%0.3f" % all_metrics[name]["Accuracy"], end="\t", file=stream)  
        print("%0.4f" % all_metrics[name]["IOU"], end="\n", file=stream)  
        ave_acc += all_metrics[name]["Accuracy"]
        ave_iou += all_metrics[name]["IOU"]
    ave_acc /= len(all_metrics)
    ave_iou /= len(all_metrics)
    print("AVE", end="\t", file=stream)    
    print("%0.3f" % ave_acc, end="\t", file=stream)  
    print("%0.4f" % ave_iou, end="\n", file=stream)  
        
###############################################################################
# LOAD PREDICTED VIDEO TO DISPLAY IT
###############################################################################
def show_predicted_video(dog_index):
    dog_video_name = get_dog_folder_name(dog_index)
    full_dog_path = os.path.join(BASE_OUT_DIR, dog_video_name)
    dog_files = list(os.listdir(full_dog_path))
    dog_files.sort()
    dog_images = []
    for filename in dog_files:
        dog_images.append(cv2.imread(os.path.join(full_dog_path, filename)))
        
    # Loop through and show images
    index = 0
    key = -1
    while key == -1:
        image = np.copy(dog_images[index])        
        cv2.imshow(dog_video_name, image)
        key = cv2.waitKey(33)
        
        index += 1
        if index >= len(dog_images):
            index = 0
            
    cv2.destroyWindow(dog_video_name)

###############################################################################
# MAIN FUNCTION: Really for debugging/checking
###############################################################################  
def main():
    # Load dog dataset
    #[1,6,12,17,19,2]
    dog_index = 2
    max_images_to_load = 60 #120
    starting_index =  30 #0
    dog_images, dog_boxes, dog_video_name = load_dog_video(dog_index, 
                                                           max_images_to_load=max_images_to_load,
                                                           starting_frame_index=starting_index)
    
    # Loop through and show images
    index = 0
    key = -1
    while key == -1:
        image = np.copy(dog_images[index])
        draw_dog_box(image, dog_boxes[index], (0,0,0))
        cv2.imshow("DOG", image)
        key = cv2.waitKey(33)
        
        index += 1
        if index >= len(dog_images):
            index = 0


if __name__ == "__main__":
    main()
    