import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2
import cv2
import sys
import numpy as np
import os
import time
from prettytable import PrettyTable
from sklearn.metrics import (accuracy_score, f1_score)
from Prepare_A04 import *
import A04 as ASSIGN

###############################################################################
# ASK FOR APPROACH NAME
###############################################################################
def ask_for_approach_name(allow_select_all=False):
    # Get names of all approaches
    all_names = ASSIGN.get_approach_names()
    
    # Which one?
    if allow_select_all:
        print("Approach names (-1 for all):")
    else:
        print("Approach names:")
        
    for i in range(len(all_names)):
        print(str(i) + ". " + all_names[i])
    choice = int(input("Enter choice: "))
    
    if allow_select_all:
        if choice < 0:
            chosen_approach_names = all_names
        else:
            chosen_approach_names = [all_names[choice]]
    else:
        chosen_approach_names = all_names[choice]
    
    return chosen_approach_names

###############################################################################
# PRINT OUT WEIGHTS FOR NETWORK
# TAKEN FROM: https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model
###############################################################################
def print_count_parameters(model, stream=sys.stdout):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():        
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table, file=stream)
    print(f"Total Trainable Params: {total_params}", file=stream)
    return total_params

###############################################################################
# GET HMDB51 DATASET
###############################################################################
def get_hmdb_dataset(data_params, train, transform):
    return torchvision.datasets.HMDB51( root=data_params["video_dir"],
                                        annotation_path=data_params["split_dir"],
                                        frames_per_clip=30,
                                        step_between_clips=30, 
                                        frame_rate=15,
                                        num_workers=3,
                                        fold=1, 
                                        train=train, 
                                        transform=transform, 
                                        output_format='TCHW')

###############################################################################
# PERFORM TRAINING
###############################################################################
def train():   
    # Ask about undergraduate or graduate
    data_params = ask_for_correct_data_params()
    
    # Create output directory
    os.makedirs(data_params["output_dir"], exist_ok=True)
    
    # Get approach name
    approach_name = ask_for_approach_name()
    
    # Create data transforms
    train_transform = ASSIGN.get_data_transform(approach_name, training=True)
    test_transform = ASSIGN.get_data_transform(approach_name, training=False)
    
    # Load data
    training_data = get_hmdb_dataset(data_params, True, train_transform)    
    test_data = get_hmdb_dataset(data_params, False, test_transform)
            
    print("TRAINING DATA VIDEO SAMPLES:", len(training_data))
    print("TESTING DATA VIDEO SAMPLES:", len(test_data))
        
    # Create dataloaders
    batch_size = ASSIGN.get_batch_size(approach_name)
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    
    # Get class count
    class_cnt = len(data_params["class_list"])
            
    # Create the model
    model = ASSIGN.create_model(approach_name, class_cnt)
    print("MODEL:", approach_name)
    print(model)
    print_count_parameters(model)
    
    # Move to GPU if possible
    device = ("cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")
        
    model = model.to(device)
                 
    # Train classifiers
    start_time = time.time()
    print("Training " + approach_name + "...")
    model = ASSIGN.train_model(approach_name, model, device, train_dataloader, test_dataloader)
    print("Training complete!")
    print("Time taken:", (time.time() - start_time))
    
    # Save the model
    model_path = os.path.join(data_params["output_dir"], "model_" + approach_name + ".pth")
    torch.save(model.state_dict(), model_path)
    print("Model saved to:", model_path)
    
###############################################################################
# CALCULATE METRICS
###############################################################################
def compute_metrics(ground, pred):    
    scores = {}
    scores["accuracy"] = accuracy_score(y_true=ground, y_pred=pred)
    scores["f1"] = f1_score(y_true=ground, y_pred=pred, average="macro")
    return scores

###############################################################################
# GET PREDICTIONS FROM MODEL
###############################################################################
def get_predictions_and_ground(model, dataloader, device):
    # Set model to evaluation mode
    model.eval()
    # Create lists for ground and pred
    all_ground = []
    all_pred = []
    
    with torch.no_grad():
        for X, y in dataloader:
            # Append ground truth info
            all_ground.append(y)
            
            # Move data to device
            X, y = X.to(device), y.to(device)
            
            # Run prediction
            pred = model(X)
            
            # Get largest class prediction
            pred = pred.argmax(1)
            
            # Move to CPU
            pred = pred.cpu()
            
            # Append to list
            all_pred.append(pred)
            
    # Convert to single Tensor and then numpy
    all_ground = torch.concat(all_ground).numpy()
    all_pred = torch.concat(all_pred).numpy()
    
    return {"ground": all_ground, "pred": all_pred}

###############################################################################
# PRINTS RESULTS (to STDOUT or file)
###############################################################################
def print_results(approach_data, stream=sys.stdout):
    boundary = "****************************************"
    
    ###########################################################################
    # Names and descriptions
    ###########################################################################
    
    print(boundary, file=stream)
    print("APPROACHES: ", file=stream)   
    print(boundary, file=stream)
    print("", file=stream)
    
    for approach_name in approach_data:
        print("*", approach_name, file=stream)    
        print("\t", ASSIGN.get_approach_description(approach_name), file=stream)
        print("", file=stream)  
        
        # Grab at least one model metric list
        model_metrics = approach_data[approach_name]["metrics"]
       
    ###########################################################################   
    # Results
    ###########################################################################
    
    print(boundary, file=stream)
    print("RESULTS:", file=stream)     
    print(boundary, file=stream) 
    
    # Create header
    header = "APPROACH"    
    for data_type in model_metrics:        
        data_metrics = model_metrics[data_type]
        for key in data_metrics:
            header += "\t" + data_type + "_" + key    
    table_data = header + "\n"
    
    # Add data
    for approach_name in approach_data:
        model_metrics = approach_data[approach_name]["metrics"]
        table_data += approach_name
                
        for data_type in model_metrics:        
            data_metrics = model_metrics[data_type]
            for key in data_metrics:
                cell_string = "\t%.4f" % data_metrics[key]
                table_data += cell_string
        table_data += "\n"
        
    print(table_data, file=stream)       
    
    ###########################################################################   
    # Models
    ###########################################################################
                    
    print(boundary, file=stream)
    print("MODEL ARCHITECTURES:", file=stream)       
    print(boundary, file=stream)
    for approach_name in approach_data:
        model = approach_data[approach_name]["model"]
        print("*", approach_name, file=stream)    
        print(model, file=stream)    
        print("", file=stream)     
        print_count_parameters(model, file=stream)
        print("", file=stream)      
  
###############################################################################
# EVALUATE MODEL(S)
###############################################################################
def evaluate():  
    # Ask about undergraduate or graduate
    data_params = ask_for_correct_data_params()
           
    # Get approach name(s)
    chosen_approach_names = ask_for_approach_name(allow_select_all=True)
    
    # For each approach...
    approach_data = {}
    
    for approach_name in chosen_approach_names:
        print("EVALUATING APPROACH:", approach_name)    
        approach_data[approach_name] = {}
        
        # Create only the testing data transform    
        transform = ASSIGN.get_data_transform(approach_name, training=False)
            
        # Load datasets        
        training_data = get_hmdb_dataset(data_params, True, transform)    
        test_data = get_hmdb_dataset(data_params, False, transform)
                
        print("TRAINING DATA VIDEO SAMPLES:", len(training_data))
        print("TESTING DATA VIDEO SAMPLES:", len(test_data))
    
        # Create dataloaders
        batch_size = ASSIGN.get_batch_size(approach_name)
        train_dataloader = DataLoader(training_data, batch_size=batch_size)
        test_dataloader = DataLoader(test_data, batch_size=batch_size)
        
        # Set number of classes
        class_cnt = len(data_params["class_list"])
        
        # Create the model
        model = ASSIGN.create_model(approach_name, class_cnt)
        print("MODEL:", approach_name)
        print(model)
                    
        # Move to GPU if possible
        device = ("cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        print(f"Using {device} device")
            
        model = model.to(device)
            
        # Load up previous weights
        model_path = os.path.join(data_params["output_dir"], "model_" + approach_name + ".pth")
        model.load_state_dict(torch.load(model_path))
        print("Model loaded from:", model_path)
        approach_data[approach_name]["model"] = model
    
        # Evaluate    
        train_eval_data = get_predictions_and_ground(model, train_dataloader, device)
        print("Data acquired from training...")
        test_eval_data = get_predictions_and_ground(model, test_dataloader, device)
        print("Data acquired from testing...")
        
        # Get metric values
        model_metrics = {}
        model_metrics["TRAINING"] = compute_metrics(**train_eval_data)
        model_metrics["TESTING"] = compute_metrics(**test_eval_data)       
        
        # Store model metrics
        approach_data[approach_name]["metrics"] = model_metrics
             
    # Print and save metrics
    print_results(approach_data)
    if len(chosen_approach_names) == 1:
        result_filename = chosen_approach_names[0] + "_RESULTS.txt"
    else:
        result_filename = "ALL_RESULTS.txt"
        
    with open(data_params["output_dir"] + "/" + result_filename, "w") as f:
        print_results(approach_data, stream=f)
    
###############################################################################
# DEBUGGING MAIN FUNCTION
###############################################################################
def main():  
    # DEBUG: Visualize the training or testing data
    
    # Ask about undergraduate or graduate
    data_params = ask_for_correct_data_params()
    
    # Set whether we're using train/test and/or shuffling    
    training = False    
    shuffle = True
    
    # Get approach name
    approach_name = ask_for_approach_name()
    
    # Create data transforms
    data_transform = ASSIGN.get_data_transform(approach_name, training=training)
    
    # Load dataset
    hmdb_data = get_hmdb_dataset(data_params, train=training, transform=data_transform)      
    print("NUMBER OF VIDEOS:", len(hmdb_data))
    
    # Create dataloader
    dataloader = DataLoader(hmdb_data, batch_size=1, shuffle=shuffle)
        
    #for data in dataloader: 
    for X,_,y in dataloader:
        # Grab the first item
        X = X[0]
        y = y[0]
        
        # Move to cpu and convert to numpy
        video = X.detach().cpu().numpy()
        label = y.detach().cpu().numpy()
                
        print("Video shape:", video.shape)
        print("Data type of video:", video.dtype)
        print("Maximum value in video:", np.amax(video))
        label_str = "Label: " + data_params["class_list"][label] + " (" + str(label) + ")"
        print(label_str)
                
        frame_index = 0
        key = -1
        while key != 27:
            image = video[frame_index]
            image = np.transpose(image, [1,2,0])
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imshow(label_str, image)
            key = cv2.waitKey(33)
            
            frame_index += 1
            
            if frame_index >= len(video):
                frame_index = 0

        cv2.destroyAllWindows()
   
if __name__ == "__main__":
    main()
    