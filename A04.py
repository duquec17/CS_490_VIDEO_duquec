# Information on Code
# Assignment 04 of CS 490 Video Processing and Vision
# 
# Purpose: Goal of this assignment is to write code to 
# train and evaluate PyTorch neural network models to 
# predict classes from a SUBSET of the HMDB51 dataset.
# 
# Writer: Cristian Duque
# - Based off material from Professor Reale

###############################################################################
# IMPORTS
###############################################################################

from pathlib import Path
import shutil
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import v2
import cv2
import numpy as np
import os
import sys
from prettytable import PrettyTable
from MemeData import *

###############################################################################
# Definitions of functions
###############################################################################

# Function that returns a list of the names of all combinations
# that will be tested. Makes use of self-documented names.
def get_approach_names():
    # Still thinking about whether to do ResNet or not
 return ["SimpleCNN", "ResNetTransferLearning"]

# Function that given the approach name returns a text
# description of what makes this approach distinct. No 
# more than 1 sentence in terms of length.
def get_approach_description(approach_name):
    # Definition of each approach
    descrip = {
        "SimpleCNN": "Basic Convolutional Neural Network with 3 Convolutional layers",
        "ResNetTransferLearning": "ResNet-based model pre-trained on ImageNet, fine-tuned on HMDB51"
    }
    
    return descrip.get(approach_name, "Unknown approach.")



# Function that when given approach name and if told it is
# training data, returns the appropriate dataset transform
# Does NOT augment data for non-training data.
def get_data_transform(approach_name, training):
    # Define and adjustable resolution size
    target_resolution = (240,240)
    
    # Checks to see if this transformation is for data training
    if training:
        # For training data, uses transforms & data augmentation
        data_transforms = v2.Compose([
            # Sets input as image and resizes to target resolution variable
            v2.ToImage(),
            v2.Resize(target_resolution),
            
             # Data augmentation - flip horizontally
            v2.RandomHorizontalFlip(),
            v2.RandomRotation(10), 
            
            # Coverts to correct data type
            v2.ToTensor(),
            v2.ConvertImageDtype(torch.float32)
        ])
    else:
        # For non-training data, uses only transforms & no augments
        data_transforms = v2.Compose([
            v2.ToImage(),  # Ensure input is treated as an image
            v2.Resize(target_resolution),  # Resize frames to the target resolution
            v2.ToTensor(),
            v2.ConvertImageDtype(torch.float32)
        ])
    return data_transforms

# Function that given the approach name, returns the preferred
# batch size. This can be hardcoded to a single value if there
# is no preference, but allows one to use a smaller batch size
# if the architecture is large enough where that is a concern.
def get_batch_size(approach_name):
    
    # Alter batch size based on name
    if approach_name == "SimpleCNN":
        # Size for SimpleCNN
        batch_size = 64
    else:
        # Size for default batch_size
        batch_size = 32
        
    return batch_size

# Function that given the approach name & output class_cnt,
# builds and returns a PyTorch neural network that takes a
# video of shape[batch, frames, channels, height, width] and
# outputs a vector with class_cnt elements.
def create_model(approach_name, class_cnt):
    # Checks to see what approach name is listed as and act based on match
    if approach_name == "SimpleCNN":
        # Create a simple CNN model
        model = nn.Sequential(
            nn.Conv3d(2,16,kernel_size=2,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2,stride=2),
            nn.Conv3d(16,32,kernel_size=2,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2,stride=2),
            nn.Flatten(),
            nn.Linear(32*7*7*7, 128),
            nn.ReLU(),
            nn.Linear(128,class_cnt)
        )
        return model
    else:
        return ValueError("Unknown approach_name: {approach_name}")

# Function that given the provided model, the device it is
# located, and the relevant dataloaders, train this model and
# return it.
def train_model(approach_name, model, device, train_dataloader, test_dataloader):
    # Define the loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for batch in train_dataloader:
        print(f"Batch contents: {len(batch)}")
        print(batch)
        break
    
    # Training loop (simplified)
    model.train()
    for epoch in range(10):  # Train for 10 epochs (can be adjusted)
        for inputs, labels in train_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/10, Loss: {loss.item():.4f}")
    
    return model