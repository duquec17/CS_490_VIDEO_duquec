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

###############################################################################
# Definitions of functions
###############################################################################

# Function that returns a list of the names of all combinations
# that will be tested. Makes use of self-documented names.
def get_approach_names():
    
 return ["SimpleCNN", "ResNetTransferLearning"]

# Function that given the approach name returns a text
# description of what makes this approach distinct. No 
# more than 1 sentence in terms of length.
def get_approach_description(approach_name):
    descrip = {
        "SimpleCNN": "Basic CNN with 3 Convolutional layers",
        "ResNetTransferLearning": "ResNet-based model pre-trained on ImageNet, fine-tuned on HMDB51"
    }
    
    return descrip.get(approach_name, "Unknown approach.")



# Function that when given approach name and if told it is
# training data, returns the appropriate dataset transform
# Does NOT augment data for non-training data.
def get_data_transform(approach_name, training):
    # Checks to see if this is for transformation is for data training
    if training:
        # For training data, applies data augmentation
        # Augment ex: Croping, flipping, etc.
        data_transforms = v2.Compose([v2.ToImageTensor(), 
                                  v2.ConvertImageDtype()])
    else:
        # For non-training data, only necessary transformations (no augments)
        data_transforms = v2.Compose([v2.ToImageTensor(), 
                                  v2.ConvertImageDtype()])
    return data_transforms

# Function that given the approach name, returns the preferred
# batch size. This can be hardcoded to a single value if there
# is no preference, but allows one to use a smaller batch size
# if the architecture is large enough where that is a concern.
def get_batch_size(approach_name):
    
 return

# Function that given the approach name & output class_cnt,
# builds and returns a PyTorch neural network that takes a
# video of shape[batch, frames, channels, height, width] and
# outputs a vector with class_cnt elements.
def create_model(approach_name, class_cnt):
    
 return 

# Function that given the provided model, the device it is
# located, and the relevant dataloaders, train this model and
# return it.
def train_model(approach_name, model, device, train_dataloader, test_dataloader):
    
 return 
