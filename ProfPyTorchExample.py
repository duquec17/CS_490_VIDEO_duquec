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

class BasicVideoNet(nn.Module):
    def __init__(self, class_cnt):
        super().__init__()
        # Create a module list so we have slightly more control
        self.feature_extract = nn.ModuleList([
            # Really 2D convolutions (with zero padding)
            nn.Conv3d(3, 32, (1,3,3), padding="same"), # For no padding: "valid"
            nn.ReLU(),
            nn.Conv3d(32, 32, (1,3,3), padding="same"),
            nn.ReLU(),
            nn.MaxPool3d((1,2,2)),
            
            # This one uses strided conv instead of pooling
            nn.Conv3d(32, 32, (1,3,3), padding="same"),
            nn.ReLU(),
            nn.Conv3d(32, 32, (1,3,3), padding=(0,1,1), stride=(1,2,2)),
            nn.ReLU(),
            #nn.MaxPool3d((1,2,2)),
            
            nn.Conv3d(32, 32, (1,3,3), padding="same"),
            nn.ReLU(),
            nn.Conv3d(32, 32, (1,3,3), padding="same"),
            nn.ReLU(),
            nn.MaxPool3d((1,2,2)),
            
            nn.Conv3d(32, 32, (1,3,3), padding="same"),
            nn.ReLU(),
            nn.Conv3d(32, 32, (1,3,3), padding="same"),
            nn.ReLU(),
            nn.MaxPool3d((1,2,2)),
            
            # True 3D convolution (3 frames x 5 height x 5 width), padding ONLY in space
            nn.Conv3d(32, 64, (3,5,5), padding=(0,2,2)),
            nn.ReLU(),
            nn.Conv3d(64, 64, (3,5,5), padding=(0,2,2)),
            nn.ReLU(),
            nn.Conv3d(64, 64, (3,5,5), padding=(0,2,2)),
            nn.ReLU(),
            nn.MaxPool3d(2), 
            
            # Temporal ONLY filtering
            nn.Conv3d(64, 64, (4,1,1), padding="valid"),
            nn.ReLU(),                         
        ])
        
        # Classifier section        
        self.flatten = nn.Flatten()
        self.classifier_stack = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.ReLU(),                
            nn.Linear(1024, class_cnt)
        )
    def forward(self, x):
        PRINT_DEBUG = False
        # Input: (b, t, c, h, w)
        x = torch.transpose(x, 1, 2)
        # After transpose: (b, c, t, h, w)
        for index, layer in enumerate(self.feature_extract):
            if PRINT_DEBUG: print(index, ":", x.shape)
            x = layer(x)      
        if PRINT_DEBUG: print("BEFORE FLAT:", x.shape)      
        x = self.flatten(x)
        logits = self.classifier_stack(x)
        return logits
    
class RNNVideoNet(nn.Module):
    def __init__(self, class_cnt):
        super().__init__()
        # Create a module list so we have slightly more control
        self.feature_extract = nn.ModuleList([
            # Really 2D convolutions (with zero padding)
            nn.Conv3d(3, 32, (1,3,3), padding="same"), # For no padding: "valid"
            nn.ReLU(),
            nn.Conv3d(32, 32, (1,3,3), padding="same"),
            nn.ReLU(),
            nn.MaxPool3d((1,2,2)),
            
            nn.Conv3d(32, 32, (1,3,3), padding="same"),
            nn.ReLU(),
            nn.Conv3d(32, 32, (1,3,3), padding="same"),
            nn.ReLU(),
            nn.MaxPool3d((1,2,2)),
            
            nn.Conv3d(32, 32, (1,3,3), padding="same"),
            nn.ReLU(),
            nn.Conv3d(32, 32, (1,3,3), padding="same"),
            nn.ReLU(),
            nn.MaxPool3d((1,2,2)),
            
            nn.Conv3d(32, 32, (1,3,3), padding="same"),
            nn.ReLU(),
            nn.Conv3d(32, 32, (1,3,3), padding="same"),
            nn.ReLU(),
            nn.MaxPool3d((1,2,2)),
            
            nn.Conv3d(32, 64, (1,3,3), padding="same"),
            nn.ReLU(),
            nn.Conv3d(64, 64, (1,3,3), padding="same"),
            nn.ReLU(),
            nn.MaxPool3d((1,2,2)),
            
            nn.Conv3d(64, 64, (1,3,3), padding="same"),
            nn.ReLU(),
            nn.Conv3d(64, 64, (1,3,3), padding="same"),
            nn.ReLU(),
            nn.MaxPool3d((1,2,2))                                     
        ])
        
        # RNN and classifier section        
        self.flatten = nn.Flatten(start_dim=2)
        
        self.rnn = nn.RNN(input_size=1024, 
                          hidden_size=1024,
                          num_layers=1,
                          batch_first=True)
        
        self.classifier_stack = nn.Sequential(                           
            nn.Linear(1024, class_cnt)
        )
    def forward(self, x):
        PRINT_DEBUG = False
        # Input: (b, t, c, h, w)
        x = torch.transpose(x, 1, 2)
        # After transpose: (b, c, t, h, w)
        for index, layer in enumerate(self.feature_extract):
            #print(index, ":", x.shape)
            x = layer(x)
        if PRINT_DEBUG: print("FEATURES:", x.shape)
        # After features: (b, c, t, h, w)    
        x = torch.transpose(x, 1, 2)
        # After swap AGAIN: (b, t, c, h, w)                       
        x = self.flatten(x)
        # After flatten: (b, t, c*h*w) 
        if PRINT_DEBUG: print("FLATTENED:", x.shape)
        out, _ = self.rnn(x)
        if PRINT_DEBUG: print("OUT:", out.shape)
        out = out[:,-1,:]        
        logits = self.classifier_stack(out)
        return logits
    
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
# TRAIN ONE EPOCH
###############################################################################
def train_one_epoch(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    # For HMDB: (X, _, y)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            
###############################################################################
# TEST/EVALUATE ONE EPOCH
###############################################################################            
def test_one_epoch(dataloader, model, loss_fn, data_name, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        # For HMDB: X, _, y
        for X, y in dataloader:            
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(data_name + f" Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss

###############################################################################
# MAIN
###############################################################################
def main():
    # Set device
    device = "cpu" # "cuda"
    
    # Create data transform
    target_size = (256,256)
    data_transform = v2.Compose([v2.ToImage(), 
                                 v2.ToDtype(torch.float32, scale=True),
                                 v2.Resize(target_size)])
    
    # Create model
    model = BasicVideoNet(class_cnt=2)
    #model = RNNVideoNet(class_cnt=2)
    
    # Print out model
    print("NETWORK:")
    print(model)
    print_count_parameters(model)
    
    # Move to GPU
    model = model.to(device)
    
    # Create datasets and dataloaders
    max_frame_cnt = 15
    train_data = MemeDataset("./upload/memes", is_train=True, 
                          transform=data_transform,
                          frame_stride=1,
                          max_frame_cnt=max_frame_cnt)
    test_data = MemeDataset("./upload/memes", is_train=False, 
                          transform=data_transform,
                          frame_stride=1,
                          max_frame_cnt=max_frame_cnt)

    train_ds = DataLoader(train_data, batch_size=2)
    test_ds = DataLoader(test_data, batch_size=2)
    
    # Set loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Do training!
    epochs = 15
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_one_epoch(train_ds, model, loss_fn, optimizer, device)
        train_loss = test_one_epoch(train_ds, model, loss_fn, "Train", device)
        test_loss = test_one_epoch(test_ds, model, loss_fn, "Test", device)
    print("Done!")

if __name__ == "__main__":
    main()
    
