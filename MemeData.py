import torch
from torch.utils.data import Dataset, DataLoader
import os
from sklearn.model_selection import train_test_split
import numpy as np
from torchvision.io import read_video
from torchvision.transforms import v2
import cv2

class MemeDataset(Dataset):
    def __init__(self, basedir, is_train,
                 seed=42, transform=None,
                 target_transform=None,
                 frame_stride=1):
        self.basedir = basedir
        self.transform = transform
        self.target_transform = target_transform
        self.frame_stride = frame_stride
        
        train_list = []
        test_list = []
        
        for subfolder in ["good", "bad"]:
            subpath = os.path.join(basedir, subfolder)
            all_files = os.listdir(subpath)
            for i in range(len(all_files)):
                all_files[i] = os.path.join(subfolder, all_files[i])
            all_files = np.array(all_files)
            train, test = train_test_split(all_files, test_size=0.25,
                                           random_state=seed)
            train_list.append(train)
            test_list.append(test)
        
        if is_train:
            self.file_list = train_list
        else:
            self.file_list = test_list
            
        self.file_list = np.array(self.file_list)
        self.file_list = self.file_list.flatten()
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        filepath = self.file_list[idx]
        entirepath = os.path.join(self.basedir, filepath)
        
        video_data = read_video(entirepath, pts_unit="sec",
                                start_pts=0,
                                end_pts=None,
                                output_format="TCHW")
        video = video_data[0]
        
        if self.frame_stride > 1:
            out_video = []
            for i in range(0, len(video), self.frame_stride):
                out_video.append(video[i])
            video = torch.stack(out_video, 0)            
        
        if self.transform is not None:
            video = self.transform(video)
        
        label = 1
        if "bad" in filepath:
            label = 0
            
        if self.target_transform is not None:
            label = self.target_transform(label)
            
        return video, label
    
def main():
    
    # 0.15
    #transform = v2.Compose([
    #    v2.ToImageTensor(),
    #    v2.ConvertImageDtype(dtype=torch.float32)        
    #])
    
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((360, 640)),
        v2.RandomHorizontalFlip(p=0.5)
    ])
    
    dataset = MemeDataset("./upload/memes", is_train=True, 
                          transform=transform,
                          frame_stride=3)

    train_ds = DataLoader(dataset, batch_size=1)
    
    for X, y in train_ds:
        print(X.shape)
        X = X.numpy()
        X = X[0]        
        X = np.transpose(X, [0, 2, 3, 1])
                
        index = 0
        key = -1
        while key != 27:
            image = cv2.cvtColor(X[index], cv2.COLOR_RGB2BGR)
            cv2.imshow("Video", image)
            key = cv2.waitKey(33)
            index += 1
            if index >= len(X):
                index = 0
                
        cv2.destroyAllWindows()           


if __name__ == "__main__":
    main()
    
        
    
    
