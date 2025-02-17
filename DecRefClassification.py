# import os 
import torch
# import torchvision
# import numpy as np
# import matplotlib.pyplot as plt
# from torchvision import transforms
# from torchvision.datasets import ImageFolder
# from torch.utils.data.dataloader import DataLoader
# from torch.utils.data import random_split
# from torchvision.utils import make_grid
# from torch.utils.data import Dataset


import torch.nn as nn
import torch.nn.functional as F
# from PIL import Image
# from datetime import datetime
# from VideoSinglePatchDataset import VideoSinglePatchDataset
# from DeviceDataLoader import DeviceDataLoader
from utils import *
# from ImageClassificationBase import *
from ImageClassificationBase_old import * # TODO



class DecRefClassification(ImageClassificationBase):
    def __init__(self, num_framerates, num_resolutions, VELOCITY=True):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size = 3, padding = 1), # input 128x128, output 128x128 cause padding is 1
            nn.ReLU(), 
            nn.Conv2d(32,64, kernel_size = 3, stride = 1, padding = 1),  # Padding: 1 (which ensures the output has the same width and height as the input)
            nn.ReLU(), 
            nn.MaxPool2d(2,2), # the output should be 64
        
            nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1), 
            nn.ReLU(),
            nn.Conv2d(128 ,128, kernel_size = 3, stride = 1, padding = 1), 
            nn.ReLU(),  
            nn.MaxPool2d(2,2), 
            
            nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding = 1), 
            nn.ReLU(),
            nn.Conv2d(256,256, kernel_size = 3, stride = 1, padding = 1), 
            nn.ReLU(),
            nn.MaxPool2d(2,2), # output is 16, see official document to compute 
            
            nn.Flatten(),
            # output vector of size 1024, 65536 = 256 * 16 * 16 for 128x128, 16384 for 64x64, 256 * 32 * 32 for 256x256
            nn.Linear(256 * 16 * 16, 1024), 
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 32) # embedding of size 32
        )

        # self.batch_norm = nn.BatchNorm1d(32)  # Initialize BatchNorm
        # self.sigmoid = nn.Sigmoid()

        self.velocity = VELOCITY
        num_extra_features = 4 if self.velocity else 3 # 4
        self.fc_network = nn.Sequential(
            nn.Linear(32+num_extra_features, 16),  # fps, bitrate, velocity
            # nn.Linear(32 + 2, 16),  # Adjust input features to match your extended vector size
            nn.ReLU(),
            # nn.Linear(16, 8),
            # nn.ReLU(),
            # nn.Linear(8, 1)  # Adjust the output size based on your specific task
        )

        # Branch for resolution and framerate prediction
        self.fc_res = nn.Linear(16, num_resolutions)
        self.fc_fps = nn.Linear(16, num_framerates)

    def forward(self, images, fps, bitrate, resolution, velocity): # velocity=0
    # def forward(self, images, bitrate, velocity): # velocity=0
        """images, fps, image_bitrate, resolution, velocity"""
        # print(f'image {images.size()} ')
        features = self.network(images)    

        fps_resolution_bitrate = torch.stack([fps, bitrate, resolution, velocity], dim=1).float()  # TODO dim=1 Example way to combine fps and bitrate
        # fps_resolution_bitrate = torch.stack([fps, resolution, bitrate], dim=1).float()  # Example way to combine fps and bitrate
        # print(f'fps_resolution_bitrate {fps_resolution_bitrate.size()} {fps_resolution_bitrate}')

        combined = torch.cat((features, fps_resolution_bitrate), dim=1)
        # print(f'combined {combined.size()}\n {combined}')               

        x = self.fc_network(combined)
        # x = self.fc_network(fps_resolution_bitrate)
        res_out = self.fc_res(x) 
        fps_out = self.fc_fps(x) 
        # print(f'res_out {res_out.squeeze(1)} \n\n\n')
        return res_out, fps_out

    
    # with batch norm
    # def forward(self, images, fps, bitrate, resolution, velocity): # velocity=0
    #     """images, fps, image_bitrate, resolution, velocity"""
    #     # print(f'images {images.size()} {images.dtype}')
    #     unnorm_features = self.network(images)    
    #     normalized_features = self.batch_norm(unnorm_features)
    #     self.batch_norm.eval()
    #     normalized_features = self.sigmoid(normalized_features)  # Apply Sigmoid

    #     fps_resolution_bitrate = torch.stack([fps, bitrate, resolution, velocity], dim=1).float() # TODO dim=1 Example way to combine fps and bitrate
    #     # fps_resolution_bitrate = torch.stack([bitrate, velocity], dim=1).float()  # Example way to combine fps and bitrate
    #     # print(f'fps_resolution_bitrate {fps_resolution_bitrate.size()} {fps_resolution_bitrate}')

    #     combined = torch.cat((normalized_features, fps_resolution_bitrate), dim=1)
    #     # print(f'combined {combined.dtype}\n') # float16           

    #     x = self.fc_network(combined)
    #     res_out = self.fc_res(x) 
    #     fps_out = self.fc_fps(x) 
    #     # print(f'res_out {res_out.squeeze(1)} \n\n\n')
    #     return res_out, fps_out
