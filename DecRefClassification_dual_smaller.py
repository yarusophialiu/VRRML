import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from ImageClassificationBase_old import * # TODO

def get_NN_with_BatchNorm(NUM_PATCHES):
    nnSequentialWithBatchNorm = nn.Sequential(
        nn.Conv2d(3 * NUM_PATCHES, 16 * NUM_PATCHES, kernel_size=3, padding=1),  # Reduce filters from 32 to 16
        nn.BatchNorm2d(16 * NUM_PATCHES),  # Added BatchNorm, inputoutput same shape
        nn.ReLU(), 

        nn.Conv2d(16 * NUM_PATCHES, 32 * NUM_PATCHES, kernel_size=3, padding=1), 
        nn.BatchNorm2d(32 * NUM_PATCHES),  # Added BatchNorm
        nn.ReLU(), 
        nn.MaxPool2d(2,2),  # Output: 64x64

        nn.Conv2d(32 * NUM_PATCHES, 64 * NUM_PATCHES, kernel_size=3, padding=1), 
        nn.BatchNorm2d(64 * NUM_PATCHES),  # Added BatchNorm
        nn.ReLU(),
        nn.Conv2d(64 * NUM_PATCHES, 64 * NUM_PATCHES, kernel_size=3, padding=1), 
        nn.BatchNorm2d(64 * NUM_PATCHES),  # Added BatchNorm
        nn.ReLU(),  
        nn.MaxPool2d(2,2),  # Output: 32x32

        nn.Conv2d(64 * NUM_PATCHES, 128 * NUM_PATCHES, kernel_size=3, padding=1), 
        nn.BatchNorm2d(128 * NUM_PATCHES),  # Added BatchNorm
        nn.ReLU(),
        nn.Conv2d(128 * NUM_PATCHES, 128 * NUM_PATCHES, kernel_size=3, padding=1), 
        nn.BatchNorm2d(128 * NUM_PATCHES),  # Added BatchNorm
        nn.ReLU(),
        nn.MaxPool2d(2,2)  

        # nn.Linear(64, 32), # embedding of size 32
        # nn.Sigmoid()  # Force output to [0,1]
    )
    return nnSequentialWithBatchNorm

class DecRefClassification_dual(ImageClassificationBase):
    def __init__(self, num_framerates=10, num_resolutions=5, NUM_PATCHES=2, FPS=True, RESOLUTION=True, VELOCITY=True):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3 * NUM_PATCHES, 16 * NUM_PATCHES, kernel_size=3, padding=1),  # Reduce filters from 32 to 16
            nn.ReLU(), 
            nn.Conv2d(16 * NUM_PATCHES, 32 * NUM_PATCHES, kernel_size=3, padding=1), 
            nn.ReLU(), 
            nn.MaxPool2d(2,2),  # Output: 64x64

            nn.Conv2d(32 * NUM_PATCHES, 64 * NUM_PATCHES, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.Conv2d(64 * NUM_PATCHES, 64 * NUM_PATCHES, kernel_size=3, padding=1), 
            nn.ReLU(),  
            nn.MaxPool2d(2,2),  # Output: 32x32

            nn.Conv2d(64 * NUM_PATCHES, 128 * NUM_PATCHES, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.Conv2d(128 * NUM_PATCHES, 128 * NUM_PATCHES, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(2,2),  # Output: 16x16
                    
            # reduce the spatial dimensions (height and width) of the feature map to 1x1, 
            # converting the entire 2D feature map into a single value per feature channel
            nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling
            nn.Flatten(),
            # output vector of size 1024, 65536 = 256 * 16 * 16 for 128x128, 16384 for 64x64, 256 * 32 * 32 for 256x256
            nn.Linear(128 * NUM_PATCHES, 64 * NUM_PATCHES),  # only 128 if adaptive avg pool enabled
            nn.ReLU(),
            nn.Linear(64 * NUM_PATCHES, 64),  # only 128 if adaptive avg pool enabled
            nn.ReLU(),
            nn.Linear(64, 32), # embedding of size 32
            nn.Sigmoid()  # Force output to [0,1]
        )

        self.fps = FPS
        self.resolution = RESOLUTION
        self.velocity = VELOCITY
        parameters = [FPS, RESOLUTION, VELOCITY]
        num_extra_features = sum(parameters) + 1
        print(f'num_extra_features {num_extra_features}')

        # num_extra_features = 3
        self.fc_network = nn.Sequential(
            nn.Linear(32+num_extra_features, 16),  # fps, bitrate, velocity
            nn.ReLU(),
        )
        self.fc_res = nn.Linear(16, num_resolutions)
        self.fc_fps = nn.Linear(16, num_framerates)

    
    def forward(self, images, fps, bitrate, resolution, velocity): # velocity=0
        """images, fps, image_bitrate, resolution, velocity"""
        features = self.network(images)
        selected_tensors = []

        if self.fps:
            selected_tensors.append(fps)
        selected_tensors.append(bitrate)
        if self.resolution:
            selected_tensors.append(resolution)
        if self.velocity:
            selected_tensors.append(velocity)

        # fps_resolution_bitrate2 = torch.stack([bitrate, resolution, velocity], dim=1).float()  # TODO dim=1 Example way to combine fps and bitrate
        fps_resolution_bitrate = torch.stack(selected_tensors, dim=1).float()
        # print(f'fps_resolution_bitrate {fps_resolution_bitrate.size()} \ {fps_resolution_bitrate}')

        combined = torch.cat((features, fps_resolution_bitrate), dim=1)
        x = self.fc_network(combined)
        res_out = self.fc_res(x) 
        fps_out = self.fc_fps(x) 
        return res_out, fps_out

    