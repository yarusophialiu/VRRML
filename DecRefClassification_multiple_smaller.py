import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from MultipleImageClassificationBase import * 


# model takes 2 patches as input, and output 2 embeddings
# then apply order-invariant pooling to the 2 embedings 
class DecRefClassification_multiple(MultiplemageClassificationBase):
    def __init__(self, num_framerates=10, num_resolutions=5, FPS=True, RESOLUTION=True, VELOCITY=True):
        super().__init__()
        
        # Feature extraction network (shared CNN)
        self.network = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(), 
            nn.Conv2d(16, 32, kernel_size=3, padding=1), 
            nn.ReLU(), 
            nn.MaxPool2d(2, 2),  

            nn.Conv2d(32, 64, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), 
            nn.ReLU(),  
            nn.MaxPool2d(2, 2),  

            nn.Conv2d(64, 128, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(2, 2), 
                        nn.AdaptiveAvgPool2d((1, 1)),  
            nn.Flatten(),  
            nn.Linear(128, 64),  
            nn.ReLU(),
            nn.Linear(64, 32)  
        )

        self.fps = FPS
        self.resolution = RESOLUTION
        self.velocity = VELOCITY
        
        parameters = [FPS, RESOLUTION, VELOCITY]
        num_extra_features = sum(parameters) + 1  # bitrate always included

        self.fc_network = nn.Sequential(
            nn.Linear(32 + num_extra_features, 16),
            nn.ReLU(),
        )
        self.fc_res = nn.Linear(16, num_resolutions)
        self.fc_fps = nn.Linear(16, num_framerates)
    
    
    def forward(self, images1, images2, fps, bitrate, resolution, velocity):
        """
        Takes two image patches and outputs an order-invariant embedding.
        """
        batch_size = images1.shape[0]
        stacked_images = torch.cat((images1, images2), dim=0)  # Shape: (2B, C, H, W)
        # Forward pass through shared CNN
        all_embeddings = self.network(stacked_images)  # Shape: (2B, 32)
        features1, features2 = all_embeddings[:batch_size], all_embeddings[batch_size:]

        # Order-invariant pooling
        pooled_features = (features1 + features2) / 2  # Mean pooling
        # pooled_features = torch.max(features1, features2)  # Max pooling (alternative)

        # Prepare extra metadata
        selected_tensors = []
        if self.fps:
            selected_tensors.append(fps)
        selected_tensors.append(bitrate)
        if self.resolution:
            selected_tensors.append(resolution)
        if self.velocity:
            selected_tensors.append(velocity)

        extra_features = torch.stack(selected_tensors, dim=1).float()  # (B, num_extra_features)
        # print(f'images1, images2 {images1.size()} {images2.size()}')
        # print(f'extra_features {extra_features.size()}')

        # Concatenate pooled embeddings with metadata
        combined = torch.cat((pooled_features, extra_features), dim=1)  # Shape: (B, 32 + num_extra_features)

        # Fully connected layers
        x = self.fc_network(combined)
        res_out = self.fc_res(x) 
        fps_out = self.fc_fps(x) 
        
        return res_out, fps_out