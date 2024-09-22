import os 
import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from torchvision.utils import make_grid
from torch.utils.data import Dataset


import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from utils import *

class DeviceDataLoader():
    """ Wrap a dataloader to move data to a device """
    
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
    
    def __iter__(self): # called once for every batch
        """ Yield a batch of data after moving it to device"""
        # b is a dictionary with all infos of 32 images
        # print(f'self.dl \n {self.dl}')
        for b in self.dl: # a list of 3 dictionary or 1 dictionary with 5 entries
            # b is the output of getitem in dataset, so b is a tuple of 5 elements
            # images_tensor, metadata, fps_target, resolution_target, image_bitrate
            # print(f'b {len(b)}')
            if isinstance(b, dict):
                # batch = {k: to_device(v, self.device) for k, v in b.items()}
                batch = {k: to_device(v, self.device) if not isinstance(v, str) else v for k, v in b.items()}
            else:
                batch = tuple(to_device(v, self.device) for v in b)
            yield batch
            
    def __len__(self):
        """ Number of batches """
        return len(self.dl)
