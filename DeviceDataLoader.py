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
    
    def __init__(self, dl, device, FLOAT16=False):
        self.dl = dl
        self.device = device
        self.float16 = FLOAT16
    
    def __iter__(self): # called once for every batch
        """ Yield a batch of data after moving it to device"""
        # b is a dictionary with all infos of 32 images
        # {"image":..., "fps": [1.5110, dtype=torch.float64)], ...}
        try:
            for b in self.dl:
                if isinstance(b, dict):
                    # print(f'dictionary')
                    # batch = {k: to_device(v, self.device) for k, v in b.items()}
                    if self.float16:
                        batch = {k: to_device(v, self.device).half() if not isinstance(v, (str, list)) else v for k, v in b.items()}
                        if len(b) == 9:
                            print(f'b path {b["path"]}')
                    # elif self.float16 and len(b) == 9: # has path, which is used for computing jod loss in validation
                    #     for k, v in b.items():
                    #         batch = {}
                    #         if k == "path":

                    else:
                        batch = {k: to_device(v, self.device) if not isinstance(v, str) else v for k, v in b.items()}
                else:
                    # print(f'non dictionary')
                    
                    if self.float16:
                        batch = tuple(to_device(v, self.device).half() for v in b)
                    else:
                        batch = tuple(to_device(v, self.device) for v in b)
                yield batch
            
        except Exception as e:
                print(f"Corrupted image Error: {e}")
        # for b in self.dl:
            # b is the output of getitem in dataset, so b is a tuple of 5 elements
            # print(f'b {len(b)}')
            # print(f'b {b}')
            # if isinstance(b, dict):
            #     # print(f'dictionary')
            #     # batch = {k: to_device(v, self.device) for k, v in b.items()}
            #     if self.float16:
            #         batch = {k: to_device(v, self.device).half() if not isinstance(v, (str, list)) else v for k, v in b.items()}
            #         if len(b) == 9:
            #             print(f'b path {b["path"]}')
            #     # elif self.float16 and len(b) == 9: # has path, which is used for computing jod loss in validation
            #     #     for k, v in b.items():
            #     #         batch = {}
            #     #         if k == "path":

            #     else:
            #         batch = {k: to_device(v, self.device) if not isinstance(v, str) else v for k, v in b.items()}
            # else:
            #     # print(f'non dictionary')
                
            #     if self.float16:
            #         batch = tuple(to_device(v, self.device).half() for v in b)
            #     else:
            #         batch = tuple(to_device(v, self.device) for v in b)
            # yield batch
            
    def __len__(self):
        """ Number of batches """
        return len(self.dl)
