import os 
import torch
import torchvision
import numpy as np
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
from datetime import datetime

# from ConsecutivePatchDataset import ConsecutivePatchDataset
from DeviceDataLoader import DeviceDataLoader
from utils import *



class ImageClassificationBase(nn.Module):
    
    def training_step(self, images, metadata, VELOCITY=False):
        """
        metadata has fps, resolution, fps_target, resolution_target, image_bitrate, velocity/1000
        """
        fps = metadata[:, 0]
        resolution = metadata[:, 1]
        fps_target = metadata[:, 2]
        resolution_target = metadata[:, 3]
        image_bitrate = metadata[:, 4]
        velocity = metadata[:, 5]
        # print(f'images {images.size()}')
        # print(f'fps {fps}')
        # print(f'resolution {resolution}')
        # print(f'fps_target {fps_target}')
        # print(f'resolution_target {resolution_target}')
        # print(f'image_bitrate {image_bitrate}')
        # print(f'velocity {velocity}')
        # print(f'\n\n\n training step')
        
        if VELOCITY:
            print(f'forward with velocity {VELOCITY}')
            res_out, fps_out = self(images, fps, image_bitrate, resolution, velocity)  # NaturalSceneClassification.forward
        else:
            res_out, fps_out = self(images, fps, image_bitrate, resolution)

        # print(f'res_out {res_out.size()} \n {res_out}')
        # print(f'res_targets {resolution_target.size()} \n {resolution_target}\n')
        # print(f'fps_out {fps_out.size()} \n {fps_out}')
        total_loss = compute_weighted_loss(res_out, fps_out, resolution_target.long(), fps_target.long())
        # print(f'loss_res {loss_res}')
        # print(f'loss_fps {loss_fps}')
        return total_loss
    

    
    def validation_step(self, images, metadata):
        # print(f'\n\n\n validation step')
        fps = metadata[:, 0]
        resolution = metadata[:, 1]
        fps_target = metadata[:, 2]
        resolution_target = metadata[:, 3]
        image_bitrate = metadata[:, 4]
        velocity = metadata[:, 5]
        res_out, fps_out = self(images, fps, image_bitrate, resolution, velocity)  # NaturalSceneClassification.forward
        # print(f'training_step out {out.size()} \n {out.squeeze()}')
        # print(f'res_targets {res_targets}')
        # loss_fn_res = nn.CrossEntropyLoss()
        # loss_fn_fps = nn.CrossEntropyLoss()
    
        total_loss = compute_weighted_loss(res_out, fps_out, resolution_target.long(), fps_target.long())
        framerate_accuracy, resolution_accuracy, both_correct_accuracy = compute_accuracy(fps_out, res_out, fps_target, resolution_target)

        # Calculate accuracy, i.e.  proportion of the variance in the dependent variable that is predictable 
     
        # print(f'val_r2_score {val_r2_score}\n\n\n')
        return {'val_loss': total_loss.detach(), 'res_acc': resolution_accuracy, 'fps_acc': framerate_accuracy, 'both_acc': both_correct_accuracy} 
        


    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        # print(f'batch_losses \n {batch_losses}')
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_res_accs = [x['res_acc'] for x in outputs]
        batch_fps_accs = [x['fps_acc'] for x in outputs]
        batch_both_accs = [x['both_acc'] for x in outputs]
        # print(f'batch_accs \n {batch_accs}')

        epoch_res_acc = torch.stack(batch_res_accs).mean() # Combine accuracies
        epoch_fps_acc = torch.stack(batch_fps_accs).mean()
        epoch_both_acc = torch.stack(batch_both_accs).mean()
        # val_loss is average of batch loss, same for other losses
        return {'val_loss': epoch_loss.item(), 'val_res_acc': epoch_res_acc.item(), \
                'val_fps_acc': epoch_fps_acc.item(), 'val_both_acc': epoch_both_acc.item()}
    


    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_res_acc: {:.4f}, val_fps_acc: {:.4f}, val_both_acc: {:.4f}".format(
                epoch, result['train_loss'], result['val_loss'], result['val_res_acc'], result['val_fps_acc'], result['val_both_acc']))
        
