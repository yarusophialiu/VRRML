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

from DeviceDataLoader import DeviceDataLoader
from utils import *



class ImageClassificationBase(nn.Module):
    def training_step(self, batch, VELOCITY=True):
        # print(f'================== training_step ==================')
        images = batch["image"]
        fps = batch["fps"]
        bitrate = batch["bitrate"]
        resolution = batch["resolution"]
        velocity = batch["velocity"] # if VELOCITY else 0
        # print(f'velocity {VELOCITY} {velocity}')

        res_targets = batch["res_targets"]
        fps_targets = batch["fps_targets"]
        # print(f'\n\n\n training step')
        
        if VELOCITY:
            res_out, fps_out = self(images, fps, bitrate, resolution, velocity)  # images
        else:
            res_out, fps_out = self(images, fps, bitrate, resolution)

        # print(f'res_out {res_out.size()} \n {res_out}')
        # print(f'res_targets {res_targets.size()} \n {res_targets}')
        # print(f'fps_out {fps_out.size()} \n {fps_out}')
        total_loss = compute_weighted_loss(res_out, fps_out, res_targets, fps_targets)
        # loss = F.mse_loss(out.squeeze(), labels.float()) # Calculate loss
        # print(f'loss_res {loss_res}')
        # print(f'loss_fps {loss_fps}')

        return total_loss
    
    def validation_step(self, batch):
        # print(f'\n\n\n validation step')
        images = batch["image"]
        fps = batch["fps"]
        bitrate = batch["bitrate"]
        resolution = batch["resolution"]
        velocity = batch["velocity"]

        res_targets = batch["res_targets"]
        fps_targets = batch["fps_targets"]
        path = batch["path"]
        res_out, fps_out = self(images, fps, bitrate, resolution, velocity)  # NaturalSceneClassification.forward
        # print(f'training_step out {out.size()} \n {out.squeeze()}')
        # print(f'res_targets {res_targets}')
    
        total_loss = compute_weighted_loss(res_out, fps_out, res_targets, fps_targets)
        framerate_accuracy, resolution_accuracy, both_correct_accuracy, jod_loss = compute_accuracy(fps_out, res_out, fps_targets, res_targets, bitrate, path)
        # Calculate accuracy, i.e.  proportion of the variance in the dependent variable that is predictable 
        # print(f'validation_step fps_out {fps_out.size()} \n {fps_out}')
     
        return {'val_loss': total_loss.detach(), 'res_acc': resolution_accuracy, 'fps_acc': framerate_accuracy, \
                'both_acc': both_correct_accuracy, 'jod_loss': round(jod_loss, 3)} 
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        # print(f'batch_losses \n {batch_losses}')
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_res_accs = [x['res_acc'] for x in outputs]
        batch_fps_accs = [x['fps_acc'] for x in outputs]
        batch_both_accs = [x['both_acc'] for x in outputs]
        batch_jod_loss = [x['jod_loss'] for x in outputs]
        # print(f'batch_jod_loss \n {batch_jod_loss}')

        epoch_res_acc = torch.stack(batch_res_accs).mean() # Combine accuracies
        epoch_fps_acc = torch.stack(batch_fps_accs).mean()
        epoch_both_acc = torch.stack(batch_both_accs).mean()
        # epoch_jod_loss = torch.stack(batch_jod_loss).mean()
        epoch_jod_loss = sum(batch_jod_loss) / len(batch_jod_loss)
        # print(f'batch_jod_loss {batch_jod_loss}')
        return {'val_loss': epoch_loss.item(), 'val_res_acc': epoch_res_acc.item(), \
                'val_fps_acc': epoch_fps_acc.item(), 'val_both_acc': epoch_both_acc.item(), 'jod_loss': epoch_jod_loss}
    
    def epoch_end(self, epoch, result):
        # val_acc_R2
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_res_acc: {:.4f}, val_fps_acc: {:.4f}, val_both_acc: {:.4f} jod_loss: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_res_acc'], result['val_fps_acc'], result['val_both_acc'], result['jod_loss']))
        
