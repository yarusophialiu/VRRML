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

from torchvision.transforms.functional import to_pil_image
from PIL import Image


import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from datetime import datetime

from DeviceDataLoader import DeviceDataLoader
from utils import *



class ImageClassificationBase(nn.Module):
    def training_step(self, batch, VELOCITY=True):
        images = batch["image"]
        fps = batch["fps"]
        bitrate = batch["bitrate"]
        resolution = batch["resolution"]
        velocity = batch["velocity"] # if VELOCITY else 0
        res_targets = batch["res_targets"]
        fps_targets = batch["fps_targets"]
        
        res_out, fps_out = self(images, fps, bitrate, resolution, velocity)  # NaturalSceneClassification.forward
        total_loss = compute_weighted_loss(res_out, fps_out, res_targets, fps_targets)
        # loss = F.mse_loss(out.squeeze(), labels.float()) # Calculate loss
        return total_loss
    
    def validation_step(self, batch):
        images = batch["image"]
        fps = batch["fps"]
        bitrate = batch["bitrate"]
        resolution = batch["resolution"]
        velocity = batch["velocity"]

        res_targets = batch["res_targets"]
        fps_targets = batch["fps_targets"]
        path = batch["path"]
        # TODO
        res_out, fps_out = self(images, fps, bitrate, resolution, velocity)  # NaturalSceneClassification.forward
        # res_out, fps_out = self(images, bitrate, velocity)  # NaturalSceneClassification.forward
        # print(f'training_step out {out.size()} \n {out.squeeze()}')
        # print(f'path {path}')
        _, fps_preds = torch.max(fps_out, dim=1)
        _, res_preds = torch.max(res_out, dim=1)
    
        total_loss = compute_weighted_loss(res_out, fps_out, res_targets, fps_targets)
        framerate_accuracy, resolution_accuracy, both_correct_accuracy, jod_preds, jod_targets = compute_accuracy(fps_preds, res_preds, fps_targets, res_targets, bitrate, path)
        # Calculate accuracy, i.e.  proportion of the variance in the dependent variable that is predictable 
        # print(f'validation_step fps_out {fps_out.size()} \n {fps_out}')
        _, fps_preds = torch.max(fps_out, dim=1)
        _, res_preds = torch.max(res_out, dim=1)
        predicted_fps = torch.tensor([reverse_fps_map[int(pred)] for pred in fps_preds])
        target_fps = torch.tensor([reverse_fps_map[int(target)] for target in fps_targets])

        predicted_res = torch.tensor([reverse_res_map[int(pred)] for pred in res_preds])
        target_res = torch.tensor([reverse_res_map[int(target)] for target in res_targets])
        resolution_RMSE = compute_RMSE(predicted_res, target_res)
        fps_RMSE = compute_RMSE(predicted_fps, target_fps)
        jod_RMSE = compute_RMSE(jod_preds, jod_targets)

        # Root Mean Squared Percentage Error (RMSPE)
        resolution_RMSEP = relative_error_metric(predicted_res, target_res) 
        fps_RMSEP = relative_error_metric(predicted_fps, target_fps) 

        # print(f'jod_RMSE {jod_RMSE}')
        return {'val_loss': total_loss.detach(), 
                'res_acc': resolution_accuracy, 
                'fps_acc': framerate_accuracy,
                'both_acc': both_correct_accuracy, 
                'jod_RMSE': jod_RMSE, 
                'resolution_RMSEP': resolution_RMSEP, \
                'fps_RMSEP': fps_RMSEP,
                "resolution_RMSE": resolution_RMSE,
                "fps_RMSE": fps_RMSE} # 'jod_loss': round(jod_loss, 3) 
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        # print(f'batch_losses \n {batch_losses}')
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_res_accs = [x['res_acc'] for x in outputs]
        batch_fps_accs = [x['fps_acc'] for x in outputs]
        batch_both_accs = [x['both_acc'] for x in outputs]
        batch_jod_RMSE = [x['jod_RMSE'] for x in outputs]
        batch_resolution_RMSEP = [x['resolution_RMSEP'] for x in outputs]
        batch_fps_RMSEP = [x['fps_RMSEP'] for x in outputs]
        batch_resolution_RMSE = [x['resolution_RMSE'] for x in outputs]
        batch_fps_RMSE = [x['fps_RMSE'] for x in outputs]
        # print(f'batch_jod_loss \n {batch_jod_loss}')

        epoch_res_acc = torch.stack(batch_res_accs).mean() # Combine accuracies
        epoch_fps_acc = torch.stack(batch_fps_accs).mean()
        epoch_both_acc = torch.stack(batch_both_accs).mean()
        # epoch_resolution_RMSEP = torch.stack(batch_resolution_RMSEP).mean()
        # epoch_jod_RMSE_torch = torch.stack(batch_jod_RMSE).mean()
        epoch_jod_RMSE = sum(batch_jod_RMSE) / len(batch_jod_RMSE)
        epoch_resolution_RMSEP = sum(batch_resolution_RMSEP) / len(batch_resolution_RMSEP)
        epoch_fps_RMSEP = sum(batch_fps_RMSEP) / len(batch_fps_RMSEP)
        epoch_resolution_RMSE = sum(batch_resolution_RMSE) / len(batch_resolution_RMSE)
        epoch_fps_RMSE = sum(batch_fps_RMSE) / len(batch_fps_RMSE)
        return {'val_loss': epoch_loss.item(), 'val_res_acc': epoch_res_acc.item(), \
                'val_fps_acc': epoch_fps_acc.item(), 'val_both_acc': epoch_both_acc.item(), \
                'jod_RMSE': epoch_jod_RMSE, \
                'resolution_RMSE': epoch_resolution_RMSE, 'resolution_RMSEP': epoch_resolution_RMSEP, \
                'fps_RMSE': epoch_fps_RMSE, 'fps_RMSEP': epoch_fps_RMSEP}
    
    def epoch_end(self, epoch, result):
        # val_acc_R2
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_res_acc: {:.4f}, val_fps_acc: {:.4f}, val_both_acc: {:.4f} jod_loss: {:.4f} \
              resolution_RMSE: {:.4f}, resolution_RMSEP: {:.4f}, fps_RMSE: {:.4f}, fps_RMSEP: {:.4f}".format(
              epoch, result['train_loss'], result['val_loss'], result['val_res_acc'], result['val_fps_acc'], result['val_both_acc'], result['jod_RMSE'], \
               result['resolution_RMSE'], result['resolution_RMSEP'], result['fps_RMSE'], result['fps_RMSEP'] ))
        
