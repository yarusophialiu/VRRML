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

from VideoSinglePatchDataset import VideoSinglePatchDataset
from DeviceDataLoader import DeviceDataLoader
from utils import *
from DecRefClassification import *
from torch.utils.tensorboard import SummaryWriter


def display_img(img,label):
    print(f"Label : {dataset.classes[label]}")
    plt.imshow(img.permute(1,2,0))
    plt.show()

def show_batch(dl):
    """Plot images grid of single batch"""
    for batch in dl: # dl calls __getitem__
        images = batch["image"]
        print(f'images {images.dtype}')
        labels = batch["label"]
        print(f'Labels: {labels}')
        fig,ax = plt.subplots(figsize = (16,12))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(make_grid(images,nrow=16).permute(1,2,0))
        plt.show()
        break




def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    # print(f'eval outputs \n {outputs}')
    return model.validation_epoch_end(outputs) # get loss dictionary


def evaluate_test_data(model, test_loader):
    model.eval()
    with torch.no_grad():  # Ensure gradients are not computed
        for batch in test_loader:
            images = batch["image"]
            print(f'images {images.size()}')
            fps = batch["fps"]
            bitrate = batch["bitrate"]
            resolution = batch["resolution"]
            velocity = batch["velocity"]
            res_targets = batch["res_targets"]
            fps_targets = batch["fps_targets"]
            path = batch["path"]
            framenumber = batch["framenumber"]
            print(f'fps {fps.size()}')
            print(f'framenumber {framenumber.size()}')

            unique_indices = {}
            # Iterate over the tensor and populate the dictionary
            for index, value in enumerate(bitrate.tolist()):
                if value not in unique_indices:
                    unique_indices[value] = index

            res_out, fps_out = model(images, fps, bitrate, resolution, velocity)  # NaturalSceneClassification.forward
            # print(f'training_step out {out.size()} \n {out.squeeze()}')
            # print(f'labels out {labels}')
            total_loss = compute_weighted_loss(res_out, fps_out, res_targets, fps_targets)
            framerate_accuracy, resolution_accuracy, both_correct_accuracy, jod_loss = compute_accuracy(fps_out, res_out, fps_targets, res_targets, bitrate, path)
            # result = {'val_loss': total_loss.detach(), 'res_acc': resolution_accuracy, 'fps_acc': framerate_accuracy, \
            #     'both_acc': both_correct_accuracy, 'jod_loss': round(jod_loss, 3)} 
            # print(f'result {result}')
            fps = [30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
            resolution = [360, 480, 720, 864, 1080]

            fps_idx = torch.argmax(fps_out, dim=1) # fps_out is a probabiblity, of size eg. (8, 10)
            res_idx = torch.argmax(res_out, dim=1)
            # print(f'fps_idx {fps_idx}')

            fps_values = [fps[idx] for idx in fps_idx]
            res_values = [resolution[idx] for idx in res_idx]
            # print(f'fps_values {fps_values}')
            res_values = torch.tensor(res_values)
            fps_values = torch.tensor(fps_values)
            original_framenumber = torch.round(framenumber * 276)
            return {'val_loss': total_loss.detach(), 'res_acc': resolution_accuracy, 'fps_acc': framerate_accuracy, \
                    'both_acc': both_correct_accuracy, 'jod_loss': round(jod_loss, 3)}, res_out, fps_out, res_targets, fps_targets, \
                        res_values, fps_values, original_framenumber, unique_indices



if __name__ == "__main__":
    PLOT_TEST_RESULT = False
    SAVE_PLOT = True
    TEST_UNSEEN_SCENE = False # True
    
    model_pth_path = f'models/patch128-256/patch128_batch128.pth' # patch128_batch128 patch256_batch64
    folder = 'ML_smaller/reference128x128' # TODO change model size reference128x128
    if TEST_UNSEEN_SCENE:
        print(f'test on unseen scenes')
        data_test_directory = f'{VRRML}/ML/test_scenes128x128'  # f'{VRRML}/ML/test_scenes128x128' f'{VRR_Patches}/reference_living_room_128x128'
    else:
        data_test_directory = f'{VRR_Patches}/reference_bistro_128x128' # f'{VRRML}/{folder}/test_demo'

    opt_func = torch.optim.Adam
    batch_size = 16 # TODO
    patch_size = (128, 128) # TODO, change patch structure in DecRefClassification.py
    num_framerates, num_resolutions = 10, 5
    VELOCITY = True
    VALIDATION = True

    device = get_default_device()
    print(f'device {device.type}')
    cuda  = device.type == 'cuda'

    test_dataset = VideoSinglePatchDataset(directory=data_test_directory, min_bitrate=500, max_bitrate=2000, patch_size=patch_size, \
                                           VELOCITY=VELOCITY, VALIDATION=VALIDATION, FRAMENUMBER=True) # len 27592
    print(f'\ntest_size {len(test_dataset)}, patch_size {patch_size}, batch_size {batch_size}\n')
    sample = test_dataset[0]
    print('sample image has ', sample['fps'], 'fps,', sample['resolution'], ' resolution,', sample['bitrate'], 'bps')
    print(f'sample velocity is {sample["velocity"]}, framenumber {sample["framenumber"]}') if VELOCITY else None
    print(f'sample path is {sample["path"]}') if VELOCITY else None


    model = DecRefClassification(num_framerates, num_resolutions, VELOCITY=VELOCITY)
    model.load_state_dict(torch.load(model_pth_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dl = DataLoader(test_dataset, batch_size, shuffle = True, num_workers = 4, pin_memory = True)

    if device.type == 'cuda':
        print(f'Loading data to cuda...')
        test_dl = DeviceDataLoader(test_dl, device)
        to_device(model, device)

    print(f'model_path {model_pth_path}')
    result, res_out, fps_out, res_targets, fps_targets, \
                    res_values, fps_values, framenumber, unique_indices = evaluate_test_data(model, test_dl)
    _, fps_preds = torch.max(fps_out, dim=1)
    _, res_preds = torch.max(res_out, dim=1)
    # print(f'res_preds {res_preds}')
    # print(f'res_targets {res_targets}\n')
    
    reverse_fps_map = {v: k for k, v in fps_map.items()}
    reverse_res_map = {v: k for k, v in res_map.items()}

    # Convert the predicted and target indices to actual values
    predicted_fps = torch.tensor([reverse_fps_map[int(pred)] for pred in fps_preds])
    target_fps = torch.tensor([reverse_fps_map[int(target)] for target in fps_targets])

    predicted_res = torch.tensor([reverse_res_map[int(pred)] for pred in res_preds])
    target_res = torch.tensor([reverse_res_map[int(target)] for target in res_targets])
    print(f'predicted_fps {predicted_fps}')
    print(f'predicted_res {predicted_res}\n')
    print(f'target_fps {target_fps}')
    print(f'target_res {target_res}\n')
    print(f'framenumber {framenumber}\n')

    absolute_errors_fps = torch.abs(predicted_fps - target_fps)
    absolute_errors_res = torch.abs(predicted_res - target_res)
    # print(f'absolute_errors_res {absolute_errors_res}\n')
    expected_error_fps = torch.mean(absolute_errors_fps.float())
    expected_error_res = torch.mean(absolute_errors_res.float())
    # print(f"Expected Error vertical resolution (Mean Absolute Error): {expected_error_res.item()}\n")
    # print(f"Expected Error fps (Mean Absolute Error): {expected_error_fps.item()}")

    # Compute the absolute percentage error
    # print(f'predicted_fps {predicted_fps}')
    percentage_errors_fps = torch.abs((predicted_fps - target_fps) / target_fps) * 100
    percentage_errors_res = torch.abs((predicted_res - target_res) / target_res) * 100
    # print(f"percentage_errors_fps: {percentage_errors_fps}") 
    # print(f"percentage_errors_res: {percentage_errors_res}\n") 

    # Compute the mean absolute percentage error (MAPE)
    mape_fps = torch.mean(percentage_errors_fps.float())
    mape_res = torch.mean(percentage_errors_res.float())
    print(f"FPS: Mean Absolute Error {expected_error_fps.item()}, Mean Absolute Percentage Error (MAPE): {round(mape_fps.item(), 3)}%")
    print(f"Resolution: Mean Absolute Error {expected_error_res.item()}, Mean Absolute Percentage Error (MAPE): {round(mape_res.item(), 3)}%\n")

    print(f'test result \n {result}\n')

    