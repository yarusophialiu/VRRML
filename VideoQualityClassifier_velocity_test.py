import os 
import torch
import numpy as np
def set_manual_seed():
    torch.manual_seed(42)
    torch.cuda.manual_seed(42) # Ensure CUDA also uses the same seed
    torch.cuda.manual_seed_all(42)  # If using multi-GPU
    np.random.seed(42) # Ensure reproducibility with NumPy and Python's random
    torch.backends.cudnn.deterministic = True # Ensure deterministic behavior (may slow down training slightly)
    torch.backends.cudnn.benchmark = False
 

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
from datetime import datetime

# from VideoSinglePatchDataset_test import VideoSinglePatchDataset_test
from VideoMultiplePatchDataset import VideoMultiplePatchDataset
from VideoDualPatchDataset import VideoDualPatchDataset
from VideoSinglePatchDataset import VideoSinglePatchDataset
from DeviceDataLoader import DeviceDataLoader
from utils import *
from DecRefClassification_smaller import *
# from DecRefClassification_dual_smaller import *
from torch.utils.tensorboard import SummaryWriter

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# regressin, learn the curves
# https://docs.google.com/presentation/d/16yqaaq5zDZ5-S4394VLBUfxpNjM7nlpssqcShFkklec/edit#slide=id.g2c751bc0d9c_0_18


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    # print(f'eval outputs \n {outputs}')
    return model.validation_epoch_end(outputs) # get loss dictionary


# def process_test_outputs(result, fps_out, res_out, fps_targets, res_targets, \
#                          fps_preds_all, res_preds_all, bitrate, path):
#     _, fps_preds = torch.max(fps_out, dim=1)
#     _, res_preds = torch.max(res_out, dim=1)
#     res_preds_all = res_preds if res_preds_all is None else torch.cat((res_preds_all, res_preds), dim=0)
#     fps_preds_all = fps_preds if fps_preds_all is None else torch.cat((fps_preds_all, fps_preds), dim=0)
#     res_targets_all = res_targets if res_targets_all is None else torch.cat((res_targets_all, res_targets), dim=0)
#     fps_targets_all = fps_targets if fps_targets_all is None else torch.cat((fps_targets_all, fps_targets), dim=0)
#     total_loss = compute_weighted_loss(res_out, fps_out, res_targets, fps_targets)
#     framerate_accuracy, resolution_accuracy, both_correct_accuracy, jod_preds, jod_targets = compute_accuracy(fps_out, res_out, fps_targets, res_targets, bitrate, path)
#     jod_preds_all = jod_preds if jod_preds_all is None else torch.cat((jod_preds_all, jod_preds), dim=0)
#     jod_targets_all = jod_targets if jod_targets_all is None else torch.cat((jod_targets_all, jod_targets), dim=0)
    
#     result['test_losses'].append(total_loss)
#     result['fps_acc'].append(framerate_accuracy)
#     result['res_acc'].append(resolution_accuracy)
#     result['both_acc'].append(both_correct_accuracy)
#     return result, fps_preds_all, res_preds_all
def get_test_dataloader(ML_DATA_TYPE, batch_size, patch_size, device, parent_dir, FRAMENUMBER=True):
    velocity_type = 'frame-velocity'
    PATCH_SIZE = 64
    if 'invariant_consecutive' in parent_dir:
        print(f'parent_dir {parent_dir} VideoMultiplePatchDataset')
        data_test_folder = f'{ML_DATA_TYPE}/{velocity_type}/test_consecutive_{PATCH_SIZE}x{PATCH_SIZE}' 
        test_dataset = VideoMultiplePatchDataset(directory=f'{VRRML}/{data_test_folder}', min_bitrate=500, \
                                                max_bitrate=2000, patch_size=patch_size, VELOCITY=True, \
                                                VALIDATION=True, FRAMENUMBER=FRAMENUMBER) 
    elif 'invariant_random' in parent_dir:
        print(f'parent_dir {parent_dir} VideoMultiplePatchDataset')
        data_test_folder = f'{ML_DATA_TYPE}/{velocity_type}/test_random_{PATCH_SIZE}x{PATCH_SIZE}' 
        test_dataset = VideoMultiplePatchDataset(directory=f'{VRRML}/{data_test_folder}', min_bitrate=500, \
                                                max_bitrate=2000, patch_size=patch_size, VELOCITY=True, \
                                                VALIDATION=True, FRAMENUMBER=FRAMENUMBER)
    elif 'random' in parent_dir:
        print(f'parent_dir {parent_dir} VideoDualPatchDataset')
        data_test_folder = f'{ML_DATA_TYPE}/{velocity_type}/test_random_{PATCH_SIZE}x{PATCH_SIZE}' 
        test_dataset = VideoDualPatchDataset(directory=f'{VRRML}/{data_test_folder}', min_bitrate=500, \
                                                max_bitrate=2000, patch_size=patch_size, VELOCITY=True, \
                                                VALIDATION=True, FRAMENUMBER=FRAMENUMBER)
    elif 'consecutive' in parent_dir:
        print(f'parent_dir {parent_dir} VideoDualPatchDataset')
        data_test_folder = f'{ML_DATA_TYPE}/{velocity_type}/test_consecutive_{PATCH_SIZE}x{PATCH_SIZE}' 
        test_dataset = VideoDualPatchDataset(directory=f'{VRRML}/{data_test_folder}', min_bitrate=500, \
                                                max_bitrate=2000, patch_size=patch_size, VELOCITY=True, \
                                                VALIDATION=True, FRAMENUMBER=FRAMENUMBER)
    else:
        print(f'parent_dir {parent_dir} VideoSinglePatchDataset')
        data_test_folder = f'{ML_DATA_TYPE}/{velocity_type}/test_single_{PATCH_SIZE}x{PATCH_SIZE}' 
        test_dataset = VideoSinglePatchDataset(directory=f'{VRRML}/{data_test_folder}', min_bitrate=500, \
                                                max_bitrate=2000, patch_size=patch_size, VELOCITY=True, \
                                                VALIDATION=True, FRAMENUMBER=FRAMENUMBER)
    print(f'\nTest data size {len(test_dataset)}, patch_size {patch_size}, batch_size {batch_size}')
    print(f'data_test_folder {data_test_folder}\n')
    sample = test_dataset[0]
    print('sample image has ', sample['fps'], 'fps,', sample['resolution'], ' resolution,', sample['bitrate'], 'bps')
    print(f'sample velocity is {sample["velocity"]}') if VELOCITY else None
    print(f'sample path is {sample["path"]}') if VELOCITY else None
    test_dl = DataLoader(test_dataset, batch_size*2, shuffle = False, num_workers = 4, pin_memory = True)
    test_dl = DeviceDataLoader(test_dl, device) 
    return test_dl


# from videoqualityclassifier_velocity.py
def evaluate_test_data(model, test_loader, training_mode):
    fps = [30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
    resolution = [360, 480, 720, 864, 1080]

    model.eval()
    with torch.no_grad():  # Ensure gradients are not computed
        result = {'jod_loss': [], 'test_losses': [], 'res_acc': [], 'fps_acc': [], 'both_acc': [],}
        res_preds_all = None
        fps_preds_all = None
        jod_preds_all = None
        res_targets_all = None
        fps_targets_all = None
        jod_targets_all = None
        bitrate_all = []
        # path_all = None
        path_all = []

        if 'invariant' in training_mode:
            for batch in test_loader:
                images1 = batch["image1"]
                images2 = batch["image2"]
                fps = batch["fps"]
                bitrate = batch["bitrate"]
                resolution = batch["resolution"]
                velocity = batch["velocity"]
                res_targets = batch["res_targets"]
                fps_targets = batch["fps_targets"]
                path = batch["path"]

                # unique_indices = {}
                # # Iterate over the tensor and populate the dictionary
                # for index, value in enumerate(bitrate.tolist()):
                #     if value not in unique_indices:
                #         unique_indices[value] = index

                res_out, fps_out = model(images1, images2, fps, bitrate, resolution, velocity) 
                _, fps_preds = torch.max(fps_out, dim=1)
                _, res_preds = torch.max(res_out, dim=1)
                res_preds_all = res_preds if res_preds_all is None else torch.cat((res_preds_all, res_preds), dim=0)
                fps_preds_all = fps_preds if fps_preds_all is None else torch.cat((fps_preds_all, fps_preds), dim=0)
                res_targets_all = res_targets if res_targets_all is None else torch.cat((res_targets_all, res_targets), dim=0)
                fps_targets_all = fps_targets if fps_targets_all is None else torch.cat((fps_targets_all, fps_targets), dim=0)
                total_loss = compute_weighted_loss(res_out, fps_out, res_targets, fps_targets)
                framerate_accuracy, resolution_accuracy, both_correct_accuracy, jod_preds, jod_targets = compute_accuracy(fps_out, res_out, fps_targets, res_targets, bitrate, path)
                jod_preds_all = jod_preds if jod_preds_all is None else torch.cat((jod_preds_all, jod_preds), dim=0)
                jod_targets_all = jod_targets if jod_targets_all is None else torch.cat((jod_targets_all, jod_targets), dim=0)
                
                result['test_losses'].append(total_loss)
                result['fps_acc'].append(framerate_accuracy)
                result['res_acc'].append(resolution_accuracy)
                result['both_acc'].append(both_correct_accuracy)
        else:
            for batch in test_loader:
                images = batch["image"]
                fps = batch["fps"]
                bitrate = batch["bitrate"] # tensor
                resolution = batch["resolution"]
                velocity = batch["velocity"]
                res_targets = batch["res_targets"]
                fps_targets = batch["fps_targets"]
                path = batch["path"] # list
                bitrate_all.extend(bitrate.tolist())
                path_all.extend(path)
                # path_all = path if path_all is None else torch.cat((path_all, path), dim=0)

                res_out, fps_out = model(images, fps, bitrate, resolution, velocity) 
                # print(f'fps_out {fps_out.shape}, res_out {res_out.shape}')
                # result = process_test_outputs(result, fps_out, res_out, fps_targets, res_targets, bitrate, path)
                _, fps_preds = torch.max(fps_out, dim=1)
                _, res_preds = torch.max(res_out, dim=1)
                res_preds_all = res_preds if res_preds_all is None else torch.cat((res_preds_all, res_preds), dim=0)
                fps_preds_all = fps_preds if fps_preds_all is None else torch.cat((fps_preds_all, fps_preds), dim=0)
                res_targets_all = res_targets if res_targets_all is None else torch.cat((res_targets_all, res_targets), dim=0)
                fps_targets_all = fps_targets if fps_targets_all is None else torch.cat((fps_targets_all, fps_targets), dim=0)
                
                total_loss = compute_weighted_loss(res_out, fps_out, res_targets, fps_targets)

        framerate_accuracy, resolution_accuracy, both_correct_accuracy, jod_preds_all, jod_targets_all = compute_accuracy(fps_preds_all, res_preds_all, fps_targets_all, res_targets_all, bitrate_all, path_all)
        # jod_preds_all = jod_preds if jod_preds_all is None else torch.cat((jod_preds_all, jod_preds), dim=0)
        # jod_targets_all = jod_targets if jod_targets_all is None else torch.cat((jod_targets_all, jod_targets), dim=0)

        result['test_losses'].append(total_loss)
        epoch_test_losses = torch.stack(result['test_losses']).mean() # Combine accuracies

        # # epoch_jod_loss = sum(result['jod_loss']) / len(result['jod_loss'])
        # # print(f'batch_jod_loss {batch_jod_loss}')
        return {
            'test_losses': round(epoch_test_losses.item(), 4), 
            'res_acc': round(resolution_accuracy.item(), 4), 
            'fps_acc': round(framerate_accuracy.item(), 4), 
            'both_acc': round(both_correct_accuracy.item(), 4)
        }, res_preds_all, fps_preds_all, res_targets_all, fps_targets_all, jod_preds_all, jod_targets_all
                # res_values, fps_values, unique_indices


# disable each parameter and test model performance 
# change model(images, fps, bitrate, velocity) in evaluate_test_data based on the used model
if __name__ == "__main__":
    TEST_EVAL = True
    
    ML_DATA_TYPE = 'ML'
    data_test_directory = f'{VRRML}/{ML_DATA_TYPE}/frame-velocity/test_single_64x64' 
    model_parent_folder = 'no_fps_no_resolution_no_velocity_20_43' # -frame-dropjod-sigmoid
    model_pth_path = f'2025-03-02/{model_parent_folder}/classification.pth' 

    training_mode = 'no_fps_no_resolution_no_velocity'
    FPS = False # True False
    RESOLUTION = False
    MODEL_VELOCITY = False
    batch_size = 128 * 20
    patch_size = (64, 64)

    num_framerates, num_resolutions = 10, 5
    VALIDATION = True
    VELOCITY = True
    CHECKPOINT = True
    FRAMENUMBER = True # False True

    device = get_default_device()

    if TEST_EVAL: # test data size 35756
        set_manual_seed()
        model = DecRefClassification(num_framerates, num_resolutions, \
                                          FPS=FPS, RESOLUTION=RESOLUTION, VELOCITY=MODEL_VELOCITY)
        model.load_state_dict(torch.load(model_pth_path))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # test_dl = DataLoader(test_dataset, batch_size*2, shuffle = False, num_workers = 4, pin_memory = True)
        test_dl = get_test_dataloader(ML_DATA_TYPE, 128, patch_size, device, model_parent_folder, FRAMENUMBER=True)

        if device.type == 'cuda':
            print(f'Loading data to cuda...')
            test_dl = DeviceDataLoader(test_dl, device)
            to_device(model, device)

        print(f'model_path {model_pth_path}')
        result, res_preds, fps_preds, res_targets, fps_targets, jod_preds, jod_targets = evaluate_test_data(model, test_dl, training_mode)
        predicted_fps = torch.tensor([reverse_fps_map[int(pred)] for pred in fps_preds])
        target_fps = torch.tensor([reverse_fps_map[int(target)] for target in fps_targets])

        predicted_res = torch.tensor([reverse_res_map[int(pred)] for pred in res_preds])
        target_res = torch.tensor([reverse_res_map[int(target)] for target in res_targets])
        print(f'predicted_res {predicted_res}')
        print(f'target_res {target_res}')
        print(f'predicted_fps {predicted_fps}')
        print(f'target_fps {target_fps}')

        # inference_output_dir = 'inference_outputs'
        # with open(f"{inference_output_dir}/predicted_res_{training_mode}.py", "w") as f:
        #     f.write(f"predicted_res = {predicted_res.tolist()}\n")
        # with open(f"{inference_output_dir}/target_res_{training_mode}.py", "w") as f:
        #     f.write(f"target_res = {target_res.tolist()}\n")
        # with open(f"{inference_output_dir}/predicted_fps_{training_mode}.py", "w") as f:
        #     f.write(f"predicted_fps = {predicted_fps.tolist()}\n")
        # with open(f"{inference_output_dir}/target_fps_{training_mode}.py", "w") as f:
        #     f.write(f"target_fps = {target_fps.tolist()}\n")


        # # Root Mean Square Error
        # # https://help.pecan.ai/en/articles/6456388-model-performance-metrics-for-regression-models
        resolution_RMSE = compute_RMSE(predicted_res, target_res)
        fps_RMSE = compute_RMSE(predicted_fps, target_fps)
        # jod_RMSE = compute_RMSE(jod_preds, jod_targets)

        # Root Mean Squared Percentage Error (RMSPE)
        resolution_RMSEP = relative_error_metric(predicted_res, target_res) 
        fps_RMSEP = relative_error_metric(predicted_fps, target_fps) 
        # resolution_RMSEP = geometric_mean_relative_error(predicted_res, target_res, "resolution") # compute_RMSEP(predicted_res, target_res)
        # fps_RMSEP = geometric_mean_relative_error(predicted_fps, target_fps, "fps") # compute_RMSEP(predicted_fps, target_fps)
        print(f"FPS: Root Mean Squared Error {fps_RMSE}, Root Mean Squared Percentage Error (RMSPE): {fps_RMSEP}%")
        print(f"Resolution: Root Mean Squared Error {resolution_RMSE}, Root Mean Squared Percentage Error (RMSPE): {resolution_RMSEP}%\n")
        # print(f'jod rmse {jod_RMSE}')
        print(f'test result \n {result}\n')

