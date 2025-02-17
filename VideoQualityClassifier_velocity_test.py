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

# from VideoSinglePatchDataset_test import VideoSinglePatchDataset_test
# from VideoSinglePatchDataset import VideoSinglePatchDataset
from VideoMultiplePatchDataset import VideoMultiplePatchDataset
from VideoDualPatchDataset import VideoDualPatchDataset
from DeviceDataLoader import DeviceDataLoader
from utils import *
# from DecRefClassification_smaller import *
from DecRefClassification_dual_smaller import *
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


def process_test_outputs(result, fps_out, res_out):
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
    return result


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

        for batch in test_loader:
            # print(f'======================================== batch ========================================')
            images = batch["image"]
            fps = batch["fps"]
            bitrate = batch["bitrate"]
            resolution = batch["resolution"]
            velocity = batch["velocity"]
            res_targets = batch["res_targets"]
            fps_targets = batch["fps_targets"]
            path = batch["path"]

            unique_indices = {}
            # Iterate over the tensor and populate the dictionary
            for index, value in enumerate(bitrate.tolist()):
                if value not in unique_indices:
                    unique_indices[value] = index

            res_out, fps_out = model(images, fps, bitrate, resolution, velocity) 
            result = process_test_outputs(result, fps_out, res_out)
            # _, fps_preds = torch.max(fps_out, dim=1)
            # _, res_preds = torch.max(res_out, dim=1)
            # res_preds_all = res_preds if res_preds_all is None else torch.cat((res_preds_all, res_preds), dim=0)
            # fps_preds_all = fps_preds if fps_preds_all is None else torch.cat((fps_preds_all, fps_preds), dim=0)
            # res_targets_all = res_targets if res_targets_all is None else torch.cat((res_targets_all, res_targets), dim=0)
            # fps_targets_all = fps_targets if fps_targets_all is None else torch.cat((fps_targets_all, fps_targets), dim=0)
            # total_loss = compute_weighted_loss(res_out, fps_out, res_targets, fps_targets)
            # framerate_accuracy, resolution_accuracy, both_correct_accuracy, jod_preds, jod_targets = compute_accuracy(fps_out, res_out, fps_targets, res_targets, bitrate, path)
            # jod_preds_all = jod_preds if jod_preds_all is None else torch.cat((jod_preds_all, jod_preds), dim=0)
            # jod_targets_all = jod_targets if jod_targets_all is None else torch.cat((jod_targets_all, jod_targets), dim=0)
            
            # result['test_losses'].append(total_loss)
            # result['fps_acc'].append(framerate_accuracy)
            # result['res_acc'].append(resolution_accuracy)
            # result['both_acc'].append(both_correct_accuracy)

            # fps_idx = torch.argmax(fps_out, dim=1) # fps_out is a probabiblity, of size eg. (8, 10)
            # res_idx = torch.argmax(res_out, dim=1)
            # # print(f'fps_idx {fps_idx}')
            # fps_values = [fps[idx] for idx in fps_idx]
            # res_values = [resolution[idx] for idx in res_idx]
            # # print(f'fps_values {fps_values}')
            # res_values = torch.tensor(res_values)
            # fps_values = torch.tensor(fps_values)


        epoch_test_losses = torch.stack(result['test_losses']).mean() # Combine accuracies
        epoch_res_acc = torch.stack(result['res_acc']).mean() # Combine accuracies
        epoch_fps_acc = torch.stack(result['fps_acc']).mean()
        epoch_both_acc = torch.stack(result['both_acc']).mean()
        # epoch_jod_loss = sum(result['jod_loss']) / len(result['jod_loss'])
        # print(f'batch_jod_loss {batch_jod_loss}')
        return {'test_losses': epoch_test_losses.item(), 'res_acc': epoch_res_acc.item(), \
                'fps_acc': epoch_fps_acc.item(), 'both_acc': epoch_both_acc.item(),}, \
                res_preds_all, fps_preds_all, res_targets_all, fps_targets_all, jod_preds_all, jod_targets_all \
                # res_values, fps_values, unique_indices


# disable each parameter and test model performance 
# change model(images, fps, bitrate, velocity) in evaluate_test_data based on the used model
if __name__ == "__main__":
    TEST_EVAL = True
    TEST_UNSEEN_SCENE = True # True False
    
    model_pth_path = ""
    folder = 'ML/test_consecutive_patches64x64' # TODO change model size reference128x128
    if TEST_UNSEEN_SCENE:
        print(f'test on unseen scenes')
        data_test_directory = f'{VRRML}/ML/test_consecutive_patches64x64' # test_64x64 test_scenes64x64 test_scenes128x128
    else:
        data_test_directory = f'{VRRML}/{folder}/test' 
    
    if TEST_EVAL:
        model_pth_path = f'2025-02-09/consecutive_patches_with_velocity/classification.pth' # models/smaller_model/classification.pth

    MODEL_VELOCITY = True
    FPS = False # True False
    RESOLUTION = False

    batch_size = 128 
    patch_size = (64, 64) # change patch structure in DecRefClassification_test.py

    num_framerates, num_resolutions = 10, 5
    VALIDATION = True
    VELOCITY = True
    CHECKPOINT = False
    FRAMENUMBER = True # False True

    device = get_default_device()
    cuda  = device.type == 'cuda'

    if TEST_EVAL: # test data size 35756
        torch.manual_seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        test_dataset = VideoDualPatchDataset(directory=data_test_directory, min_bitrate=500, \
                                               max_bitrate=2000, patch_size=patch_size, VELOCITY=VELOCITY, \
                                                VALIDATION=VALIDATION, FRAMENUMBER=False) # len 27592
        print(f'\ntest_size {len(test_dataset)}, patch_size {patch_size}, batch_size {batch_size}\n')
        sample = test_dataset[0]
        # print('sample image has ', sample['fps'], 'fps,', sample['resolution'], ' resolution,', sample['bitrate'], 'bps')
        # print(f'sample velocity is {sample["velocity"]}') if VELOCITY else None
        # print(f'sample path is {sample["path"]}') if VELOCITY else None
        # model = DecRefClassification(num_framerates, num_resolutions, \
        #                              FPS=FPS, RESOLUTION=RESOLUTION, VELOCITY=MODEL_VELOCITY)
        model = DecRefClassification_dual(num_framerates, num_resolutions, VELOCITY=True)
        model.load_state_dict(torch.load(model_pth_path))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        test_dl = DataLoader(test_dataset, batch_size*2, shuffle = False, num_workers = 4, pin_memory = True)

        if device.type == 'cuda':
            print(f'Loading data to cuda...')
            test_dl = DeviceDataLoader(test_dl, device)
            to_device(model, device)

        print(f'model_path {model_pth_path}')
        print(f'TEST_UNSEEN_SCENE {TEST_UNSEEN_SCENE}')
        result, res_preds, fps_preds, res_targets, fps_targets, jod_preds, jod_targets = evaluate_test_data(model, test_dl)
        predicted_fps = torch.tensor([reverse_fps_map[int(pred)] for pred in fps_preds])
        target_fps = torch.tensor([reverse_fps_map[int(target)] for target in fps_targets])

        predicted_res = torch.tensor([reverse_res_map[int(pred)] for pred in res_preds])
        target_res = torch.tensor([reverse_res_map[int(target)] for target in res_targets])
        print(f'predicted_res {predicted_res}')
        print(f'target_res {target_res}')

        # Root Mean Square Error
        # https://help.pecan.ai/en/articles/6456388-model-performance-metrics-for-regression-models
        resolution_RMSE = compute_RMSE(predicted_res, target_res)
        fps_RMSE = compute_RMSE(predicted_fps, target_fps)
        jod_RMSE = compute_RMSE(jod_preds, jod_targets)

        # Root Mean Squared Percentage Error (RMSPE)
        resolution_RMSEP = relative_error_metric(predicted_res, target_res) 
        fps_RMSEP = relative_error_metric(predicted_fps, target_fps) 
        # resolution_RMSEP = geometric_mean_relative_error(predicted_res, target_res, "resolution") # compute_RMSEP(predicted_res, target_res)
        # fps_RMSEP = geometric_mean_relative_error(predicted_fps, target_fps, "fps") # compute_RMSEP(predicted_fps, target_fps)
        print(f"FPS: Root Mean Squared Error {fps_RMSE}, Root Mean Squared Percentage Error (RMSPE): {fps_RMSEP}%")
        print(f"Resolution: Root Mean Squared Error {resolution_RMSE}, Root Mean Squared Percentage Error (RMSPE): {resolution_RMSEP}%\n")
        print(f'jod rmse {jod_RMSE}')
        print(f'test result \n {result}\n')


    # SAVE_MODEL = True
    # SAVE_MODEL_HALF_WAY = True
    # START_TRAINING= False # True False
    # data_train_directory = f'{VRRML}/{folder}/train' # ML_smaller
    # data_val_directory = f'{VRRML}/{folder}/validation' 
    # if START_TRAINING:
        # num_epochs = 16
        # lr = 0.0003
        # opt_func = torch.optim.Adam
    #     model = DecRefClassification_test(num_framerates, num_resolutions, VELOCITY=VELOCITY)
    #     optimizer = opt_func(model.parameters(),lr)
    #     epochs = range(num_epochs)
    #     if CHECKPOINT:
    #         checkpoint = torch.load('2024-09-26/15_19/checkpoint10.pth')
    #         model.load_state_dict(checkpoint['model_state_dict'])
    #         optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #         # epoch = checkpoint['epoch']
    #         epoch = 10
    #         epochs = range(epoch+1, num_epochs)
    #         for state in optimizer.state.values():
    #             for k, v in state.items():
    #                 if isinstance(v, torch.Tensor):
    #                     state[k] = v.to(device)
    #     print(f'epochs {epochs}')
    #     train_dl = DataLoader(dataset, batch_size, shuffle = True, num_workers = 4, pin_memory = True)
    #     val_dl = DataLoader(val_dataset, batch_size*2, shuffle = True, num_workers = 4, pin_memory = True)
    #     if device.type == 'cuda':
    #         print(f'Loading data to cuda...')
    #         train_dl = DeviceDataLoader(train_dl, device)
    #         val_dl = DeviceDataLoader(val_dl, device)
    #         to_device(model, device)

    #     writer = SummaryWriter('runs/test_param')
    #     # fitting the model on training data and record the result after each epoch
    #     history = fit(epochs, model, train_dl, val_dl, optimizer, SAVE_MODEL=SAVE_MODEL, \
    #                     SAVE_HALFWAY=SAVE_MODEL_HALF_WAY, VELOCITY=VELOCITY, CHECKPOINT=CHECKPOINT)
    #     writer.close()
