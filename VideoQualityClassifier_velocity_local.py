import os 
import time
import torch
import torchvision
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from torchvision.utils import make_grid
from torch.utils.data import Dataset
from EarlyStopping import EarlyStopping

import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from datetime import datetime

from VideoSinglePatchDataset import VideoSinglePatchDataset
from VideoDualPatchDataset import VideoDualPatchDataset
from DeviceDataLoader import DeviceDataLoader
from utils import *
from DecRefClassification_smaller import *
from DecRefClassification_dual_smaller import *
from VideoQualityClassifier_velocity_test import evaluate_test_data
from torch.utils.tensorboard import SummaryWriter

# regressin, learn the curves
# https://docs.google.com/presentation/d/16yqaaq5zDZ5-S4394VLBUfxpNjM7nlpssqcShFkklec/edit#slide=id.g2c751bc0d9c_0_18

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    # print(f'eval outputs \n {outputs}')
    return model.validation_epoch_end(outputs) # get loss dictionary


def fit(epochs, model, train_loader, val_loader, optimizer, training_mode, \
                SAVE_MODEL=False, SAVE_HALFWAY=False, VELOCITY=False, CHECKPOINT=False):
    history = []
    model_path = ""
    # if SAVE_MODEL or SAVE_HALFWAY:
    now = datetime.now()
    dir_pth = now.strftime("%Y-%m-%d")
    os.makedirs(dir_pth, exist_ok=True)
    hrmin = now.strftime("%H_%M")
    model_path = os.path.join(dir_pth, f'{training_mode}_{hrmin}')
    os.makedirs(model_path, exist_ok=True)
    
    early_stopping = None
    early_stopping = EarlyStopping(patience=PATIENCE, delta=0.001, path=f"{model_path}")
    early_stopping_triggered = False
    
    # now = datetime.now()
    # dir_pth = now.strftime("%Y-%m-%d")
    # os.makedirs(dir_pth, exist_ok=True)
    # runs_path = os.path.join(dir_pth, now.strftime("%H_%M"))
    # os.makedirs(f'runs/{runs_path}', exist_ok=True)
    writer = SummaryWriter(model_path)  # f'runs/{runs_path}'
    # an epoch is one pass through the entire dataset
    for epoch in epochs:
        print(f'================================ epoch {epoch} ================================')
        model.train()
        train_losses = []
        count = 0
        running_loss = 0.0
        # for each batch, compute gradients for every data, evaluate after each batch
        # requests an iterator from DeviceDataLoader, i.e. __iter__ function

        # are batch1 and batch2 not overlap? NO when you use DataLoader with shuffle=True, 
        # batches will not overlap during training
        for batch in train_loader: # batch is a dictionary with 32 images information, e.g. 'fps': [70, 80, ..., 150]
            count += 1
            loss = model.training_step(batch, VELOCITY=VELOCITY) # model
            train_losses.append(loss)
            running_loss += loss.item()

            # computes the gradient of the loss with respect to the model parameters
            # part of the backpropagation algorithm, which is how the neural network learns
            loss.backward()  
            optimizer.step() # update the model parameters based on the gradients calculated
            optimizer.zero_grad() # clears the old gradients, so they don't accumulate

        result = evaluate(model, val_loader) # val_loss val_res_acc val_fps_acc val_both_acc
        writer.add_scalar('Loss/validation', result['val_loss'], epoch)
        writer.add_scalar('Accuracy/val_res_acc', result['val_res_acc'], epoch)
        writer.add_scalar('Accuracy/val_fps_acc', result['val_fps_acc'], epoch)
        writer.add_scalar('Accuracy/val_both_acc', result['val_both_acc'], epoch)
        writer.add_scalar('Loss/jod_RMSE', result['jod_RMSE'], epoch)
        writer.add_scalar('Loss/resolution_RMSE', result['resolution_RMSE'], epoch)
        writer.add_scalar('Loss/resolution_RMSEP', result['resolution_RMSEP'], epoch)
        writer.add_scalar('Loss/fps_RMSE', result['fps_RMSE'], epoch)
        writer.add_scalar('Loss/fps_RMSEP', result['fps_RMSEP'], epoch)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        writer.add_scalar('Loss/train', result['train_loss'], epoch)
        model.epoch_end(epoch, result)
        history.append(result)
        
        early_stopping(result['val_loss'], model, epoch)
        if early_stopping.early_stop:
            print(f"Early stopping triggered. Training stopped at epoch {epoch}.")
            early_stopping_triggered = True
            break

        if SAVE_HALFWAY and epoch % 45 == 0 and epoch > 0:
            os.makedirs(model_path, exist_ok=True)
            print(f"Epoch {epoch} is a multiple of 10.")
            save_checkpoint(model, optimizer,  f'{model_path}/checkpoint{epoch}.pth', epoch)
    
    if SAVE_MODEL and (not early_stopping_triggered):
        os.makedirs(model_path, exist_ok=True)
        torch.save(model.state_dict(), f'{model_path}/classification.pth')
    print(f'Trained model saved to {model_path}')
    writer.flush()
    writer.close()
    return history, model, model_path

def load_checkpoint_from_path(path, model, optimizer):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print(f'checkpoint epoch {epoch}')
    epochs = range(epoch+1, num_epochs)
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
    print(f'epochs {epochs}')
    return model, epochs, optimizer

def fetch_dataloader(batch_size, patch_size, device, patch_type, FRAMENUMBER=True):
    if patch_type == 'single':
        dataset = VideoSinglePatchDataset(directory=f'{VRRML}/{data_folder}/train', min_bitrate=500, max_bitrate=2000, \
                                        patch_size=patch_size, VELOCITY=True, FRAMENUMBER=FRAMENUMBER) # len 27592
        val_dataset = VideoSinglePatchDataset(directory=f'{VRRML}/{data_folder}/validation', min_bitrate=500, max_bitrate=2000, \
                                            patch_size=patch_size, VELOCITY=True, VALIDATION=True, FRAMENUMBER=FRAMENUMBER) # len 27592
        print(f'train_size {len(dataset)}, val_size {len(val_dataset)}, batch_size {batch_size}\n')
        print(f"Train dataset fps labels are: \n{dataset.fps_targets}\nTrain dataset res labels are: \n{dataset.res_targets}\n")
        sample = val_dataset[0]
        print('sample image has ', sample['fps'], 'fps,', sample['resolution'], ' resolution,', sample['bitrate'], 'bps')
        print(f'normalized velocity is {sample["velocity"]}, path is {sample["path"]}')
        print(f'learning rate {lr}, batch_size {batch_size}')
    else:
        dataset = VideoDualPatchDataset(directory=f'{VRRML}/{data_folder}/train', min_bitrate=500, max_bitrate=2000, \
                                        patch_size=patch_size, VELOCITY=True, FRAMENUMBER=FRAMENUMBER) # len 27592
        val_dataset = VideoDualPatchDataset(directory=f'{VRRML}/{data_folder}/validation', min_bitrate=500, max_bitrate=2000, \
                                            patch_size=patch_size, VELOCITY=True, VALIDATION=True, FRAMENUMBER=FRAMENUMBER) # len 27592
        print(f'train_size {len(dataset)}, val_size {len(val_dataset)}, batch_size {batch_size}\n')
        print(f"Train dataset fps labels are: \n{dataset.fps_targets}\nTrain dataset res labels are: \n{dataset.res_targets}\n")
        sample = val_dataset[0]
        print('sample image has ', sample['fps'], 'fps,', sample['resolution'], ' resolution,', sample['bitrate'], 'bps')
        print(f'normalized velocity is {sample["velocity"]}, path is {sample["path"]}')
        print(f'learning rate {lr}, batch_size {batch_size}')

    train_dl = DataLoader(dataset, batch_size, shuffle = True, num_workers = 4, pin_memory = True)
    val_dl = DataLoader(val_dataset, batch_size*2, shuffle = True, num_workers = 4, pin_memory = True)
    if device.type == 'cuda':
        print(f'Loading data to cuda...')
        train_dl = DeviceDataLoader(train_dl, device)
        val_dl = DeviceDataLoader(val_dl, device)
    return train_dl, val_dl

def fetch_test_dataloader(batch_size, patch_size, device, patch_type, FRAMENUMBER=True):
    if patch_type == 'single':
        test_dataset = VideoSinglePatchDataset(directory=f'{VRRML}/{data_test_folder}', min_bitrate=500, \
                                                max_bitrate=2000, patch_size=patch_size, VELOCITY=True, \
                                                VALIDATION=True, FRAMENUMBER=FRAMENUMBER) # len 27592
        print(f'Test data size {len(test_dataset)}, patch_size {patch_size}, batch_size {batch_size}\n')
        # sample = test_dataset[0]
    else:
        test_dataset = VideoDualPatchDataset(directory=f'{VRRML}/{data_test_folder}', min_bitrate=500, \
                                                max_bitrate=2000, patch_size=patch_size, VELOCITY=True, \
                                                VALIDATION=True, FRAMENUMBER=FRAMENUMBER) # len 27592
        print(f'\nTest data size {len(test_dataset)}, patch_size {patch_size}, batch_size {batch_size}\n')
   
    test_dl = DataLoader(test_dataset, batch_size*2, shuffle = False, num_workers = 4, pin_memory = True)
    test_dl = DeviceDataLoader(test_dl, device) 
    return test_dl

def set_manual_seed():
    torch.manual_seed(42)
    torch.cuda.manual_seed(42) # Ensure CUDA also uses the same seed
    torch.cuda.manual_seed_all(42)  # If using multi-GPU
    np.random.seed(42) # Ensure reproducibility with NumPy and Python's random
    torch.backends.cudnn.deterministic = True # Ensure deterministic behavior (may slow down training slightly)
    torch.backends.cudnn.benchmark = False
 


# train smaller model
# python .\VideoQualityClassifier_velocity_local.py --training_mode no_fps
if __name__ == "__main__":
    SAVE_MODEL = True
    SAVE_MODEL_HALF_WAY = True
    START_TRAINING= True # True False
    CHECKPOINT = False
    TEST_EVAL = True
    PATIENCE = 10 # early stopping
    num_epochs = 150 # 150
    ML_DATA_TYPE = 'ML' # ML_smaller
    PATCH_SIZE = 64

    parser = argparse.ArgumentParser(description="Training Configuration")
    parser.add_argument('--training_mode', type=str, choices=[
        'no_fps', 'no_res', 'no_fps_no_resolution', 'no_velocity', 
        'consecutive_patch', 'consecutive_patch_no_velocity', 'random_patch'
    ], required=True, help="Specify the training mode")
    args = parser.parse_args()

    training_params = {
        'no_fps': {'FPS': False, 'RESOLUTION': True, 'MODEL_VELOCITY': True, 'patch_type': 'single'},
        'no_res': {'FPS': True, 'RESOLUTION': False, 'MODEL_VELOCITY': True, 'patch_type': 'single'},
        'no_fps_no_resolution': {'FPS': False, 'RESOLUTION': False, 'MODEL_VELOCITY': True, 'patch_type': 'single'},
        'no_velocity': {'FPS': True, 'RESOLUTION': True, 'MODEL_VELOCITY': False, 'patch_type': 'single'},
        'full': {'FPS': True, 'RESOLUTION': True, 'MODEL_VELOCITY': True, 'patch_type': 'single'},
        'consecutive_patch': {'FPS': True, 'RESOLUTION': True, 'MODEL_VELOCITY': True, 'patch_type': 'consecutive'},
        'consecutive_patch_no_velocity': {'FPS': True, 'RESOLUTION': True, 'MODEL_VELOCITY': False, 'patch_type': 'consecutive'},
        'random_patch': {'FPS': True, 'RESOLUTION': True, 'MODEL_VELOCITY': True, 'patch_type': 'random'}
    }

    config = training_params[args.training_mode]
    FPS = config['FPS']
    RESOLUTION = config['RESOLUTION']
    MODEL_VELOCITY = config['MODEL_VELOCITY']
    patch_type = config['patch_type']
    print(f'FPS {FPS}, RESOLUTION {RESOLUTION}, MODEL_VELOCITY {MODEL_VELOCITY}, patch_type {patch_type}')

    velocity_type = 'frame-velocity' # frame-velocity patch-velocity
    data_folder = f'{ML_DATA_TYPE}/{velocity_type}/train_{patch_type}_{PATCH_SIZE}x{PATCH_SIZE}' 
    data_test_folder = f'{ML_DATA_TYPE}/{velocity_type}/test_{patch_type}_{PATCH_SIZE}x{PATCH_SIZE}' 
    checkpoint_path = ''

    lr = 0.0003
    opt_func = torch.optim.Adam # torch.optim.SGD not work
    batch_size = 128 
    patch_size = (PATCH_SIZE, PATCH_SIZE) 

    FRAMENUMBER = True # True

    device = get_default_device()
    saved_model_path = None
    set_manual_seed()
   
    if START_TRAINING:
        print(f'Start training...')
        start_time = time.time()
        if patch_type == 'single':
            model = DecRefClassification(FPS=FPS, RESOLUTION=RESOLUTION, VELOCITY=MODEL_VELOCITY)
        else:
            model = DecRefClassification_dual(FPS=FPS, RESOLUTION=RESOLUTION, VELOCITY=MODEL_VELOCITY)
        optimizer = opt_func(model.parameters(), lr)
        epochs = range(num_epochs)
        if CHECKPOINT:
            model, epochs, optimizer = load_checkpoint_from_path(checkpoint_path, model, optimizer)
        model.to(device)
        train_dl, val_dl = fetch_dataloader(batch_size, patch_size, device, patch_type, FRAMENUMBER=FRAMENUMBER)
        history, model, saved_model_path = fit(epochs, model, train_dl, val_dl, optimizer, args.training_mode, SAVE_MODEL=SAVE_MODEL, \
                                               SAVE_HALFWAY=SAVE_MODEL_HALF_WAY, VELOCITY=True, CHECKPOINT=CHECKPOINT)
        elapsed_time = time.time() - start_time  # Compute elapsed time
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)
        print(f"Elapsed Time: {hours}h {minutes}m {seconds}s")


    if TEST_EVAL:
        print('\nTest evaluating...')
        test_dl = fetch_test_dataloader(batch_size, patch_size, device, patch_type, FRAMENUMBER=True)        
        result, res_preds, fps_preds, res_targets, fps_targets, jod_preds, jod_targets = evaluate_test_data(model, test_dl)
        predicted_fps = torch.tensor([reverse_fps_map[int(pred)] for pred in fps_preds])
        target_fps = torch.tensor([reverse_fps_map[int(target)] for target in fps_targets])

        predicted_res = torch.tensor([reverse_res_map[int(pred)] for pred in res_preds])
        target_res = torch.tensor([reverse_res_map[int(target)] for target in res_targets])
        print(f'predicted_res {predicted_res}')
        print(f'target_res {target_res}')

        # Root Mean Square Error https://help.pecan.ai/en/articles/6456388-model-performance-metrics-for-regression-models
        resolution_RMSE = compute_RMSE(predicted_res, target_res)
        fps_RMSE = compute_RMSE(predicted_fps, target_fps)
        jod_RMSE = compute_RMSE(jod_preds, jod_targets)

        # Root Mean Squared Percentage Error (RMSPE)
        resolution_RMSEP = relative_error_metric(predicted_res, target_res) 
        fps_RMSEP = relative_error_metric(predicted_fps, target_fps) 
        print(f"FPS: Root Mean Squared Error {fps_RMSE}, Root Mean Squared Percentage Error (RMSPE): {fps_RMSEP}%")
        print(f"Resolution: Root Mean Squared Error {resolution_RMSE}, Root Mean Squared Percentage Error (RMSPE): {resolution_RMSEP}%\n")
        print(f'jod rmse {jod_RMSE}')
        print(f'test result \n {result}\n')

        # TODO: rename saved_model_path based on training type
        with open(f'{saved_model_path}/model.txt', 'a') as f:
            f.write(f"Elapsed Time: {hours}h {minutes}m {seconds}s\n\n")
            f.write(f"FPS: Root Mean Squared Error {fps_RMSE}, Root Mean Squared Percentage Error (RMSPE): {fps_RMSEP}%\n")
            f.write(f'Resolution: Root Mean Squared Error {resolution_RMSE}, Root Mean Squared Percentage Error (RMSPE): {resolution_RMSEP}%\n')
            f.write(f'test result \n {result}\n\n')
            f.write(f"{round(fps_RMSE, 1)} {round(fps_RMSEP)}%\n")
            f.write(f'{round(resolution_RMSE, 1)} {round(resolution_RMSEP)}%\n')