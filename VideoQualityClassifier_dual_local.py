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
from EarlyStopping import EarlyStopping


import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from datetime import datetime

from VideoSinglePatchDataset import VideoSinglePatchDataset
# from VideoDualPatchDataset import VideoDualPatchDataset
from DeviceDataLoader import DeviceDataLoader
from utils import *
# from DecRefClassification_dual_smaller import *
from DecRefClassification_smaller import *
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import time


# regressin, learn the curves
# https://docs.google.com/presentation/d/16yqaaq5zDZ5-S4394VLBUfxpNjM7nlpssqcShFkklec/edit#slide=id.g2c751bc0d9c_0_18

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

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


def fit(epochs, model, train_loader, val_loader, optimizer, \
                SAVE_MODEL=False, SAVE_HALFWAY=False, VELOCITY=False, CHECKPOINT=False):
    print(f'fit VELOCITY {VELOCITY}')
    early_stopping = None

    history = []
    # optimizer = opt_func(model.parameters(),lr)
    model_path = ""
    # if SAVE_MODEL or SAVE_HALFWAY:
    now = datetime.now()
    dir_pth = now.strftime("%Y-%m-%d")
    os.makedirs(dir_pth, exist_ok=True)
    hrmin = now.strftime("%H_%M")
    model_path = os.path.join(dir_pth, hrmin)
    early_stopping = EarlyStopping(patience=PATIENCE, delta=0.001, path=f"{model_path}")
    early_stopping_triggered = False


    # an epoch is one pass through the entire dataset
    for epoch in epochs:
        print(f'================================ epoch {epoch} ================================')
        model.train()
        train_losses = []
        count = 0
        running_loss = 0.0
        # for each batch, compute gradients for every data
        # after the batch finishes, evaluate
        # requests an iterator from DeviceDataLoader, i.e. __iter__ function

        # are batch1 and batch2 not overlap? NO when you use DataLoader with shuffle=True, 
        # batches will not overlap during training
        # do all batches cover the whole dataset?
        for batch in train_loader: # batch is a dictionary with 32 images information, e.g. 'fps': [70, 80, ..., 150]
            # print(f'batch {batch[fps]}')
            # print(f'=============== batch {count} ===============') # train_size / batch_size
            count += 1
            # images= batch['image']
            # print(f"Input batch shape: {images.size()}")
            # get accuracy
            loss = model.training_step(batch, VELOCITY=VELOCITY) # model
            train_losses.append(loss)
            running_loss += loss.item()

            # computes the gradient of the loss with respect to the model parameters
            # part of the backpropagation algorithm, which is how the neural network learns
            loss.backward()  
            optimizer.step() # update the model parameters based on the gradients calculated
            optimizer.zero_grad() # clears the old gradients, so they don't accumulate

        # avg_train_loss = running_loss / len(train_loader) # len(train_loader) is number of batches
        result = evaluate(model, val_loader) # val_loss val_res_acc val_fps_acc val_both_acc
        # Log training and validation metrics to TensorBoard
        # writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/validation', result['val_loss'], epoch)
        writer.add_scalar('Accuracy/val_res_acc', result['val_res_acc'], epoch)
        writer.add_scalar('Accuracy/val_fps_acc', result['val_fps_acc'], epoch)
        writer.add_scalar('Accuracy/val_both_acc', result['val_both_acc'], epoch)
        writer.add_scalar('Loss/jod_RMSE', result['jod_RMSE'], epoch)
        writer.add_scalar('Loss/resolution_RMSE', result['resolution_RMSE'], epoch)
        writer.add_scalar('Loss/resolution_RMSEP', result['resolution_RMSEP'], epoch)
        writer.add_scalar('Loss/fps_RMSE', result['fps_RMSE'], epoch)
        writer.add_scalar('Loss/fps_RMSEP', result['fps_RMSEP'], epoch)

        # print(f'result \n {result}')
        result['train_loss'] = torch.stack(train_losses).mean().item()
        writer.add_scalar('Loss/train', result['train_loss'], epoch)
        # print(f'len(val_loader) {len(val_loader)}, avg_train_loss {avg_train_loss} {torch.stack(train_losses).mean().item()}')

        model.epoch_end(epoch, result)
        history.append(result)
        # val_loss = result['val_loss']
        # print(f'val loss {val_loss}')
        early_stopping(result['val_loss'], model, epoch)
        if early_stopping.early_stop:
            print(f"Early stopping triggered. Training stopped at epoch {epoch}.")
            early_stopping_triggered = True
            break

        if SAVE_HALFWAY and epoch % 30 == 0 and epoch > 0:
            os.makedirs(model_path, exist_ok=True)
            print(f"Epoch {epoch} is a multiple of 10.")
            save_checkpoint(model, optimizer,  f'{model_path}/checkpoint{epoch}.pth', epoch)
        
    if SAVE_MODEL and (not early_stopping_triggered):
        os.makedirs(model_path, exist_ok=True)
        print(f'epoch = {epoch}')
        torch.save(model.state_dict(), f'{model_path}/classification{epoch}.pth')
    print(f'model_path {model_path}')
    return history



# OLD from the last VRRML on desktop
if __name__ == "__main__":
    SAVE_MODEL = True
    SAVE_MODEL_HALF_WAY = False
    START_TRAINING= True # True False
    CHECKPOINT = False
    CHECKPOINT_PATH = '2025-02-10/consecutive_patches_with_velocity/checkpoint60.pth'
    TEST_EVAL = False
    TEST_UNSEEN_SCENE = False # True
    PATIENCE = 10

    model_pth_path = ""
    folder = 'ML/reference64x64' # reference64x64 # TODO change model size reference128x128
    if TEST_UNSEEN_SCENE:
        print(f'test on unseen scenes')
        data_test_directory = f'{VRRML}/ML/test_scenes128x128' 
    else:
        data_test_directory = f'{VRRML}/{folder}/test'
    data_train_directory = f'{VRRML}/{folder}/train' # ML_smaller
    data_val_directory = f'{VRRML}/{folder}/validation'  

    if TEST_EVAL:
        model_pth_path = f'models/patch128-256/patch128_batch128.pth' # patch128_batch128 patch256_batch64

    num_epochs = 150 # 131
    lr = 0.0003
    # opt_func = torch.optim.SGD
    opt_func = torch.optim.Adam
    batch_size = 128 # TODO
    patch_size = (64, 64) # TODO, change patch structure in DecRefClassification.py

    FPS=False 
    RESOLUTION=False
    MODEL_VELOCITY=True
    BATCHNORM=False # False

    num_framerates, num_resolutions = 10, 5
    VELOCITY = True
    VALIDATION = True
    FRAMENUMBER = False # False True reference64x64 doesnt have framenumber

    dataset = VideoSinglePatchDataset(directory=data_train_directory, min_bitrate=500, max_bitrate=2000, patch_size=patch_size, VELOCITY=VELOCITY, FRAMENUMBER=FRAMENUMBER) # len 27592
    val_dataset = VideoSinglePatchDataset(directory=data_val_directory, min_bitrate=500, max_bitrate=2000, patch_size=patch_size, VELOCITY=VELOCITY, VALIDATION=VALIDATION, FRAMENUMBER=FRAMENUMBER) # len 27592
    print(f'train_size {len(dataset)}, val_size {len(val_dataset)}, batch_size {batch_size}\n')
    print(f"Train dataset fps labels are: \n{dataset.fps_targets}\nTrain dataset res labels are: \n{dataset.res_targets}\n")
    print(f"Validation dataset fps labels are: \n{val_dataset.fps_targets}\nValidation dataset res labels are: \n{val_dataset.res_targets}\n")
    sample = val_dataset[0]
    print('sample image has ', sample['fps'], 'fps,', sample['resolution'], ' resolution,', sample['bitrate'], 'bps')
    print(f'sample velocity is {sample["velocity"]}') if VELOCITY else None
    print(f'sample path is {sample["path"]}') if VELOCITY else None
    print(f'learning rate {lr}, batch_size {batch_size}')

    device = get_default_device()
    cuda  = device.type == 'cuda'

    if START_TRAINING:
        torch.manual_seed(42)
        # Ensure CUDA also uses the same seed
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)  # If using multi-GPU
        # Ensure reproducibility with NumPy and Python's random
        np.random.seed(42)
        # Ensure deterministic behavior (may slow down training slightly)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print("Time with seconds:", datetime.now().strftime("%H:%M:%S"))
        start_time = time.time()  # Record start time

        # model = DecRefClassification(num_framerates, num_resolutions, VELOCITY=VELOCITY)
        model = DecRefClassification(num_framerates, num_resolutions, \
                                     FPS=FPS, RESOLUTION=RESOLUTION, VELOCITY=MODEL_VELOCITY,\
                                     BATCHNORM=BATCHNORM)
        optimizer = opt_func(model.parameters(),lr)
        epochs = range(num_epochs)
        if CHECKPOINT:
            # enable nn.AdaptiveAvgPool2d((1, 1)) if avg_pools
            checkpoint = torch.load(CHECKPOINT_PATH)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            print(f'epoch {epoch}')
            # epoch = 20
            epochs = range(epoch+1, num_epochs)
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
        print(f'epochs {epochs}')
        train_dl = DataLoader(dataset, batch_size, shuffle = True, num_workers = 4, pin_memory = True)
        val_dl = DataLoader(val_dataset, batch_size*2, shuffle = True, num_workers = 4, pin_memory = True)
        if device.type == 'cuda':
            print(f'Loading data to cuda...')
            train_dl = DeviceDataLoader(train_dl, device)
            val_dl = DeviceDataLoader(val_dl, device)
            to_device(model, device)

        now = datetime.now()
        dir_pth = now.strftime("%Y-%m-%d")
        os.makedirs(dir_pth, exist_ok=True)
        hrmin = now.strftime("%H_%M")
        runs_path = os.path.join(dir_pth, hrmin)
        os.makedirs(f'runs/{runs_path}', exist_ok=True)
        writer = SummaryWriter(f'runs/{runs_path}')
        # fitting the model on training data and record the result after each epoch
        history = fit(epochs, model, train_dl, val_dl, optimizer, SAVE_MODEL=SAVE_MODEL, \
                        SAVE_HALFWAY=SAVE_MODEL_HALF_WAY, VELOCITY=VELOCITY, CHECKPOINT=CHECKPOINT)
        writer.flush()
        writer.close()
        elapsed_time = time.time() - start_time  # Compute elapsed time
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)
        print(f"Elapsed Time: {hours}h {minutes}m {seconds}s")


    # if TEST_EVAL: # test data size 35756
    #     test_dataset = VideoSinglePatchDataset(directory=data_test_directory, min_bitrate=500, max_bitrate=2000, patch_size=patch_size, VELOCITY=VELOCITY, VALIDATION=VALIDATION) # len 27592
    #     print(f'\ntest_size {len(test_dataset)}, patch_size {patch_size}, batch_size {batch_size}\n')
    #     model = DecRefClassification(num_framerates, num_resolutions, VELOCITY=VELOCITY)
    #     # model_pth_path = f'models/patch128_batch256.pth' # patch128_batch128 patch256_batch64
    #     model.load_state_dict(torch.load(model_pth_path))
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     # model.to(device)
    #     # test_dl = DataLoader(test_dataset, len(test_dataset))
    #     test_dl = DataLoader(test_dataset, batch_size, shuffle = True, num_workers = 4, pin_memory = True)

    #     if device.type == 'cuda':
    #         print(f'Loading data to cuda...')
    #         test_dl = DeviceDataLoader(test_dl, device)
    #         to_device(model, device)

    #     print(f'model_path {model_pth_path}')
    #     result, res_out, fps_out, res_targets, fps_targets, \
    #                     res_values, fps_values, unique_indices = evaluate_test_data(model, test_dl)
    #     _, fps_preds = torch.max(fps_out, dim=1)
    #     _, res_preds = torch.max(res_out, dim=1)
    #     # print(f'res_preds {res_preds}')
    #     # print(f'res_targets {res_targets}\n')
        
    #     reverse_fps_map = {v: k for k, v in fps_map.items()}
    #     reverse_res_map = {v: k for k, v in res_map.items()}

    #     # Convert the predicted and target indices to actual values
    #     predicted_fps = torch.tensor([reverse_fps_map[int(pred)] for pred in fps_preds])
    #     target_fps = torch.tensor([reverse_fps_map[int(target)] for target in fps_targets])

    #     predicted_res = torch.tensor([reverse_res_map[int(pred)] for pred in res_preds])
    #     target_res = torch.tensor([reverse_res_map[int(target)] for target in res_targets])
    #     # print(f'predicted_res {predicted_res}')
    #     # print(f'target_res {target_res}')

    #     absolute_errors_fps = torch.abs(predicted_fps - target_fps)
    #     absolute_errors_res = torch.abs(predicted_res - target_res)
    #     # print(f'absolute_errors_res {absolute_errors_res}\n')
    #     expected_error_fps = torch.mean(absolute_errors_fps.float())
    #     expected_error_res = torch.mean(absolute_errors_res.float())
    #     # print(f"Expected Error vertical resolution (Mean Absolute Error): {expected_error_res.item()}\n")
    #     # print(f"Expected Error fps (Mean Absolute Error): {expected_error_fps.item()}")

    #     # Compute the absolute percentage error
    #     # print(f'predicted_fps {predicted_fps}')
    #     percentage_errors_fps = torch.abs((predicted_fps - target_fps) / target_fps) * 100
    #     percentage_errors_res = torch.abs((predicted_res - target_res) / target_res) * 100
    #     # print(f"percentage_errors_fps: {percentage_errors_fps}") 
    #     # print(f"percentage_errors_res: {percentage_errors_res}\n") 

    #     # Compute the mean absolute percentage error (MAPE)
    #     mape_fps = torch.mean(percentage_errors_fps.float())
    #     mape_res = torch.mean(percentage_errors_res.float())
    #     print(f"FPS: Mean Absolute Error {expected_error_fps.item()}, Mean Absolute Percentage Error (MAPE): {round(mape_fps.item(), 3)}%")
    #     print(f"Resolution: Mean Absolute Error {expected_error_res.item()}, Mean Absolute Percentage Error (MAPE): {round(mape_res.item(), 3)}%\n")

    #     print(f'test result \n {result}\n')

     