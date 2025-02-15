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
from DecRefClassification_smaller import *
from VideoQualityClassifier_velocity_test import evaluate_test_data
from torch.utils.tensorboard import SummaryWriter

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
    history = []
    model_path = ""
    if SAVE_MODEL or SAVE_HALFWAY:
        now = datetime.now()
        dir_pth = now.strftime("%Y-%m-%d")
        os.makedirs(dir_pth, exist_ok=True)
        hrmin = now.strftime("%H_%M")
        model_path = os.path.join(dir_pth, hrmin)
        os.makedirs(model_path, exist_ok=True)
    
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

        if SAVE_HALFWAY and epoch % 30 == 0 and epoch > 0:
            os.makedirs(model_path, exist_ok=True)
            print(f"Epoch {epoch} is a multiple of 10.")
            save_checkpoint(model, optimizer,  f'{model_path}/checkpoint{epoch}.pth', epoch)
        
    if SAVE_MODEL:
        os.makedirs(model_path, exist_ok=True)
        torch.save(model.state_dict(), f'{model_path}/classification.pth')
    print(f'Trained model saved to {model_path}')
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

def fetch_dataloader(batch_size, patch_size, device, FRAMENUMBER=True):
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

    train_dl = DataLoader(dataset, batch_size, shuffle = True, num_workers = 4, pin_memory = True)
    val_dl = DataLoader(val_dataset, batch_size*2, shuffle = True, num_workers = 4, pin_memory = True)
    if device.type == 'cuda':
        print(f'Loading data to cuda...')
        train_dl = DeviceDataLoader(train_dl, device)
        val_dl = DeviceDataLoader(val_dl, device)
    return train_dl, val_dl

def fetch_test_dataloader(batch_size, patch_size, device, FRAMENUMBER=True):
    test_dataset = VideoSinglePatchDataset(directory=f'{VRRML}/{data_test_folder}', min_bitrate=500, \
                                            max_bitrate=2000, patch_size=patch_size, VELOCITY=True, \
                                            VALIDATION=True, FRAMENUMBER=FRAMENUMBER) # len 27592
    print(f'\ntest_size {len(test_dataset)}, patch_size {patch_size}, batch_size {batch_size}\n')
    # sample = test_dataset[0]
    test_dl = DataLoader(test_dataset, batch_size*2, shuffle = False, num_workers = 4, pin_memory = True)
    test_dl = DeviceDataLoader(test_dl, device) 
    return test_dl
    



# train smaller model
if __name__ == "__main__":
    SAVE_MODEL = True
    SAVE_MODEL_HALF_WAY = True
    START_TRAINING= True # True False
    CHECKPOINT = False
    TEST_EVAL = True
    
    velocity_type = 'frame-velocity' # frame-velocity patch-velocity
    patch_type = 'random' # consecutive random single
    PATCH_SIZE = 64
    data_folder = f'ML_smaller/{velocity_type}/train_{patch_type}_{PATCH_SIZE}x{PATCH_SIZE}' 
    data_test_folder = f'ML_smaller/{velocity_type}/test_{patch_type}_{PATCH_SIZE}x{PATCH_SIZE}' 
    checkpoint_path = ''

    num_epochs = 1 # 150
    lr = 0.0003
    opt_func = torch.optim.Adam # torch.optim.SGD not work
    batch_size = 128 
    patch_size = (PATCH_SIZE, PATCH_SIZE) 

    FPS=True # training params
    RESOLUTION=True
    MODEL_VELOCITY=True

    FRAMENUMBER = True # True

    device = get_default_device()
    saved_model_path = None

    torch.manual_seed(42)
    torch.cuda.manual_seed(42) # Ensure CUDA also uses the same seed
    torch.cuda.manual_seed_all(42)  # If using multi-GPU
    np.random.seed(42) # Ensure reproducibility with NumPy and Python's random
    torch.backends.cudnn.deterministic = True # Ensure deterministic behavior (may slow down training slightly)
    torch.backends.cudnn.benchmark = False
    
    if START_TRAINING:
        print(f'Start training...')
        model = DecRefClassification(FPS=FPS, RESOLUTION=RESOLUTION, VELOCITY=MODEL_VELOCITY)
        optimizer = opt_func(model.parameters(), lr)
        epochs = range(num_epochs)
        if CHECKPOINT:
            model, epochs, optimizer = load_checkpoint_from_path(checkpoint_path, model, optimizer)
        model.to(device)
        train_dl, val_dl = fetch_dataloader(batch_size, patch_size, device, FRAMENUMBER=FRAMENUMBER)
        history, model, saved_model_path = fit(epochs, model, train_dl, val_dl, optimizer, SAVE_MODEL=SAVE_MODEL, \
                                               SAVE_HALFWAY=SAVE_MODEL_HALF_WAY, VELOCITY=True, CHECKPOINT=CHECKPOINT)
    if TEST_EVAL:
        print('Test evaluating...')
        test_dl = fetch_test_dataloader(batch_size, patch_size, device, FRAMENUMBER=True)        
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
            f.write(f"FPS: Root Mean Squared Error {fps_RMSE}, Root Mean Squared Percentage Error (RMSPE): {fps_RMSEP}%\n")
            f.write(f'Resolution: Root Mean Squared Error {resolution_RMSE}, Root Mean Squared Percentage Error (RMSPE): {resolution_RMSEP}%\n')
            f.write(f'test result \n {result}\n\n')
            f.write(f"{round(fps_RMSE, 1)} {round(fps_RMSEP)}%\n")
            f.write(f'{round(resolution_RMSE, 1)} {round(resolution_RMSEP)}%\n')
