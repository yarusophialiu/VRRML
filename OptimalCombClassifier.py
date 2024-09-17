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

# from VideoSinglePatchDataset import VideoSinglePatchDataset
from PatchDataset_noval import PatchDataset
from DeviceDataLoader import DeviceDataLoader
from utils import *
from DecRefClassification import *

# regressin, learn the curves
# https://docs.google.com/presentation/d/16yqaaq5zDZ5-S4394VLBUfxpNjM7nlpssqcShFkklec/edit#slide=id.g2c751bc0d9c_0_18


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    # outputs = [model.validation_step(batch) for batch in val_loader]
    outputs = []
    for mini_batch_idx, (images, metadata) in enumerate(val_loader):
        print(f'=============== batch {mini_batch_idx} ===============') # train_size / batch_size
        print(f"mini_batch_idx: {mini_batch_idx}")
        # print(f"image size: {images.size()}")  # (batch_size, 10000, 3, 64, 64) if 10000 images per folder
        # print(f"metadata: {metadata.size()}")
        images = images.reshape(-1, 3, 64, 64)
        metadata = metadata.reshape(-1, 6)
        val_res = model.validation_step(images, metadata)
        outputs.append(val_res)

    # print(f'eval outputs \n {outputs}')
    return model.validation_epoch_end(outputs)


def evaluate_test_data(model, test_loader):
    model.eval()
    with torch.no_grad():  # Ensure gradients are not computed
        for batch in test_loader:
            images = batch["image"]
            # labels = batch["label"]
            fps = batch["fps"]
            bitrate = batch["bitrate"]
            resolution = batch["resolution"]
            # velocity = batch["velocity"]
        
            print(f'bitrate {bitrate}')

            # arr = [30, 30, 30, 40, 40, 50, 50, 50]
            # tensor = torch.tensor(arr)
            unique_indices = {}

            # Iterate over the tensor and populate the dictionary
            for index, value in enumerate(bitrate.tolist()):
                if value not in unique_indices:
                    unique_indices[value] = index

            # TODO: convert labels into res_targets, fps_targets 
            res_targets = batch["res_targets"]
            fps_targets = batch["fps_targets"]
            res_out, fps_out = model(images, fps, bitrate, resolution)  # NaturalSceneClassification.forward
            # print(f'training_step out {out.size()} \n {out.squeeze()}')
            # print(f'labels out {labels}')
            total_loss = compute_weighted_loss(res_out, fps_out, res_targets, fps_targets)
            framerate_accuracy, resolution_accuracy, both_correct_accuracy = compute_accuracy(fps_out, res_out, fps_targets, res_targets)

            fps = [30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
            resolution = [360, 480, 720, 864, 1080]

            fps_idx = torch.argmax(fps_out, dim=1)
            res_idx = torch.argmax(res_out, dim=1)

            fps_values = [fps[idx] for idx in fps_idx]
            res_values = [resolution[idx] for idx in res_idx]


            return {'val_loss': total_loss.detach(), 'res_acc': resolution_accuracy, 'fps_acc': framerate_accuracy, \
                    'both_acc': both_correct_accuracy}, res_out, fps_out, res_targets, fps_targets, \
                        res_values, fps_values, unique_indices


def fit(epochs, lr, model, train_loader, val_loader, opt_func, SAVE_MODEL=False, SAVE_HALFWAY=False, VELOCITY=True):
    
    history = []
    train_accuracies = []
    val_accuracies = []
    optimizer = opt_func(model.parameters(),lr)

    model_path = ""
    if SAVE_MODEL or SAVE_HALFWAY:
        now = datetime.now()
        dir_pth = now.strftime("%Y-%m-%d")
        os.makedirs(dir_pth, exist_ok=True)
        hrmin = now.strftime("%H_%M")
        model_path = os.path.join(dir_pth, hrmin)
        # os.makedirs(model_path, exist_ok=True)

    # an epoch is one pass through the entire dataset
    for epoch in range(epochs):
        print(f'================================ epoch {epoch} ================================')
        model.train()
        train_losses = []
        count = 0
        # for each batch, compute gradients for every data
        # after the batch finishes, evaluate
        # requests an iterator from DeviceDataLoader, i.e. __iter__ function

        # are batch1 and batch2 not overlap?
        # do all batches cover the whole dataset?
        for mini_batch_idx, (images, metadata) in enumerate(train_loader):
            print(f'=============== batch {mini_batch_idx} ===============') # train_size / batch_size
            print(f"mini_batch_idx: {mini_batch_idx}")
            # print(f"image size: {images.size()}")  # (batch_size, 10000, 3, 64, 64) if 10000 images per folder
            # print(f"metadata: {metadata.size()}")
            images = images.reshape(-1, 3, 64, 64)
            metadata = metadata.reshape(-1, 6)
            # print(f"image tensor ize: {images.size()}")  # (batch_size, 10000, 3, 64, 64) if 10000 images per folder
            # print(f"metadata: {metadata.size()}\n")
            size = metadata.size()[0]
            perm = torch.randperm(size)
            shuffled_images = images[perm]
            shuffled_metadata = metadata[perm] # fps, resolution, fps_target, resolution_target, image_bitrate, velocity/1000
            # print(f'shuffled_metadata \n {shuffled_metadata}')
            # print(f"fps_target: {fps_target}, res_target: {res_target}, bitrate {bitrate}\n")
            # TODO: shuffle images before fit
            
            loss = model.training_step(shuffled_images, shuffled_metadata, VELOCITY=VELOCITY) # model
            train_losses.append(loss)

            # computes the gradient of the loss with respect to the model parameters
            # part of the backpropagation algorithm, which is how the neural network learns
            loss.backward()  
            # update the model parameters based on the gradients calculated
            optimizer.step()
            # clears the old gradients, so they don't accumulate
            optimizer.zero_grad()
            
        result = evaluate(model, val_loader) # a dictionary of validation losses
        # print(f'result \n {result}')
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result) # print loss results
        history.append(result)

        if SAVE_HALFWAY and epoch % 10 == 0 and epoch > 0:
            os.makedirs(model_path, exist_ok=True)
            print(f"Epoch {epoch} is a multiple of 20.")
            save_checkpoint(model, optimizer,  f'{model_path}/checkpoint{epoch}.pth', epoch)
        
    if SAVE_MODEL:
        os.makedirs(model_path, exist_ok=True)
        torch.save(model.state_dict(), f'{model_path}/classification.pth')
    
    return history

# input image 64x64, decocded OR reference patch
# fps useful, resolution not useful
if __name__ == "__main__":
    # data_directory = 'C:/Users/15142/Desktop/data/VRR-video-classification/'
    # data_directory = 'C:/Users/15142/Desktop/data/VRR-classification/'
    data_train_directory = f'{VRRML}/train' 
    data_val_directory = f'{VRRML}/validation' 
    # data_test_directory = f'{data_directory}/bistro-fast/test_dec_ref_bistro/test'
    # data_test_directory = f'{VRRML}/test'
    SAVE_MODEL = True
    SAVE_MODEL_HALF_WAY = True
    START_TRAINING= True # True False
    TEST_EVAL = False
    PLOT_TEST_RESULT = False
    SAVE_PLOT = True
    TEST_SINGLE_IMG = False

    num_epochs = 1
    lr = 0.001
    opt_func = torch.optim.Adam
    batch_size = 1 # 450 path folders, 1 path folder for 1 mini batch if batch_size = 1
    # batch_size = 1 # 450 path folders, 1 path folder for 1 mini batch if batch_size = 1
    # val_size = 3000 # each path folder has 10k patches

    num_framerates, num_resolutions = 10, 5

    train_dataset = PatchDataset(root_dir=data_train_directory, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # One path folder per batch
    val_dataset = PatchDataset(root_dir=data_val_directory, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)  # One path folder per batch

    device = get_default_device()
    cuda  = device.type == 'cuda'
    print(f'cuda {cuda}')


    if START_TRAINING:
        # step2 split data and prepare batches
        # train_data, val_data = random_split(dataset,[train_size,val_size])
        # print(f"Length of Train Data : {len(train_data)}")
        # print(f"Length of Validation Data : {len(val_data)} \n")

        model = DecRefClassification(num_framerates, num_resolutions)
        # print(model)
 
        if device.type == 'cuda':
            print(f'Loading data to cuda...')
            train_dl = DeviceDataLoader(train_loader, device)
            val_dl = DeviceDataLoader(val_loader, device)
            to_device(model, device)
            
        # fitting the model on training data and record the result after each epoch
        history = fit(num_epochs, lr, model, train_dl, val_dl, opt_func, \
                      SAVE_MODEL=SAVE_MODEL, SAVE_HALFWAY=SAVE_MODEL_HALF_WAY)


    if TEST_EVAL:
        model = DecRefClassification(num_framerates, num_resolutions)
        model_path = f'2024-05-06/17_02/classification_reference.pth'
        model.load_state_dict(torch.load(model_path))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        print(f'test_dataset {len(test_dataset)}')
        test_dl = DataLoader(test_dataset, len(test_dataset))
        if device.type == 'cuda':
            print(f'\nLoading data to cuda...')
            test_dl = DeviceDataLoader(test_dl, device)
            to_device(model, device)


        # {'val_loss': total_loss.detach(), 'res_acc': resolution_accuracy, 'fps_acc': framerate_accuracy, \
                    # 'both_acc': both_correct_accuracy} 

        result, res_out, fps_out, res_targets, fps_targets, \
            res_values, fps_values, unique_indices = evaluate_test_data(model, test_dl)
        
        # unique_indices_arr = [val for k, val in unique_indices.items()]
        bitrate_predictions = {}
        # print(f"First indices of unique values: {unique_indices}")

        res_values = torch.tensor(res_values)
        fps_values = torch.tensor(fps_values)

        for k, val in unique_indices.items():
            bitrate_predictions[k] = []
            bitrate_predictions[k].append(res_values[val].item())
            bitrate_predictions[k].append(fps_values[val].item())

        print(f'test result \n {result}\n')
        # print(f'res_out \n {res_out}')
        # print(f'fps_out \n {fps_out}')
        # print(f'res_targets \n {res_targets}')
        # print(f'fps_targets \n {fps_targets}')
        # print(f'res_values \n {(res_values)}')
        # print(f'res_values \n {torch.unique(res_values)}\n')
        # print(f'fps_values \n {(fps_values)}')
        # print(f'fps_values \n {torch.unique(fps_values)}\n')
        print(f'bitrate_predictions \n {bitrate_predictions}')


     