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
# from torch.utils.tensorboard import SummaryWriter


# regressin, learn the curves
# https://docs.google.com/presentation/d/16yqaaq5zDZ5-S4394VLBUfxpNjM7nlpssqcShFkklec/edit#slide=id.g2c751bc0d9c_0_18



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


def fit(epochs, lr, model, train_loader, val_loader, opt_func, SAVE_MODEL=False, SAVE_HALFWAY=False, VELOCITY=False):
    print(f'fit VELOCITY {VELOCITY}')
    history = []
    optimizer = opt_func(model.parameters(),lr)
    running_loss = 0.0

    model_path = ""
    if SAVE_MODEL or SAVE_HALFWAY:
        now = datetime.now()
        dir_pth = now.strftime("%Y-%m-%d")
        os.makedirs(dir_pth, exist_ok=True)
        hrmin = now.strftime("%H_%M")
        model_path = os.path.join(dir_pth, hrmin)


    # an epoch is one pass through the entire dataset
    for epoch in range(epochs):
        print(f'================================ epoch {epoch} ================================')
        model.train()
        train_losses = []
        count = 0
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

        avg_train_loss = running_loss / len(train_loader) # len(train_loader) is number of batches
        print(f'len(train_loader) {len(train_loader)}, avg_train_loss {avg_train_loss}')

        result = evaluate(model, val_loader) # val_loss val_res_acc val_fps_acc val_both_acc
        # avg_val_loss = result['val_loss'] / len(val_loader)
        # avg_val_res_acc = result['val_res_acc'] / len(val_loader)
        # avg_val_fps_acc = result['val_fps_acc'] / len(val_loader)
        # avg_val_both_acc = result['val_both_acc'] / len(val_loader)
        # # Log training and validation metrics to TensorBoard
        # writer.add_scalar('Loss/train', avg_train_loss, epoch)
        # writer.add_scalar('Loss/validation', avg_val_loss, epoch)
        # writer.add_scalar('Accuracy/val_res_acc', avg_val_res_acc, epoch)
        # writer.add_scalar('Accuracy/val_fps_acc', avg_val_fps_acc, epoch)
        # writer.add_scalar('Accuracy/val_both_acc', avg_val_both_acc, epoch)

        # print(f'result \n {result}')
        result['train_loss'] = torch.stack(train_losses).mean().item()
        # print(f'len(val_loader) {len(val_loader)}, avg_train_loss {avg_train_loss} {torch.stack(train_losses).mean().item()}')

        model.epoch_end(epoch, result)
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
# OLD from the last VRRML on desktop
if __name__ == "__main__":
    folder = 'ML/reference128x128'
    data_train_directory = f'{VRRML}/{folder}/train' # ML_smaller
    data_val_directory = f'{VRRML}/{folder}/validation' 

    SAVE_MODEL = False
    SAVE_MODEL_HALF_WAY = False
    START_TRAINING= True # True False
    TEST_EVAL = False
    PLOT_TEST_RESULT = False
    SAVE_PLOT = True
    TEST_SINGLE_IMG = False

    num_epochs = 1
    lr = 0.005
    # opt_func = torch.optim.SGD
    opt_func = torch.optim.Adam
    batch_size = 64
    patch_size = (128, 128)

    num_framerates, num_resolutions = 10, 5
    VELOCITY = True
    VALIDATION = True

    # step1 load data
    dataset = VideoSinglePatchDataset(directory=data_train_directory, min_bitrate=500, max_bitrate=2000, patch_size=patch_size, VELOCITY=VELOCITY) # len 27592
    val_dataset = VideoSinglePatchDataset(directory=data_val_directory, min_bitrate=500, max_bitrate=2000, patch_size=patch_size, VELOCITY=VELOCITY, VALIDATION=VALIDATION) # len 27592
    # test_dataset = VideoSinglePatchDataset(directory=data_test_directory, TYPE=TYPE, VELOCITY=VELOCITY) 
    train_size = len(dataset)
    val_size = len(val_dataset)
    # print(f'total data {len(dataset)}, batch_size {batch_size}')
    print(f'train_size {train_size}, val_size {val_size}, batch_size {batch_size}\n')
    # print(f"Train dataset  labels are: \n{dataset.labels}")
    print(f"Train dataset fps labels are: \n{dataset.fps_targets}")
    print(f"Train dataset res labels are: \n{dataset.res_targets}\n")
    print(f"Validation dataset fps labels are: \n{val_dataset.fps_targets}")
    print(f"Validation dataset res labels are: \n{val_dataset.res_targets}\n")
    sample = val_dataset[0]
    print('sample image has ', sample['fps'], 'fps,', sample['resolution'], ' resolution,', sample['bitrate'], 'bps')
    print(f'sample velocity is {sample["velocity"]}') if VELOCITY else None
    print(f'sample path is {sample["path"]}') if VELOCITY else None

    device = get_default_device()
    cuda  = device.type == 'cuda'

    if START_TRAINING:
        # step2 split data and prepare batches
        # train_data, val_data = random_split(dataset,[train_size,val_size])
        # print(f"Length of Train Data : {len(train_data)}")
        # print(f"Length of Validation Data : {len(val_data)} \n")

        model = DecRefClassification(num_framerates, num_resolutions, VELOCITY=VELOCITY)
        # print(model)

        # dataloader gives batch
        train_dl = DataLoader(dataset, batch_size, shuffle = True, num_workers = 4, pin_memory = True)
        val_dl = DataLoader(val_dataset, batch_size*2, shuffle = True, num_workers = 4, pin_memory = True)
        
        if device.type == 'cuda':
            print(f'Loading data to cuda...')
            train_dl = DeviceDataLoader(train_dl, device)
            val_dl = DeviceDataLoader(val_dl, device)
            to_device(model, device)

        # Initialize TensorBoard writer
        # writer = SummaryWriter('runs/VRRML')
        # fitting the model on training data and record the result after each epoch
        history = fit(num_epochs, lr, model, train_dl, val_dl, opt_func, SAVE_MODEL=SAVE_MODEL, SAVE_HALFWAY=SAVE_MODEL_HALF_WAY, VELOCITY=VELOCITY)
        # writer.close()


    if TEST_EVAL:
        model = DecRefClassification(num_framerates, num_resolutions, VELOCITY=VELOCITY)
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


     