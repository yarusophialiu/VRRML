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

from VideoSinglePatchDataset_test import VideoSinglePatchDataset
from DeviceDataLoader import DeviceDataLoader
from utils import *
from DecRefClassification_test import *
from torch.utils.tensorboard import SummaryWriter


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
            fps = batch["fps"]
            bitrate = batch["bitrate"]
            resolution = batch["resolution"]
            velocity = batch["velocity"]
            res_targets = batch["res_targets"]
            fps_targets = batch["fps_targets"]
            path = batch["path"]
            # print(f'bitrate {bitrate.size()}')

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

            return {'val_loss': total_loss.detach(), 'res_acc': resolution_accuracy, 'fps_acc': framerate_accuracy, \
                    'both_acc': both_correct_accuracy, 'jod_loss': round(jod_loss, 3)}, res_out, fps_out, res_targets, fps_targets, \
                        res_values, fps_values, unique_indices


def fit(epochs, model, train_loader, val_loader, optimizer, \
                SAVE_MODEL=False, SAVE_HALFWAY=False, VELOCITY=False, CHECKPOINT=False):
    print(f'fit VELOCITY {VELOCITY}')
    history = []
    # optimizer = opt_func(model.parameters(),lr)
    model_path = ""
    if SAVE_MODEL or SAVE_HALFWAY:
        now = datetime.now()
        dir_pth = now.strftime("%Y-%m-%d")
        os.makedirs(dir_pth, exist_ok=True)
        hrmin = now.strftime("%H_%M")
        model_path = os.path.join(dir_pth, hrmin)

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
        writer.add_scalar('Loss/jod_loss', result['jod_loss'], epoch)

        # print(f'result \n {result}')
        result['train_loss'] = torch.stack(train_losses).mean().item()
        writer.add_scalar('Loss/train', result['train_loss'], epoch)
        # print(f'len(val_loader) {len(val_loader)}, avg_train_loss {avg_train_loss} {torch.stack(train_losses).mean().item()}')

        model.epoch_end(epoch, result)
        history.append(result)

        if SAVE_HALFWAY and epoch % 5 == 0 and epoch > 0:
            os.makedirs(model_path, exist_ok=True)
            print(f"Epoch {epoch} is a multiple of 10.")
            save_checkpoint(model, optimizer,  f'{model_path}/checkpoint{epoch}.pth', epoch)
        
    if SAVE_MODEL:
        os.makedirs(model_path, exist_ok=True)
        torch.save(model.state_dict(), f'{model_path}/classification.pth')
    print(f'model_path {model_path}')
    return history

# OLD from the last VRRML on desktop
# disable each parameter and test model performance 
if __name__ == "__main__":
    SAVE_MODEL = True
    SAVE_MODEL_HALF_WAY = True
    START_TRAINING= False # True False
    TEST_EVAL = True
    TEST_UNSEEN_SCENE = False # True
    
    model_pth_path = ""
    folder = 'ML/reference128x128' # TODO change model size reference128x128
    if TEST_UNSEEN_SCENE:
        print(f'test on unseen scenes')
        data_test_directory = f'{VRRML}/ML/test_scenes128x128' 
    else:
        data_test_directory = f'{VRRML}/{folder}/test'
    data_train_directory = f'{VRRML}/{folder}/train' # ML_smaller
    data_val_directory = f'{VRRML}/{folder}/validation'  

    if TEST_EVAL:
        model_pth_path = f'models/test_no_param/p128_b128_nores.pth' # patch128_batch128 patch256_batch64

    num_epochs = 13
    lr = 0.0003
    # opt_func = torch.optim.SGD
    opt_func = torch.optim.Adam
    batch_size = 128 # TODO
    patch_size = (128, 128) # TODO, change patch structure in DecRefClassification.py

    num_framerates, num_resolutions = 10, 5
    VELOCITY = True
    VALIDATION = True
    CHECKPOINT = False

    dataset = VideoSinglePatchDataset(directory=data_train_directory, min_bitrate=500, max_bitrate=2000, patch_size=patch_size, VELOCITY=VELOCITY) # len 27592
    val_dataset = VideoSinglePatchDataset(directory=data_val_directory, min_bitrate=500, max_bitrate=2000, patch_size=patch_size, VELOCITY=VELOCITY, VALIDATION=VALIDATION) # len 27592
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
        model = DecRefClassification(num_framerates, num_resolutions, VELOCITY=VELOCITY)
        optimizer = opt_func(model.parameters(),lr)
        epochs = range(num_epochs)
        if CHECKPOINT:
            checkpoint = torch.load('2024-09-26/15_19/checkpoint10.pth')
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # epoch = checkpoint['epoch']
            epoch = 10
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

        writer = SummaryWriter('runs/test_param')
        # fitting the model on training data and record the result after each epoch
        history = fit(epochs, model, train_dl, val_dl, optimizer, SAVE_MODEL=SAVE_MODEL, \
                        SAVE_HALFWAY=SAVE_MODEL_HALF_WAY, VELOCITY=VELOCITY, CHECKPOINT=CHECKPOINT)
        writer.close()


    if TEST_EVAL: # test data size 35756
        test_dataset = VideoSinglePatchDataset(directory=data_test_directory, min_bitrate=500, max_bitrate=2000, patch_size=patch_size, VELOCITY=VELOCITY, VALIDATION=VALIDATION) # len 27592
        print(f'\ntest_size {len(test_dataset)}, patch_size {patch_size}, batch_size {batch_size}\n')
        model = DecRefClassification(num_framerates, num_resolutions, VELOCITY=VELOCITY)
        # model_pth_path = f'models/patch128_batch256.pth' # patch128_batch128 patch256_batch64
        model.load_state_dict(torch.load(model_pth_path))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # model.to(device)
        # test_dl = DataLoader(test_dataset, len(test_dataset))
        test_dl = DataLoader(test_dataset, batch_size*2, shuffle = True, num_workers = 4, pin_memory = True)

        if device.type == 'cuda':
            print(f'Loading data to cuda...')
            test_dl = DeviceDataLoader(test_dl, device)
            to_device(model, device)

        print(f'model_path {model_pth_path}')
        result, res_out, fps_out, res_targets, fps_targets, \
                        res_values, fps_values, unique_indices = evaluate_test_data(model, test_dl)
        
        # get mean absolute error and mean absolute percentage error
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
        # print(f'predicted_res {predicted_res}')
        # print(f'target_res {target_res}')

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

     