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
# from datetime import date
import datetime

from VideoSinglePatchDataset import VideoSinglePatchDataset
from DeviceDataLoader import DeviceDataLoader
from utils import *
from DecRefClassification import *
from torch.utils.tensorboard import SummaryWriter



mean_velocity = 341011.652
std_velocity = 3676701.584
framerate_ticks = [30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
resolution_ticks = [360, 480, 720, 864, 1080]

def evaluate_test_data(model, test_loader):
    model.eval()
    with torch.no_grad():  # Ensure gradients are not computed
        result = {'test_losses': [], 'res_acc': [], 'fps_acc': [], 'both_acc': [], 'jod_loss': []}
        velocities = []
        framenumbers = []
        round = 0
        for batch in test_loader:
            # print(f'batch round {round}')
            round += 1
            images = batch["image"]
            # print(f'images {images.size()}')
            fps = batch["fps"]
            bitrate = batch["bitrate"]
            resolution = batch["resolution"]
            velocity = batch["velocity"]
            res_targets = batch["res_targets"]
            fps_targets = batch["fps_targets"]
            path = batch["path"]
            framenumber = batch["framenumber"]
            # print(f'fps {fps.size()}')
            # print(f'framenumber {framenumber.size()}')

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

            result['test_losses'].append(total_loss)
            result['fps_acc'].append(framerate_accuracy)
            result['res_acc'].append(resolution_accuracy)
            result['both_acc'].append(both_correct_accuracy)
            result['jod_loss'].append(jod_loss)

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

            # TODO: unnormalize and return velocity
            unnormalized_velocity = unnormalize_z_value(velocity, mean_velocity, std_velocity)
            # print(f'unnormalized_velocity {unnormalized_velocity}')
            velocities.append(unnormalized_velocity.cpu())
            framenumbers.append(original_framenumber.cpu())
        # print(f'result {result}')
        epoch_test_losses = torch.stack(result['test_losses']).mean() # Combine accuracies
        epoch_res_acc = torch.stack(result['res_acc']).mean() # Combine accuracies
        epoch_fps_acc = torch.stack(result['fps_acc']).mean()
        epoch_both_acc = torch.stack(result['both_acc']).mean()
        epoch_jod_loss = sum(result['jod_loss']) / len(result['jod_loss'])
        # print(f'batch_jod_loss {batch_jod_loss}')
        return {'test_losses': epoch_test_losses.item(), 'res_acc': epoch_res_acc.item(), \
                'fps_acc': epoch_fps_acc.item(), 'both_acc': epoch_both_acc.item(), 'jod_loss': epoch_jod_loss}, \
                res_out, fps_out, res_targets, fps_targets, \
                res_values, fps_values, framenumbers, velocities, unique_indices

        # return {'test_losses': total_loss.detach(), 'res_acc': resolution_accuracy, 'fps_acc': framerate_accuracy, \
        #             'both_acc': both_correct_accuracy, 'jod_loss': round(jod_loss, 3)}, res_out, fps_out, res_targets, fps_targets, \
        #                 res_values, fps_values, original_framenumber, unnormalized_velocity, unique_indices


def print_error(predicted_fps, predicted_res, target_fps, target_res):
    absolute_errors_fps = torch.abs(predicted_fps - target_fps)
    absolute_errors_res = torch.abs(predicted_res - target_res)
    expected_error_fps = torch.mean(absolute_errors_fps.float())
    expected_error_res = torch.mean(absolute_errors_res.float())

    # Compute the absolute percentage error
    percentage_errors_fps = torch.abs((predicted_fps - target_fps) / target_fps) * 100
    percentage_errors_res = torch.abs((predicted_res - target_res) / target_res) * 100
    # print(f"percentage_errors_res: {percentage_errors_res}\n") 

    # Compute the mean absolute percentage error (MAPE)
    mape_fps = torch.mean(percentage_errors_fps.float())
    mape_res = torch.mean(percentage_errors_res.float())
    print(f"FPS: Mean Absolute Error {expected_error_fps.item()}, Mean Absolute Percentage Error (MAPE): {round(mape_fps.item(), 3)}%")
    print(f"Resolution: Mean Absolute Error {expected_error_res.item()}, Mean Absolute Percentage Error (MAPE): {round(mape_res.item(), 3)}%\n")
    print(f'test result \n {result}\n')


# x axis is framenumber
if __name__ == "__main__":
    PLOT_TEST_RESULT = True
    SHOW_PLOT = True
    SAVE_PLOT = False
    TEST_UNSEEN_SCENE = False # True False
    SHOW_ERROR = False
    RUN_INFERENCE = True
    SHOW_VELOCITY = False

    scene = 'bistro'
    bitrate = 500
    model_pth_path = f'models/patch128-256/patch128_batch128.pth' # patch128_batch128 patch256_batch64
    folder = 'ML_smaller/reference128x128' # TODO change model size reference128x128
    path = 'path1_seg1_1'
    if TEST_UNSEEN_SCENE:
        print(f'test on unseen scenes')
        data_test_directory = f'{VRRML}/ML/test_scenes128x128'  # f'{VRRML}/ML/test_scenes128x128' f'{VRR_Patches}/reference_living_room_128x128'
    else:
        data_test_directory = f'{VRR_Patches}/reference_{scene}_128x128' # f'{VRRML}/{folder}/test_demo'

    opt_func = torch.optim.Adam
    batch_size = 276 # TODO
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

    # if RUN_INFERENCE:
    model = DecRefClassification(num_framerates, num_resolutions, VELOCITY=VELOCITY)
    model.load_state_dict(torch.load(model_pth_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dl = DataLoader(test_dataset, len(test_dataset), shuffle = True, num_workers = 4, pin_memory = True)

    if device.type == 'cuda':
        print(f'Loading data to cuda...')
        test_dl = DeviceDataLoader(test_dl, device)
        to_device(model, device)

    print(f'model_path {model_pth_path}')
    result, res_out, fps_out, res_targets, fps_targets, \
        res_values, fps_values, framenumber, velocity, unique_indices = evaluate_test_data(model, test_dl)
    _, fps_preds = torch.max(fps_out, dim=1)
    _, res_preds = torch.max(res_out, dim=1)
    print(f'result {result}')
    # print(f'res_targets {res_targets}\n')
    
    reverse_fps_map = {v: k for k, v in fps_map.items()}
    reverse_res_map = {v: k for k, v in res_map.items()}

    # Convert the predicted and target indices to actual values
    predicted_fps = torch.tensor([reverse_fps_map[int(pred)] for pred in fps_preds])
    target_fps = torch.tensor([reverse_fps_map[int(target)] for target in fps_targets])

    predicted_res = torch.tensor([reverse_res_map[int(pred)] for pred in res_preds])
    target_res = torch.tensor([reverse_res_map[int(target)] for target in res_targets])
    # print(f'predicted_fps {predicted_fps}')
    # print(f'predicted_res {predicted_res}\n')
    # print(f'target_fps {target_fps}')
    # print(f'target_res {target_res}\n')
    # print(f'framenumber {framenumber}\n')
    # print(f'velocity {velocity}\n')
    # predicted_fps = torch.tensor([120, 120, 120, 120,  80,  80,  50,  60, 120,  90, 120, 110, 120,  50, 120, 120])
    # predicted_res = torch.tensor([ 720,  720,  720,  720, 1080, 1080,  720,  720,  720,  720,  720,  720, 720,  720,  720,  720])
    # framenumber = torch.tensor([49., 137., 204., 133., 140., 60., 172., 114., 106., 42., 126., 111., 53., 166., 112., 73.], dtype=torch.float64)
    # velocity = torch.tensor([5.3, 7.4, 6.1, 4.8, 3.9, 8.2, 6.9, 5.7, 6.3, 5.1, 4.7, 6.5, 7.1, 8.0, 6.4, 7.8])  # Example velocity data

    if PLOT_TEST_RESULT:
        framenumber = framenumber
        predicted_fps = predicted_fps.cpu().numpy()
        predicted_res = predicted_res.cpu().numpy()
        velocity = velocity

        # Scatter plot for framerate vs framenumber
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.scatter(framenumber, predicted_fps, color='b', label='Framerate (FPS)')
        
        # Annotate velocity values on framerate scatter plot
        if SHOW_VELOCITY:
            for i in range(len(framenumber)):
                plt.annotate(f'{velocity[i]:.1f}', (framenumber[i], predicted_fps[i]),  xytext=(framenumber[i], predicted_fps[i]))  

        plt.xlabel('Framenumber')
        plt.ylabel('Framerate (FPS)')
        plt.title(f'scene {scene} - path {path} - bitrate {bitrate}kbps \nFramerate vs Framenumber (Scatter Plot)')
        plt.yticks(framerate_ticks)
        plt.grid(True)
        plt.legend()

        # Scatter plot for resolution vs framenumber
        plt.subplot(2, 1, 2)
        plt.scatter(framenumber, predicted_res, color='g', label='Resolution (p)')
        if SHOW_VELOCITY:
            for i in range(len(framenumber)):
                plt.annotate(f'{velocity[i]:.1f}', (framenumber[i], predicted_res[i]), xytext=(framenumber[i], predicted_res[i]))

        plt.xlabel('Framenumber')
        plt.ylabel('Resolution (p)')
        plt.title('Resolution vs Framenumber (Scatter Plot)')
        plt.yticks(resolution_ticks)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        # plt.show()

        if SAVE_PLOT:
            # current_date = datetime.today()
            now = datetime.now()
            current_date = now.strftime("%Y-%m-%d")
            # print(f'current_date {current_date}')
            os.makedirs(current_date, exist_ok=True)
            plt.savefig(f'{current_date}/{path}_{bitrate}.png')
            print(f'saved successfully to {current_date}/{path}_{bitrate}.png')

        if SHOW_PLOT:
            plt.show()

    if SHOW_ERROR:
        print_error(predicted_fps, predicted_res, target_fps, target_res)