import cv2
import numpy as np
import os
import shutil
import secrets
from utils import *
from utils import *
import datetime
import torch
import torchvision.transforms as transforms




scene_velocity_dicts = {
    # 'bistro': bistro_max_comb_per_sequence,
    # 'suntemple': suntemple_max_comb_per_sequence,
    'suntemple_statue': suntemple_statue_max_comb_per_sequence
}

bitrates = [500, 1000, 1500, 2000]


def rename_subfolders_for_scene(scene, velocity_dict, base_dir):
    # Get the path to the scene folder
    scene_folder = os.path.join(base_dir, scene)
    
    if not os.path.exists(scene_folder):
        print(f"Scene folder '{scene_folder}' does not exist. Skipping.")
        return
    
    for sequence_name, params_list in velocity_dict.items(): # bistro_path1_seg1_1: [[xxx], [xxx], [xxx]]
        sequence_path = f'{scene_folder}/{scene}_{sequence_name}'
        if not os.path.exists(sequence_path):
            # print(f"Folder '{sequence_path}' does not exist. Skipping.")
            continue
            
        for i, bitrate in enumerate(bitrates):
            print(f'bitrate {bitrate}')
            # Subfolder to find, e.g., '500kbps', '1000kbps'
            # old_folder_name = f"{bitrate}kbps"
            # old_folder_path = os.path.join(scene_folder, old_folder_name)
            old_folder_path = f'{sequence_path}/{bitrate}kbps'

            print(f'old_folder_path {old_folder_path}')
            if not os.path.exists(old_folder_path):
                print(f"Folder '{old_folder_path}' does not exist. Skipping.")
                continue
            
            # Get the optimal parameters for the current bitrate
            # optimal_params = params_list[i]
            optimal_fps, optimal_resolution = params_list[i]
            print(f'optimal_fps, optimal_resolution {optimal_fps, optimal_resolution}')
            new_folder_name = f"{optimal_resolution}x{optimal_fps}x{bitrate}"
            new_folder_path = os.path.join(sequence_path, new_folder_name)
            
            # Rename the folder
            print(f"Renaming to '{new_folder_path}'")
            shutil.move(old_folder_path, new_folder_path)




if __name__ == "__main__":
    patch_dir = f'{VRR_Patches}/test' # scenes
    training_data_dir = f'{VRRML}/train'

    # Iterate over each scene and rename subfolders
    for scene, velocity_dict in scene_velocity_dicts.items():
        print(f'=============== scene {scene} ===============')
        rename_subfolders_for_scene(scene, velocity_dict, patch_dir)
    # for each path e.g. suntemple_statue_path1_seg1_1, find optimal label for each bitrate


    # scene_arr = ['suntemple_statue']



