import numpy as np
from PIL import Image
from utils import *
import os

def create_memmap(root_dir, memmap_file, memmap_shape):
    # Initialize memmap in write mode
    # memmap is array of patches
    memmap = np.memmap(memmap_file, dtype='uint8', mode='w+', shape=memmap_shape)

    # Store images in memmap
    image_idx = 0
    for path_bitrate_folder in os.listdir(root_dir):
        full_path = os.path.join(root_dir, path_bitrate_folder)
        if os.path.isdir(full_path):
            for image_name in os.listdir(full_path):
                image_path = os.path.join(full_path, image_name)
                image = Image.open(image_path).convert('RGB')
                image_data = np.array(image)
                
                # Store image in memmap
                memmap[image_idx] = image_data
                image_idx += 1

    memmap.flush()  # Ensure data is written to disk



def create_metadata_memmap(root_dir, memmap_file, metadata_shape):
    # Initialize memmap for metadata
    metadata_memmap = np.memmap(memmap_file, dtype='float32', mode='w+', shape=metadata_shape)

    patch_idx = 0
    for path_bitrate_folder in os.listdir(root_dir):
        print(f'\npath_bitrate_folder {path_bitrate_folder}')
        resolution_target, fps_target, bitrate = path_bitrate_folder.split('_')[-3:]
        resolution_target, fps_target  = int(resolution_target), int(fps_target)
        # print(f'resolution_target, fps_target, bitrate {resolution_target, fps_target, bitrate}')

        full_path = os.path.join(root_dir, path_bitrate_folder)
        if os.path.isdir(full_path):
            for image_name in os.listdir(full_path):
                # Parse the image name to extract metadata (fps, resolution, bitrate, etc.)
                _, fps, resolution, bitrate, velocity_with_extension  = image_name.split('_')
                velocity = velocity_with_extension.split('.')[0] 
                fps, resolution, bitrate, velocity = int(fps), int(resolution), int(bitrate), float(velocity.split('.')[0])
                velocity /= 1000
                # Store metadata in the memmap
                # TODO process metadata
                # use 1 folder to testify the metadata is correct
                # print(f'fps, resolution, fps_target, resolution_target, bitrate, velocity \n {fps, resolution, fps_target, resolution_target, bitrate, velocity}')
                resolution = round(normalize_max_min(resolution, 360, 1080), 3)
                fps = round(normalize_max_min(fps, 30, 120), 3)
                bitrate = round(normalize_max_min(bitrate, 500, 2000) , 3)
                fps_target_label = fps_map[fps_target]
                res_target_label = res_map[resolution_target]
                velocity = round(normalize_z_value(velocity, mean_velocity, std_velocity), 3)
                # print(f'fps, resolution, fps_target, resolution_target, bitrate, velocity \n {fps, resolution, fps_target_label, res_target_label, bitrate, velocity}')
                metadata_memmap[patch_idx] = [fps, resolution, fps_target_label, res_target_label, bitrate, velocity]  # Add other metadata as needed
                
                patch_idx += 1
                # if patch_idx > 2:
                #     break

    metadata_memmap.flush()  # Ensure the data is written to disk
    return metadata_memmap



if __name__ == "__main__":
    fps_map = {30: 0, 40: 1, 50: 2, 60: 3, 70: 4, 80: 5, 90: 6, 100: 7, 110: 8, 120: 9}
    res_map = {360: 0, 480: 1, 720: 2, 864: 3, 1080: 4}
    mean_velocity = 341011.652
    std_velocity = 3676701.584

    base_dir = r'C:\Users\15142\Projects\VRR\Data\VRRML\ML_smaller'
    root_dir = f'{base_dir}/train_bitratelabel'
    memmap_file = f'{base_dir}/train_bitratelabel.dat'
    metadata_memmap_file = f'{base_dir}/train_bitratelabel_metadata_normalize.dat'
    total_number_of_patches = 1200
    memmap_shape = (total_number_of_patches, 64, 64, 3)  # Assuming images are 64x64 RGB
    metadata_shape = (total_number_of_patches, 6) # 6 metadata fields
    # create_memmap(root_dir, memmap_file, memmap_shape)
    create_metadata_memmap(root_dir, metadata_memmap_file, metadata_shape)
