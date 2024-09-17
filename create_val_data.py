import os
import shutil
import random
# from utils import *



def move_random_images(src_folder, dest_folder, percentage=0.1):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    
    # List all image files in the source folder (assuming PNG or JPG files)
    all_images = [f for f in os.listdir(src_folder)]
    num_images_to_move = int(len(all_images) * percentage)
    selected_images = random.sample(all_images, num_images_to_move)
    print(f'all_images {len(all_images)}, selected_images {len(selected_images)}')

    for image in selected_images:
        src_path = os.path.join(src_folder, image)
        dest_path = os.path.join(dest_folder, image)
        if MOVE:
            shutil.move(src_path, dest_path)
    # print(f"Moved {num_images_to_move} images from {src_folder} to {dest_folder}.\n")
    print(f"Moved {num_images_to_move} images from to {dest_folder}.\n")



def create_validation_data(patch_dir, percentage=0.10):
    for path_folder in os.listdir(patch_dir):
        print(f'\n====================== path_folder {path_folder} ======================')
        path_folder_full = os.path.join(patch_dir, path_folder)
        print(f'{os.listdir(path_folder_full)}')
        # Iterate over the bitrate subfolders (e.g., 720x30x500)
        for bitrate_folder in os.listdir(path_folder_full):
            if bitrate_folder != '1000kbps':
                print(f'bitrate_folder {bitrate_folder} not valid')
                continue
            print(f'bitrate_folder {bitrate_folder} valid')

            # fps_target, resolution_target, bitrate = map(int, bitrate_folder.split('x'))
            bitrate_path = os.path.join(path_folder_full, bitrate_folder)
            
            # Collect all PNG files and their metadata
            dest_folder = f'{validation_data_dir}/{path_folder}/{bitrate_folder}'
            # print(f'bitrate_folder {bitrate_folder}')
            # print(f'dest_folder {dest_folder}')
            # move_random_images(bitrate_path, dest_folder, percentage=percentage)


if __name__ == "__main__":
    # patch_dir = f'{VRRML}/train' # scenes
    # validation_data_dir = f'{VRRML}/validation'
    MOVE = True # True False
    scene = 'suntemple_statue'
    base_dir = r'C:\Users\15142\Projects\VRR\Data\VRR_Patches'
    # patch_dir =  f'{base_dir}/2024-09-12/{scene}'

    # base_dir = '/home/yl962/rds/hpc-work/VRR/Data/VRR_Patches'
    # patch_dir =  f'{base_dir}/2024-09-12/{scene}'


    # cleaned_patches_dir = f'{base_dir}/HPC/cleaned_patches'
    # validation_data_dir = f'{cleaned_patches_dir}/{scene}'
    cleaned_patches_dir = f'{base_dir}/HPC/cleaned_patches'
    validation_data_dir = f'{base_dir}/2024-09-12/{scene}'
    patch_dir =  f'{cleaned_patches_dir}/{scene}'

    os.makedirs(validation_data_dir, exist_ok=True)

    create_validation_data(patch_dir, percentage=0.50)
