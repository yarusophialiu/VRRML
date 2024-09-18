import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
from PIL import Image
from utils import *
from DeviceDataLoader import DeviceDataLoader
from torch.utils.data import Dataset



class PatchDataset(Dataset):
    def __init__(self, root_dir, image_memmap_file, metadata_memmap_file, num_patches, num_patches_per_subfolder, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.num_patches = num_patches
        self.num_patches_per_subfolder = num_patches_per_subfolder

        self.image_memmap = np.memmap(image_memmap_file, dtype='uint8', mode='r', shape=(num_patches, *image_shape))
        self.metadata_memmap = np.memmap(metadata_memmap_file, dtype='float32', mode='r', shape=(num_patches, 6))  # 6 metadata fields

        # # Get all subfolder names (e.g., bistro_path1_seg2_2_480_110_500)
        # self.path_bitrate_folders = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

    def __len__(self):
        # determine idx range in getitem
        # needs to be number of path_bitrate folders
        return int(self.num_patches / self.num_patches_per_subfolder)
    

    # if batch size = 1, return 1 __getitem__ 
    # if batch size = 2, return 2 __getitem__
    def __getitem__(self, idx):
        # print(f'\n=================== idx {idx} ===================')
        # subfolder = self.path_bitrate_folders[idx]
        images, metadata = [], []
        start_idx = idx * self.num_patches_per_subfolder
        end_idx = start_idx + self.num_patches_per_subfolder
        # print(f'subfolder {subfolder} ')
        # print(f'start_idx - end_idx {start_idx}-{end_idx} ')

        for i in range(start_idx, end_idx):
            image_data = self.image_memmap[i]
            image = Image.fromarray(image_data)
            # if self.transform:
            #     print(f'transformation')
            #     image = self.transform(image)
            images.append(image)
            metadata.append(self.metadata_memmap[i])

        combined_np_array = np.array(images)
        images_tensor = torch.from_numpy(combined_np_array) # torch.Size([200, 64, 64, 3])
        images_tensor = images_tensor.permute(0, 3, 1, 2)

        combined_metadata_array = np.array(metadata)
        metadata_tensor = torch.from_numpy(combined_metadata_array) # torch.Size([200, 6])

        return images_tensor, metadata_tensor, idx


# structure is like bistro_path2_seg3_1/1080x60x1500
# use np.memmap to accelerate data loading for batches
# by loading data incrementally from disk rather than keeping everything in memory
if __name__ == "__main__":
    # Initialize dataset
    image_memmap_file  = f'{VRRML}/ML_smaller/train_bitratelabel.dat'
    metadata_memmap_file   = f'{VRRML}/ML_smaller/train_bitratelabel_metadata_normalize.dat'
    image_shape = (64, 64, 3)

    root_dir = f'{VRRML}/ML_smaller/train_bitratelabel/'  # Path to the train folder
    total_number_of_patches = 1200
    num_patches_per_subfolder = 200
    # memmap_shape = (total_number_of_patches, 64, 64, 3)  # Assuming images are 64x64 RGB
    # dataset = PatchDataset(root_dir=root_dir, memmap_file=memmap_file, memmap_shape=memmap_shape, transform=transform)
    # dataloader = DataLoader(dataset, batch_size=1, shuffle=True)  # One path folder per batch
    
    # call init() only, getitem not called
    # The DataLoader will use the dataset’s __getitem__() method to retrieve individual samples for each batch as you iterate over it.
    dataset = PatchDataset(root_dir, image_memmap_file, metadata_memmap_file, total_number_of_patches, num_patches_per_subfolder, image_shape)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4, pin_memory=True)

    fps_map = {30: 0, 40: 1, 50: 2, 60: 3, 70: 4, 80: 5, 90: 6, 100: 7, 110: 8, 120: 9}
    res_map = {360: 0, 480: 1, 720: 2, 864: 3, 1080: 4}
    max_velocity = 51918288
    min_velocity = 0.022
    mean_velocity = 341011.652
    std_velocity = 3676701.584

    device = get_default_device()
    cuda  = device.type == 'cuda'

    if device.type == 'cuda':
        print(f'Loading data to cuda...')
        train_dl = DeviceDataLoader(dataloader, device)

    # model = DecRefClassification(num_framerates, num_resolutions, VELOCITY=VELOCITY)
    # Training loop
    for epoch in range(1):
        print(f"================= Epoch {epoch + 1} =================")
        # for mini_batch_idx, (images, metadata, fps_target, res_target, bitrate) in enumerate(train_dl): # dataloader
        for mini_batch_idx, (images, metadata, idx) in enumerate(dataloader): # dataloader, train_dl
            print(f'============== batch {mini_batch_idx} ==============')
            # print(f"mini_batch_idx: {mini_batch_idx}")
            # images size：torch.Size([2, 200, 64, 64, 3]) (batch_size, num_patches, channels, height, width
            # print(f"images size: {images.size()}")  # (batch_size, 10000, 3, 64, 64) if 10000 images per folder
            # print(f"metadata: {metadata.size()} ") #  \n {metadata}
            # print(f"idx: {idx} ")
            show_patches(images[0], num_patches=25)

            # Your training logic here...

