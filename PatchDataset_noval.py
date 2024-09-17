import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from utils import *
from DeviceDataLoader import DeviceDataLoader




# path folder, bitrate folder, pngs
# each path folder has 4 bitrate folder like 720x90x500
# each bitrate folder has 2500 pngs, 1 path folder has 10k pngs
# if batchsize = n, each mini batch has n * 10k pngs

fps_map = {30: 0, 40: 1, 50: 2, 60: 3, 70: 4, 80: 5, 90: 6, 100: 7, 110: 8, 120: 9}
res_map = {360: 0, 480: 1, 720: 2, 864: 3, 1080: 4}
max_velocity = 51918288
min_velocity = 0.022
mean_velocity = 341011.652
std_velocity = 3676701.584


class PatchDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        # print(f'root_dir {root_dir}')
        self.root_dir = root_dir
        self.transform = transform
        
        # Get all path folders (e.g., bistro_path1_seg1_1, bistro_path1_seg2_1)
        self.path_folders = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        # print(f'self.path_folders \n {self.path_folders}')
    
    def __len__(self):
        # Each path folder represents one batch, so the length is the number of path folders
        return len(self.path_folders)



    def normalize(self, sample, data):
        # print(f'sample {sample}')
        # print(f'data {data}')
        mean = np.mean(data)
        std_dev = np.std(data)
        sample = (sample - mean) / std_dev
        return round(sample, 3)
    

    # if batch size = 1, return 1 __getitem__ 
    # if batch size = 2, return 2 __getitem__
    # e.g. DataLoader(dataset, batch_size=2, shuffle=True)
    def __getitem__(self, idx):
        path_folder = self.path_folders[idx]
        print(f'path_folder {path_folder}')
        images = []
        metadata = []

        # Collect all images and metadata from the folder's bitrate subfolders
        for bitrate_folder in os.listdir(path_folder):
            # Parse the bitrate folder name, e.g., '720x30x500'
            resolution_target, fps_target, bitrate = map(int, bitrate_folder.split('x'))
            # print(f'fps_target, resolution_target, bitrate {fps_target, resolution_target, bitrate}')
            bitrate_path = os.path.join(path_folder, bitrate_folder)
            
            for image_name in os.listdir(bitrate_path):
                image_path = os.path.join(bitrate_path, image_name)
                
                # Parse the image name, e.g., '00c9a505_90_480_500_241.png'
                image_id, fps, resolution, image_bitrate, velocity = image_name.split('_')
                fps, resolution, image_bitrate, velocity = int(fps), int(resolution), int(image_bitrate), int(velocity.split('.')[0])

                image = Image.open(image_path).convert('RGB')
                if self.transform:
                    image = self.transform(image)

                images.append(image)
                
                # fps_data = [30, 40, 50, 60, 70, 80, 90, 100, 110, 120,]
                # resolution_data = [360, 480, 720, 864, 1080]
                # bitrates_data = [500, 1000, 1500, 2000]
                metadata.append([fps, resolution, fps_target, resolution_target, image_bitrate, velocity/1000])

        # Stack the images into a single tensor (batch_size, C, H, W)
        images_tensor = torch.stack(images)
        metadata = torch.tensor(metadata) # (9000, 4)

        # fps_data = [30, 40, 50, 60, 70, 80, 90, 100, 110, 120,]
        # resolution_data = [360, 480, 720, 864, 1080]
        # bitrate_data = [500, 1000, 1500, 2000]

        fps_column = metadata[:, 0]
        resolution_column = metadata[:, 1]
        fps_target_column = metadata[:, 2].contiguous()
        res_target_column = metadata[:, 3].contiguous()
        bitrate_column = metadata[:, 4]
        velocity_column = metadata[:, 5]
        # print(f'fps_column {fps_column}')
   
        #  normalization affects the input data to the network but not the target labels
        normalized_fps_column = normalize_max_min(fps_column, 30, 120)
        normalized_res_column = normalize_max_min(resolution_column, 360, 1080)
        normalized_bitrate_column = normalize_max_min(bitrate_column, 500, 2000)
        normalized_velocity_column = normalize_z_value(velocity_column, mean_velocity, std_velocity)

        # Update the original metadata tensor with the normalized values
        metadata[:, 0] = normalized_fps_column
        metadata[:, 1] = normalized_res_column
        metadata[:, 4] = normalized_bitrate_column
        metadata[:, 5] = normalized_velocity_column

        fps_keys = torch.tensor(list(fps_map.keys()))  # Tensor of resolution values
        res_keys = torch.tensor(list(res_map.keys()))  # Tensor of resolution values

        # Use torch.searchsorted to find the index in res_keys for each resolution
        # target values correspond to the class indices based on the mapping
        fps_indices = torch.searchsorted(fps_keys, fps_target_column)
        res_indices = torch.searchsorted(res_keys, res_target_column)

        # print(f'normalized_fps_column {normalized_fps_column}')
        # print(f'fps_indices {fps_indices}')
        # print(f'res_indices {res_indices}')

        metadata[:, 2] = fps_indices
        metadata[:, 3] = res_indices

        # print(f'metadata \n {metadata}\n\n\n')

        # image_tensor size (1, 10000, 64, 64), metadata list of 10000 elements
        return images_tensor, metadata # fps_target, resolution_target, image_bitrate



if __name__ == "__main__":
    # Initialize dataset
    root_dir = f'{VRRML}/train/'  # Path to the train folder
    dataset = PatchDataset(root_dir=root_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)  # One path folder per batch


    device = get_default_device()
    cuda  = device.type == 'cuda'
    print(f'cuda {cuda}')

    if device.type == 'cuda':
        print(f'Loading data to cuda...')
        train_dl = DeviceDataLoader(dataloader, device)

    # model = DecRefClassification(num_framerates, num_resolutions, VELOCITY=VELOCITY)
    # Training loop
    for epoch in range(1):
        print(f"================= Epoch {epoch + 1} =================")
        # for mini_batch_idx, (images, metadata, fps_target, res_target, bitrate) in enumerate(train_dl): # dataloader
        for mini_batch_idx, (images, metadata) in enumerate(train_dl): # dataloader
            print(f'============== batch {mini_batch_idx} ==============')
            print(f"mini_batch_idx: {mini_batch_idx}")
            print(f"images size: {images.size()}\n")  # (batch_size, 10000, 3, 64, 64) if 10000 images per folder
            print(f"metadata: {metadata.size()}") #  \n {metadata}
            # print(f"fps_target: {fps_target}, res_target: {res_target}, bitrate {bitrate}")

            # (batch_size, 10000, 3, 64, 64) if 10000 images per folder, dataloader only shuffles on batch_size
            # TODO: shuffle images before fit
            # Your training logic here...

