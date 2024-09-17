import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from utils import *

class PatchDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path, metadata = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, metadata

def load_data_from_path(path_folder, validation_split=0.10):
    images_and_metadata = []

    # Iterate over the bitrate subfolders (e.g., 720x30x500)
    for bitrate_folder in os.listdir(path_folder):
        fps_target, resolution_target, bitrate = map(int, bitrate_folder.split('x'))
        bitrate_path = os.path.join(path_folder, bitrate_folder)
        
        # Collect all PNG files and their metadata
        all_pngs = []
        for image_name in os.listdir(bitrate_path):
            image_path = os.path.join(bitrate_path, image_name)
            
            # Parse the image filename to extract metadata
            image_id, fps, resolution, image_bitrate, velocity = image_name.split('_')
            fps, resolution, image_bitrate, velocity = int(fps), int(resolution), int(image_bitrate), int(velocity.split('.')[0])
            
            metadata = {
                'fps_target': fps_target,
                'resolution_target': resolution_target,
                'fps': fps,
                'resolution': resolution,
                'bitrate': image_bitrate,
                'velocity': velocity/1000,
                # 'bitrate_folder': bitrate_folder,
            }
            all_pngs.append((image_path, metadata))
    
        # Split the PNGs into 85% training and 15% validation
        num_validation = int(validation_split * len(all_pngs))
        # print(f'num_validation {num_validation}')
        random.shuffle(all_pngs)  # Shuffle to randomize the split
        validation_pngs = all_pngs[:num_validation] # arrays
        training_pngs = all_pngs[num_validation:]

        images_and_metadata.append((training_pngs, validation_pngs))

    return images_and_metadata














# # Get all path folders (e.g., bistro_path1_seg1_1, bistro_path1_seg2_1)
# root_dir = f'{VRRML}/train/'  # Your root directory
# all_path_folders = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
# print(f'all_path_folders {all_path_folders}')


# # Collect all training and validation data from each path folder
# train_images = []
# val_images = []
# for path_folder in all_path_folders:
#     data = load_data_from_path(path_folder) # data is a array with 2 subarrays, each sub is array (img_path, metadata)
#     for training_pngs, validation_pngs in data:
#         train_images.extend(training_pngs)
#         val_images.extend(validation_pngs)
# # train_images is array of tuples (image path, metadata)
# print(f'train_images {len(train_images)}')
# # Create dataset instances for training and validation
# train_dataset = PatchDataset(train_images, transform=transform)
# val_dataset = PatchDataset(val_images, transform=transform)

# # Create DataLoader instances
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# # Check dataset sizes
# print(f"Training dataset size: {len(train_dataset)} images")
# print(f"Validation dataset size: {len(val_dataset)} images")

# # Example training loop
# for epoch in range(1):
#     print(f"Epoch {epoch + 1}")
    
#     # Training phase
#     for mini_batch_idx, (images, metadata) in enumerate(train_loader):
#         print(f'mini_batch_idx {mini_batch_idx}')
#         print(f'images {images.shape}')
#         # print(f'metadata {metadata["bitrate_folder"]}')
#         # Process training images and metadata
#         pass  # Replace with training logic
    
    # # Validation phase
    # # with torch.no_grad():
    # for mini_batch, (images, metadata) in enumerate(val_loader):
    #     # Process validation images and metadata
    #     pass  # Replace with validation logic
