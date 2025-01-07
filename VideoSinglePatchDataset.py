
import os 
import torch
from torchvision.transforms.functional import to_pil_image
from torch.utils.data import random_split
from torchvision.utils import make_grid
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from utils import *


fps_map = {30: 0, 40: 1, 50: 2, 60: 3, 70: 4, 80: 5, 90: 6, 100: 7, 110: 8, 120: 9}
res_map = {360: 0, 480: 1, 720: 2, 864: 3, 1080: 4}
mean_velocity = 341011.652
std_velocity = 3676701.584

# dataset: handling batching, shuffling, and iterations over the dataset during training or inference
class VideoSinglePatchDataset(Dataset):
    def __init__(self, directory, min_bitrate, max_bitrate, patch_size=((64, 64)), VELOCITY=False, VALIDATION=False, FRAMENUMBER=False):
        self.root_directory = directory
        self.patch_size = patch_size
        self.velocity = VELOCITY
        self.samples = []  # To store tuples of (image path, label)
        labels = os.listdir(directory)

        self.fps_targets = [int(label.split('x')[1]) for label in labels]
        self.res_targets = [int(label.split('x')[0]) for label in labels]

        self.min_fps = 30
        self.max_fps = 120
        self.min_res = 360
        self.max_res = 1080

        self.min_bitrate = min_bitrate
        self.max_bitrate = max_bitrate
        self.validation = VALIDATION
        self.framenumber = FRAMENUMBER

        # print(f'TYPE {TYPE}')
        # print(f'self.min_bitrate, self.max_bitrate {self.min_bitrate, self.max_bitrate}')
        self.transform = transforms.Compose([
                    # CustomTransform(patch_size, TYPE) ,
                    transforms.Resize(patch_size),  # Resize images to 64x64
                    transforms.ToTensor(),  # Convert images to PyTorch tensors
                ])    

        for label in labels: 
                label_dir = os.path.join(directory, str(label))
                # print(f'label_dir {label_dir}')
                for root, _, filenames in os.walk(label_dir):
                    for filename in filenames:
                        file_path = os.path.join(root, filename)
                        self.samples.append((file_path, label))   
        # print(f'self.samples {len(self.samples)}\n')

        
    def __len__(self):
        return len(self.samples)
    
    def normalize(self, sample, min_vals, max_vals):
        # print(f'val, min_vals, max_vals {sample, min_vals, max_vals}')
        sample = (sample - min_vals) / (max_vals - min_vals)
        return round(sample, 3)
    

    
    # load individual data sample, apply transformations, 
    # if batch size = 1, return 1 __getitem__ 
    # if batch size = 2, return 2 __getitem__ 
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        # print(f'img_path {img_path}')

        fps_targets = int(label.split('x')[1])
        res_targets = int(label.split('x')[0])

        filename = os.path.basename(img_path) # 00a0f6e8_50_864_2000.png or 00cb4b2e_40_864_2000_776102.png
        parts = filename.split('_')     
        fps = float(parts[1])
        pixel = int(parts[2])  
        # velocity = 0
        if not self.velocity:
            bitrate = int(parts[-1].split('.')[0])  # Remove .png and convert to integer
        else:
            bitrate = int(parts[3])  
            velocity = int(parts[-1].split('.')[0]) / 1000  # Remove .png and convert to integer
        
        if self.framenumber:
            framenumber = int(parts[4])


        fps = self.normalize(fps, self.min_fps, self.max_fps)
        pixel = self.normalize(pixel, self.min_res, self.max_res)
        bitrate = self.normalize(bitrate, self.min_bitrate, self.max_bitrate)
        # TODO: normalize velocity
        velocity = round(normalize_z_value(velocity, mean_velocity, std_velocity), 3)
        framenumber = self.normalize(framenumber, 0, 276) if self.framenumber else -1


        image = Image.open(img_path)
        # print(f'\n\n\nimage.mode ', image.mode)
        # image = image.convert('RGBA') # convert from 3 to 4 chanel
        # print(f'\n\n\nimage.mode ', image.mode)

        if self.transform:
            image = self.transform(image)
            # print(f'image.size {image.size()}')
            # print(f'image {image}')
            # pil_image = to_pil_image(image)
            # pil_image.show()
         
        sample = {"image": image, "fps": fps, "bitrate": bitrate, "resolution": pixel, \
                  "fps_targets": fps_map[fps_targets], "res_targets": res_map[res_targets], \
                  'velocity': velocity, 'framenumber': framenumber}
        
        # # print(f'self.velocity {self.velocity}')
        # if self.framenumber and self.validation:
        #     path = '_'.join(parts[5:-1])
        #     framenumber = parts[4] # 0b0d34e8_166_1080_500_183_bistro_path1_seg1_1_183.png
        #     sample['path'] = path
        #     sample['framenumber'] = framenumber
        #     return sample
        # elif 
        if self.validation:
            path = '_'.join(parts[4:-1]) if not self.framenumber else '_'.join(parts[5:-1])
            # print(f'path {path}')
            sample['path'] = path # path is to compute JOD loss
            # print(f'velocity {velocity}')
            return sample
        else:
            return sample
        