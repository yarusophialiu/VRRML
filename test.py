from utils import *


# path = r'C:\Users\15142\Projects\VRR\Data\VRRML\train\suntemple_statue_path1_seg1_1\500kbps'
# num_files = count_files_in_folder(path)
# print(f'num_files {num_files}')


path = r'C:\Users\15142\Projects\VRR\Data\VRR_Patches\2024-09-12\suntemple_statue\suntemple_statue_path4_seg2_1\2000kbps'
# path = f'{VRR_Patches}\HPC/2024-09-12/bedroom/bedroom_path4_seg1_2'
path = f'{VRR_Patches}\HPC/2024-09-12/bistro/bistro_path1_seg1_2'
# path =r'C:\Users\15142\Projects\VRR\Data\VRRML\validation\suntemple_statue_path1_seg1_1'

# for bitrate in [500, 1000]:
# # for bitrate in [1000, 1500, 2000,]:
#     p = f'{path}/{bitrate}kbps'
#     print(f'{p}')
#     # p = f'{path}/720x120x{bitrate}'
#     # print(p)
#     print(f'number of files: {bitrate}kbps {count_files_in_folder(p)}')



path_arr = ['bistro_path1_seg3_2',]

# ids = []
# for path_name in path_arr:
#     print(f'\npath_name {path_name}')
#     parts = path_name.split('_')
#     # print(f'split {parts}')

#     path_value = parts[1][4:]  # Extracts the number after 'path'
#     seg_value = parts[2][3:]   # Extracts the number after 'seg'
#     last_value = parts[3]      # The last part is already the number

#     # Convert to integers
#     path_value = int(path_value)
#     seg_value = int(seg_value)
#     speed_value = int(last_value)

#     id = mapPathToId(path_value, seg_value, speed_value)
#     id += 1
#     ids.append(id)
#     print(f'id should be {id}')

#     id -= 1
#     path, seg, speed = mapIdToPath(id)
# print(ids)
# 10,26,29



# images = torch.rand(3, 3, 1, 1)  # Shape: [8000, 3, 64, 64]
# metadata = torch.rand(3, 6)        # Shape: [8000, 6]
# print(f'images \n {images}')
# print(f'metadata \n {metadata}')

# # Generate a random permutation of indices for the first dimension
# perm = torch.randperm(3)
# shuffled_images = images[perm]
# shuffled_metadata = metadata[perm]
# print(f'perm \n {perm}')
# print(f'shuffled_images \n {shuffled_images}')
# print(f'shuffled_metadata \n {shuffled_metadata}')

metadata = torch.tensor([[0.3600, 0.8827, 0.5540, 0.7623, 0.0234, 0.6932],
                        [0.0836, 0.8418, 0.4345, 0.4678, 0.7902, 0.0585],
                        [0.5443, 0.8243, 0.0425, 0.3644, 0.6493, 0.6191]])

fps_map = {30: 0, 40: 1, 50: 2, 60: 3, 70: 4, 80: 5, 90: 6, 100: 7, 110: 8, 120: 9}
res_map = {360: 0, 480: 1, 720: 2, 864: 3, 1080: 4}
fps_map2 = torch.tensor([1, 2, 3, 4, 5, ])


# print(fps_map[torch.tensor([30, 30, 40])])

array = [360, 1080, 720]
res_map = {360: 0, 480: 1, 720: 2, 864: 3, 1080: 4}

# Get corresponding indices from res_map
indices = [res_map[res] for res in array]


metadata = torch.tensor([[110.0, 1080.0, 110.0, 720.0, 500.0, 7.7510],
                         [40.0, 480.0, 110.0, 720.0, 500.0, 7.7510],
                         [80.0, 864.0, 110.0, 720.0, 500.0, 7.6270],
                         [60.0, 360.0, 120.0, 720.0, 2000.0, 7.6680],
                         [100.0, 720.0, 120.0, 720.0, 2000.0, 7.6990],
                         [110.0, 360.0, 120.0, 720.0, 2000.0, 7.6630]])

# Normalization datasets
fps_data = [30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
resolution_data = [360, 480, 720, 864, 1080]

# Normalization function for entire column
def normalize_column(column, data):
    mean = np.mean(data)
    std_dev = np.std(data)
    print(f'mean {mean}, std_dev {std_dev}')
    return (column - mean) / std_dev

# Normalize FPS column
fps_column = metadata[:, 0]

fps_v = [110.,  40.,  80.,  60., 100., 110.]


def normalize(sample, data):
    mean = np.mean(data)
    std_dev = np.std(data)
    # print(f'mean {mean}, std_dev {std_dev}')
    return (sample - mean) / std_dev


# fps_value = torch.tensor([normalize(fps, fps_data) for fps in fps_v])
# print(f'fps_value {fps_value}')


# normalized_fps_column = normalize_column(fps_column, fps_data)

# print(normalized_fps_column)



resolutions = torch.tensor([360, 1080, 720, 480, 864])
res_arr = [480]
fps_arr = [100]
velocity_arr = [0.557]
bitrate_arr = [500.]
# Mapping dictionary
res_map = {360: 0, 480: 1, 720: 2, 864: 3, 1080: 4}
# # Create a list of corresponding indices using the res_map
# mapped_indices = torch.tensor([res_map[res.item()] for res in resolutions])
mean_velocity = 341011.652
std_velocity = 3676701.584
# Print the result
# print(f'mapped_indices {mapped_indices}')
# res_v = [normalize_max_min(r, 360, 1080) for r in res_arr]
# fps_v = [normalize_max_min(r, 30, 120) for r in fps_arr]
# bitrate_v = [normalize_max_min(r, 500, 2000) for r in bitrate_arr]
# velocity_v = [normalize_z_value(r, mean_velocity, std_velocity) for r in velocity_arr]

# # resolutions_n = normalize_max_min(resolutions, 360, 1080)
# print(f'res_v {res_v}')
# print(f'fps_v {fps_v}')
# print(f'bitrate_v {bitrate_v}')
# print(f'velocity_v {velocity_v}\n')
# print(f'resolutions_n {resolutions_n}')

# res_keys = torch.tensor(list(res_map.keys()))  # Tensor of resolution values
# res_values = torch.tensor(list(res_map.values()))  # Tensor of corresponding indices
# print(f'res_keys {res_keys}')
# print(f'res_values {res_values}')

# # Use torch.searchsorted to find the index in res_keys for each resolution
# indices = torch.searchsorted(res_keys, resolutions)
# print(f'indices {indices}')

# # Map the found indices to their corresponding values in res_values
# mapped_indice = res_values[indices]
# print(f'mapped_indices {mapped_indice}')

# num_patches_per_subfolder = 200
# for idx in [0, 1, 2, 3]:
#     start_idx = idx * num_patches_per_subfolder
#     end_idx = start_idx + num_patches_per_subfolder
#     print(f'idx {idx}, start_idx {start_idx}, end_idx {end_idx}')

# # Assume you have a list of numpy arrays
# list_of_np_arrays = [np.random.rand(2, 2, 3) for _ in range(2)]
# print(f'list_of_np_arrays \n {list_of_np_arrays}')

# combined_np_array = np.array(list_of_np_arrays)
# print(f'combined_np_array {combined_np_array.shape} \n {combined_np_array}')


# # Step 2: Convert the numpy array to a PyTorch tensor
# tensor = torch.from_numpy(combined_np_array)
# print(f'tensor {tensor.size()} \n {tensor}')

# images_tensor = torch.stack([torch.from_numpy(np.array(img)) for img in list_of_np_arrays])
# print(f'images_tensor {images_tensor.size()} \n {images_tensor}')
# Assuming all tensors are of shape [3] and on the same device
fps = torch.tensor([0.2220, 0.6670, 0.7780], device='cuda:0', dtype=torch.float64)
resolution = torch.tensor([0.5000, 0.0000, 0.1670], device='cuda:0', dtype=torch.float64)
bitrate = torch.tensor([1.0000, 0.6670, 0.6670], device='cuda:0', dtype=torch.float64)
velocity = torch.tensor([-0.0930, -0.0930, -0.0930], device='cuda:0', dtype=torch.float64)

# fps_resolution_bitrate = torch.stack([fps, resolution, bitrate, velocity], dim=1).float()

# print(f"Stacked Tensor:\n{fps_resolution_bitrate}")

# print(3e-4)
# print(256 * 32 * 32)
# Function to find key by value
def find_key_by_value(dictionary, target_value):
    # Loop through dictionary items
    for key, value in dictionary.items():
        if value == target_value:
            return key
    return None  # Return None if the value is not found

from JOD import *
fps_targets = torch.tensor([8, 5, 5, 9, 8, 5, 7, 5, 5, 8, 5, 6, 8, 9, 8, 3, 9, 6, 6, 6, 8, 6, 6, 5,])
res_targets = torch.tensor([1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1,])
fps_preds =   torch.tensor([8, 8, 8, 8, 8, 9, 8, 9, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 8, 8])
res_preds =   torch.tensor([1, 1, 2, 1, 3, 1, 3 ,0, 2, 0 ,0, 0, 3, 0, 2, 1, 1, 2, 1, 3, 1, 3 ,0, 2,])
bitrate =  torch.tensor([0.0000, 0.333, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,])
paths = ['gallery_path4_seg3_2', 'bistro_path3_seg2_3', 'bistro_path4_seg3_3', 'bistro_path2_seg3_3', 'gallery_path5_seg3_2', 'bistro_path2_seg3_3', 'crytek_sponza_path5_seg2_2', 'crytek_sponza_path1_seg2_3', 'bedroom_path2_seg2_3', 'bedroom_path5_seg3_3', 'bistro_path4_seg3_3', 'bistro_path3_seg2_3', 'bistro_path2_seg3_3', 'bistro_path3_seg3_2', 'bistro_path2_seg2_3', 'bistro_path3_seg3_2', 'crytek_sponza_path2_seg3_3', 'gallery_path2_seg2_3', 'bistro_path3_seg2_3', 'bistro_path4_seg3_3', 'crytek_sponza_path5_seg3_3', 'bedroom_path5_seg3_3', 'bistro_path4_seg2_2', 'crytek_sponza_path5_seg2_3', 'bistro_path2_seg3_3', 'gallery_path4_seg3_3', 'bistro_path2_seg3_3', 'crytek_sponza_path5_seg3_3', 'crytek_sponza_path5_seg3_2', 'bistro_path2_seg2_3', 'bistro_path1_seg2_2', 'crytek_sponza_path1_seg1_3', 'bedroom_path5_seg3_3', 'bedroom_path2_seg3_3', 'bistro_path4_seg1_2', 'bistro_path2_seg3_3', 'bistro_path4_seg3_3', 'gallery_path5_seg3_2', 'crytek_sponza_path5_seg2_2', 'crytek_sponza_path1_seg1_2', 'gallery_path4_seg3_3', 'bistro_path4_seg2_2', 'bedroom_path2_seg3_3', 'gallery_path3_seg1_1', 'crytek_sponza_path2_seg1_3', 'bedroom_path5_seg2_3', 'crytek_sponza_path5_seg3_3', 'bistro_path1_seg2_3', 'bedroom_path3_seg3_2', 'gallery_path3_seg1_1', 'bistro_path2_seg3_3', 'bedroom_path5_seg2_3', 'crytek_sponza_path5_seg3_2', 'bistro_path3_seg3_2', 'crytek_sponza_path5_seg3_3', 'bedroom_path5_seg3_3', 'gallery_path2_seg3_3', 'bedroom_path4_seg1_2', 'bedroom_path5_seg2_3', 'bistro_path2_seg3_3']
bitrate_map = {500: 0.0, 1000: 0.333, 1500: 0.667, 2000: 1.0}


def compute_JOD_loss(paths, bitrate, fps_preds, res_preds, fps_targets, res_targets):
    # for i in range(fps_targets.size()[0]):
    for i in range(2):
        path = paths[i]
        base_name = path.split('_path')[0]  # e.g., "crytek_sponza", "bistro", "gallery"
        path_name = '_'.join(path.split('_')[1:])
        variable_name = f"{base_name}_jod"
        if variable_name in globals():
            corresponding_value = globals()[variable_name]
            fps_preds_val = find_key_by_value(fps_map, fps_preds[i].item())
            res_preds_val = find_key_by_value(res_map, res_preds[i].item())
            fps_targets_val = find_key_by_value(fps_map, fps_targets[i].item())
            res_targets_val = find_key_by_value(res_map, res_targets[i].item())
            bitrate_val = find_key_by_value(bitrate_map, bitrate[i].item())
            print(f'bitrate[i] {bitrate[i]}, bitrate_val {bitrate_val}')
            # print(f"{variable_name} fps, res preds {fps_preds_val, res_preds_val}")
            # print(f"fps, res targets {fps_targets_val, res_targets_val}")
            # print(f"path_name {path_name, res_targets_val}")
            pred = get_jod_score(corresponding_value, path_name, bitrate_val, fps_preds_val, str(res_preds_val))
            truth = get_jod_score(corresponding_value, path_name, bitrate_val, fps_targets_val, str(res_targets_val))
            print(f'pred {pred}, truth {truth}')
        else:
            print(f"Variable {variable_name} does not exist")
# # for i in range(fps_targets.size()[0]):
# for i in range(2):
#     path = paths[i]
#     base_name = path.split('_path')[0]  # e.g., "crytek_sponza", "bistro", "gallery"
#     path_name = '_'.join(path.split('_')[1:])
#     variable_name = f"{base_name}_jod"
#     if variable_name in globals():
#         corresponding_value = globals()[variable_name]
#         fps_preds_val = find_key_by_value(fps_map, fps_preds[i].item())
#         res_preds_val = find_key_by_value(res_map, res_preds[i].item())

#         fps_targets_val = find_key_by_value(fps_map, fps_targets[i].item())
#         res_targets_val = find_key_by_value(res_map, res_targets[i].item())
#         bitrate_val = find_key_by_value(bitrate_map, round(bitrate[i].item(), 3))
#         print(f' bitrate[i] {bitrate[i]}, bitrate_val {bitrate_val}')

#         # print(f"{variable_name} fps, res preds {fps_preds_val, res_preds_val}")
#         # print(f"fps, res targets {fps_targets_val, res_targets_val}")
#         # print(f"path_name {path_name, res_targets_val}")
#         pred = get_jod_score(corresponding_value, path_name, bitrate_val, fps_preds_val, str(res_preds_val))
#         truth = get_jod_score(corresponding_value, path_name, bitrate_val, fps_targets_val, str(res_targets_val))
#         print(f'pred {pred}, truth {truth}\n')
#     else:
#         print(f"Variable {variable_name} does not exist")



#     # v1 = get_jod_score(crytek_sponza_jod, 'path1_seg2_1', 1500, 50, '720')
#     # print(f'i ')
# pred = get_jod_score(gallery_jod, 'path4_seg3_2', 1500, 110, '480')
# print(f'pred {pred}')
# def normalize(sample, min_vals, max_vals):
#         # print(f'val, min_vals, max_vals {sample, min_vals, max_vals}')
#         sample = (sample - min_vals) / (max_vals - min_vals)
#         return round(sample, 3)

# def denormalize(normalized_sample, min_vals, max_vals):
#     original_sample = normalized_sample * (max_vals - min_vals) + min_vals
#     return round(original_sample, 3)

# for i in [500, 1000, 1500, 2000]:
#     bitrate_k = normalize(i, 500, 2000)
#     bitrate_v = denormalize(bitrate_k, 500, 2000)
#     print(f'bitrate {bitrate_k}, bitrate_v {bitrate_v}')

string1 = 'gallery_path5_seg3_2'
string2 = 'crytek_sponza_path5_seg3_2'

# Function to extract 'pathX_segY_Z' part
def extract_path_segment(s):
    return '_'.join(s.split('_')[1:])  # Split by '_' and join from the second element onward

# # Usage
# result1 = extract_path_segment(string1)
# result2 = extract_path_segment(string2)

# print(result1)  # Output: path5_seg3_2
# print(result2)  # Output: path5_seg3_2
# fps = torch.tensor([[30, 40, 50, 60, 70, 80, 90, 100, 110, 120]])
# # resolution = [360, 480, 720, 864, 1080]

# fps_idx = torch.argmax(fps, dim=1)
# print(f'fps_idx \n {fps_idx}')

import torch.nn.functional as F


fps_out = torch.tensor([[-2.0147e+01, -1.6176e+01, -7.1966e+00, -5.2333e+00, -3.7137e+00,
          3.0580e+00,  4.8994e+00, -2.7443e+00,  9.8523e-01, -1.4321e+00],
        [-3.4208e+01, -2.8417e+01, -1.5378e+01, -1.0567e+01, -5.9481e+00,
          3.3894e+00,  7.1927e+00, -3.6140e-01,  5.6558e+00,  1.7339e+00],
        [-2.2631e+01, -1.8321e+01, -9.9125e+00, -6.8875e+00, -3.9963e+00,
          3.0752e+00,  4.9592e+00,  3.4556e-01,  2.5968e+00, -3.0995e-02],
        [ 2.7738e+01,  2.3132e+01,  1.9887e+01,  1.1482e+01,  6.9956e+00,
          2.4360e+00,  3.9531e+00, -1.1206e+01, -1.0715e+01, -2.7066e+01],
        [-9.7161e+00, -8.6629e+00, -5.3393e+00, -3.7444e+00, -8.2156e-01,
          1.3270e+00,  1.7608e+00,  8.2745e-01,  2.3480e+00,  3.2705e+00],
        [ 2.3105e+01,  2.2503e+01,  2.0458e+01,  1.4268e+01,  1.0512e+01,
          3.7302e+00, -1.0985e-01, -8.3386e+00, -1.4244e+01, -2.3658e+01],
        [-5.1612e+01, -4.2653e+01, -2.8462e+01, -1.6689e+01, -1.2625e+01,
          8.4313e-01,  5.8361e+00,  5.5016e+00,  8.0943e+00,  1.7422e+01],
        [ 1.8354e+01,  1.6819e+01,  1.7489e+01,  1.4089e+01,  1.2344e+01,
          6.7269e+00,  1.3591e+00, -9.9619e+00, -9.5607e+00, -2.3956e+01]],
       )
_, fps_preds = torch.max(fps_out, dim=1)

# probabilities = F.softmax(fps_out, dim=1)
# _, probabilities = torch.max(probabilities, dim=1)
print(f'fps_preds {fps_preds}')
# print(f'probabilities {probabilities}')

def normalize(sample, min_vals, max_vals):
        # print(f'val, min_vals, max_vals {sample, min_vals, max_vals}')
        sample = (sample - min_vals) / (max_vals - min_vals)
        return round(sample, 3)

# for fps in [30, 120]:
#     print(normalize(fps, 30, 120))
fps_targets = torch.tensor([0, 0, 6, 9, 6, 9, 9, 2])
print(f'fps_targets {fps_targets}')

reverse_fps_map = {v: k for k, v in fps_map.items()}
print(f'reverse_fps_map {reverse_fps_map}')

# Convert the predicted and target indices to actual FPS values
# predicted_fps = torch.tensor([reverse_fps_map[int(pred)] for pred in fps_preds])
# target_fps = torch.tensor([reverse_fps_map[int(target)] for target in fps_targets])
# print(f'predicted_fps {predicted_fps}')
# print(f'target_fps {target_fps}')
# # Compute the absolute difference between predictions and targets
# absolute_errors = torch.abs(predicted_fps - target_fps)
# print(f'absolute_errors {absolute_errors}')

# # Compute the mean absolute error (expected error)
# expected_error = torch.mean(absolute_errors.float())

# # print(f"Expected Error (Mean Absolute Error): {expected_error.item()} FPS")
# predicted_fps = torch.tensor([ 90,  40,  30,  30,  30, 120, 120,  30])
# target_fps = torch.tensor([ 90,  50,  50,  50,  50, 120, 120,  50])
# percentage_errors_fps = torch.tensor([ 0., 20., 40., 40., 40.,  0.,  0., 40.])
# percentage_errors_fps = torch.abs((predicted_fps - target_fps) / target_fps)
# print(f"percentage_errors_fps: {percentage_errors_fps}\n") # 0.08333
# mape_fps = torch.mean(percentage_errors_fps.float())
# print(f"Mean Absolute Percentage Error (MAPE) of FPS: {mape_fps.item()}%")

# # percentage_errors_fps: tensor([ 0., 20., 40., 40., 40.,  0.,  0., 40.])
# # Mean Absolute Percentage Error (MAPE) of FPS: 22.5%
from DecRefClassification import *

num_framerates, num_resolutions = 10, 5
model = DecRefClassification(num_framerates, num_resolutions, VELOCITY=True)
# model_pth_path = f'{VRRML_Project}/models/patch128-256/patch128_batch128.pth'
model_pth_path = f'{VRRML_Project}/models/test_no_param/p128_b128_nofps_nores.pth'
model.load_state_dict(torch.load(model_pth_path))  # Load the trained model weights

count_model_parameters(model)