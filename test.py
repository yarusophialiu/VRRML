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
fps_map2 = torch.tensor([1, 2, 3, 4, 5, ])
res_map = {360: 0, 480: 1, 720: 2, 864: 3, 1080: 4}

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
print(f'fps_column {fps_column}')

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
res_arr = [480., 1080.,  864., 360., 1080.,  480.]
fps_arr = [120.,  80., 120.,  50.,  50.,  90.]
velocity_arr = [51918288., 51918288., 51918288., 51918288., 51918288.,51918288.]
bitrate_arr = [1000., 1000., 1000.,  500.,  500.,  500.]
# Mapping dictionary
res_map = {360: 0, 480: 1, 720: 2, 864: 3, 1080: 4}
# Create a list of corresponding indices using the res_map
mapped_indices = torch.tensor([res_map[res.item()] for res in resolutions])
mean_velocity = 341011.652
std_velocity = 3676701.584
# Print the result
# print(f'mapped_indices {mapped_indices}')
res_v = [normalize_max_min(r, 360, 1080) for r in res_arr]
fps_v = [normalize_max_min(r, 30, 120) for r in fps_arr]
bitrate_v = [normalize_max_min(r, 500, 2000) for r in bitrate_arr]
velocity_v = [normalize_z_value(r, mean_velocity, std_velocity) for r in velocity_arr]

resolutions_n = normalize_max_min(resolutions, 360, 1080)
print(f'res_v {res_v}')
print(f'fps_v {fps_v}')
print(f'bitrate_v {bitrate_v}')
print(f'velocity_v {velocity_v}\n')
print(f'resolutions_n {resolutions_n}')

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