import numpy as np
import csv
import os
import math
import torch
import sys
sys.path.append('.')
sys.path.append('./lib')
import onnxruntime
import timeit
import imageio
import torch.nn.functional as F
from datetime import datetime
from utils import get_velocities_from_patch_data

def softmax(logits):
    # Find the maximum logit for numerical stability
    max_logit = max(logits)
    # Compute the exponentials and their sum
    exp_values = [math.exp(logit - max_logit) for logit in logits]
    sum_exp = sum(exp_values)
    probabilities = [exp_val / sum_exp for exp_val in exp_values]
    return probabilities

def process_nnoutput(res_out, fps_out):
    max_res = max(res_out[0])
    max_fps = max(fps_out[0])
    # print(f'max res fps {max_res, max_fps}')
    res_out /= max_res
    fps_out /= max_fps
    # print(f'res fps out {res_out, fps_out}')
    res_tensor = torch.from_numpy(res_out[0])
    fps_tensor = torch.from_numpy(fps_out[0])
    res_probabilities = F.softmax(res_tensor, dim=0)
    fps_probabilities =  F.softmax(fps_tensor, dim=0)
    
    # print(f'res probabilities {res_probabilities}')
    # print(f'fps probabilities {fps_probabilities}')
    return res_probabilities, fps_probabilities


def shouldChangeSettings(res_probabilities, fps_probabilities):
    # print(f'current_resolution, current_fps {current_resolution, current_fps}')
    p_res = np.zeros(5) # [0,0,0,0,0]
    p_fps = np.zeros(10) # [0,0,0,0,0,0,0,0,0,0]
    for i in range(len(res_probabilities)):
        resolution = reverse_res_map[i]
        if (resolution == current_resolution):
            p_res[i] = resolutionBias + res_probabilities[i]
        else:
            p_res[i] = res_probabilities[i]
        
    for i in range(len(fps_probabilities)):
        framerate = reverse_fps_map[i]
        if (framerate == current_fps):
            p_fps[i] = fpsBias + fps_probabilities[i]
        else:
            p_fps[i] = fps_probabilities[i]    
    # print(f'p_res {p_res}')
    max_res_index = np.argmax(p_res)
    max_fps_index = np.argmax(p_fps)
    return reverse_res_map[max_res_index], reverse_fps_map[max_fps_index] 


def print_results(predicted_resolution, predicted_fps):
    if predicted_resolution != 1080:
        print(f'\nid {id}, velocity is {velocity}')
        print(f'res_preds {res_preds}, predicted_resolution {predicted_resolution} p')
    if predicted_fps != 120:
        print(f'\nid {id}, velocity is {velocity}')
        print(f'fps_preds {fps_preds}, predicted_fps {predicted_fps} fps')

# test_data = "C:/Users/yl962/Downloads/test_data"

# now = datetime.now()
# current_time = datetime.now().strftime("%m%d_%H%M")
# print(current_time)
EXPORT_CSV = True # False 

PATCH_DATA_DIR = 'sibenik-patch-data'
PATCH_DATA_FULL_DIR = f"D:/VRRML/VRRML/{PATCH_DATA_DIR}"
# scene_folder = 'sibenik_0_500_2' # 'salledebain_0_500_2'
# velocity_folder = F'{PATCH_DATA_DIR}/{scene_folder}/velocities.csv'
resolutionBias = 0
fpsBias = 0
model_dir = "2025-01-05/22_47"
# smaller_vrr_fp32 vrr_fp32
onnx_model_path_float32 = "onnx_models/smaller_vrr_fp32.onnx" # onnx_models vrr_fp32 smaller_vrr_fp32
# onnx_model_path_float32 = f"{model_dir}/vrr_fp32.onnx" # onnx_models vrr_fp32 smaller_vrr_fp32
print(f'onnx_model {onnx_model_path_float32}')
INFERENCE = True # True False
reverse_fps_map = {0: 30, 1: 40, 2: 50, 3: 60, 4: 70, 5: 80, 6: 90, 7: 100, 8: 110, 9: 120}
reverse_res_map = {0: 360, 1: 480, 2: 720, 3: 864, 4: 1080}
# velocities = [0.000534, 0.000648, 0.000162, 0.000244, 0.000277, 9e-05, 0.000139, 0.000156, 0.000279, 0.000148, 0.000487, 0.000476, 0.000195, 0.000298, 0.00028, 0.000323, 0.000855, 0.000836, 0.00037, 0.001018, 0.001167, 0.00036, 0.000335, 0.000477, 0.000797, 0.00107, 0.000572, 0.000974, 0.000584, 0.000786, 0.001008, 0.000694, 0.00128, 0.001319, 0.000943, 0.001152, 0.001008, 0.001364, 0.000444, 0.001514, 0.00159, 0.000554, 0.001326, 0.000548, 0.00111, 0.001761, 0.001232, 0.001036, 0.000561, 0.001805, 0.000566, 0.000879, 0.001383, 0.001501, 0.001863, 0.0017, 0.00118, 0.001142, 0.000929, 0.001889, 0.000689, 0.001177, 0.001263, 0.000655, 0.001011, 0.001446, 0.001316, 0.002239, 0.000731, 0.001847, 0.002114, 0.001437, 0.001438, 0.000737, 0.000958, 0.001014, 0.002079, 0.000889, 0.000735, 0.001366, 0.002279, 0.001941, 0.002275, 0.001868, 0.001839, 0.002485, 0.001492, 0.001137, 0.001905, 0.001266, 0.003055, 0.001129, 0.001168, 0.001002, 0.001657, 0.001764, 0.001566, 0.001832, 0.000841, 0.001998, 0.002029, 0.002215, 0.001435, 0.00122, 0.000544, 0.001707, 0.002546, 0.000757, 0.00227, 0.001919, 0.002475, 0.001434, 0.001805, 0.001718, 0.001617, 0.002185, 0.000815, 0.001897, 0.000507, 0.001109, 0.00239, 0.0019, 0.000401, 0.001942, 0.001627, 0.002556, 0.002315, 0.001846, 0.000537, 0.00104, 0.002151, 0.001109, 0.002285, 0.009764, 0.001326, 0.001109, 0.001815, 0.002025, 0.005082, 0.001829, 0.001358, 0.001111, 0.001532, 0.002053, 0.002552, 0.000589, 0.000526, 0.002325, 0.001166, 0.001547, 0.008597, 0.00531, 0.001609, 0.00169, 0.001183, 0.001201, 0.000401, 0.001518, 0.000772, 0.00066, 0.010806, 0.002646, 0.000996, 0.0088, 0.000498, 0.001793, 0.001769, 0.000658, 0.00631, 0.0011, 0.009239, 0.00067, 0.001007, 0.007131, 0.001648, 0.001292, 0.002463, 0.000392, 0.000963, 0.001151, 0.002155, 0.002645, 0.00166, 0.001968, 0.004255, 0.000967, 0.001748, 0.007592, 0.005495, 0.00186, 0.007102, 0.007196, 0.001584, 0.000797, 0.002234, 0.002494, 0.000779, 0.003653, 0.001484, 0.001489, 0.002751]
path = onnx_model_path_float32

ort_session = onnxruntime.InferenceSession(path, providers=['CPUExecutionProvider']) # onnx_model_path
_, _, input_h, input_w = ort_session.get_inputs()[0].shape
# print(f'input_h, input_w {input_h, input_w}')
for root, subfolders, files in os.walk(PATCH_DATA_FULL_DIR):
    for scene_folder in subfolders:
# for scene_folder in ['sibenik_0_2000_2']:
        data = []
        print(f"Subfolder: {scene_folder}")
        velocity_folder = F'{PATCH_DATA_DIR}/{scene_folder}/velocities.csv'
        velocities = get_velocities_from_patch_data(velocity_folder)
        print(f'velocities {velocities[:10]}')
        splitted = scene_folder.split('_')
        scene, path_idx, bitrate_from_scene = splitted[0], splitted[1], int(splitted[2])
        print(f'=== {scene, path_idx, int(splitted[2])} ===')
        csv_file = ''
        if EXPORT_CSV:
            csv_file = f"csv/{scene}_{path_idx}_{int(splitted[2])}kbps.csv"

        length = len(velocities)
        print(f'total number of frames {length}')
        fps = np.array([166], dtype=np.int64)  # Example FPS input
        bitrate = np.array([int(splitted[2])], dtype=np.int64)  # Example Bitrate input
        resolution = np.array([1080], dtype=np.int64)  # Example Resolution input
        current_resolution = 1080
        current_fps = 60
        if INFERENCE:
            for id in range(2, length):
                img = imageio.v2.imread(f'{PATCH_DATA_DIR}/{scene_folder}/{id}.bmp')  # 128, 128, 3
                velocity = velocities[id-2] * 166 / current_fps 
                # print(f'\nid {id}, velocity is {velocity}')
                # img = imageio.v2.imread('C:/Users/15142/Projects/VRR/Data/VRRML/ML/reference128x128/test/360x120/0a0c3849_166_1080_500_bistro_path2_seg2_3_496.png')  # 128, 128, 3

                img = (np.transpose(np.float32(img[:,:,:,np.newaxis]), (3,2,0,1)) ) # float8
                velocity = np.array([velocity], dtype=np.float32)  # Example Velocity input
                img = np.ascontiguousarray(img)
                # t = timeit.default_timer()
                ort_inputs = {'images': img, 'fps': fps, 'bitrate': bitrate, 'resolution': resolution, 'velocity': velocity}
                # ort_inputs = {'images': img, 'bitrate': bitrate, 'velocity': velocity}

                res_out, fps_out = ort_session.run(None, ort_inputs)
                # print('onnxruntime infer time:', timeit.default_timer()-t)

                res_probabilities, fps_probabilities = process_nnoutput(res_out, fps_out)
                current_resolution, current_fps = shouldChangeSettings(res_probabilities, fps_probabilities) # selected_res, selected_fps = shouldChangeSettings

                # print_results(predicted_resolution, predicted_fps)
                # print(f'after change current_resolution {current_resolution}p, current_fps {current_fps}\n')

                if not EXPORT_CSV:
                    res_preds = np.argmax(res_out, axis=1)
                    fps_preds = np.argmax(fps_out, axis=1)
                    predicted_resolution = reverse_res_map[res_preds[0]]
                    predicted_fps = reverse_fps_map[fps_preds[0]]
                    print(f'predicted_resolution {predicted_resolution} p, predicted_fps {predicted_fps} fps')
                row = {"frame_count": id, "velocity": velocities[id-2], "placeholder": "", "fps": current_fps, "resolution": current_resolution}
                data.append(row)

            if EXPORT_CSV:
                with open(csv_file, mode='w', newline='') as file:
                    writer = csv.DictWriter(file, fieldnames=["frame_count", "velocity", "placeholder", "fps", "resolution"])        
                    # writer.writeheader()
                    for row in data:
                        writer.writerow(row)

                print(f"Data written to {csv_file}")    


                    