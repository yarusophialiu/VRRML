import numpy as np
import onnx
import sys
sys.path.append('.')
sys.path.append('./lib')
import onnxruntime
import timeit
from DecRefClassification import *
from onnxconverter_common import float16
import imageio

onnx_model_path = "onnx_models/smaller_vrr.onnx" # onnx_models vrr_float32_loaded
CONVERT_TO_FLOAT16_MODEL = False # False True   convert model to float16 
if CONVERT_TO_FLOAT16_MODEL:
    model = onnx.load(onnx_model_path)
    model_fp16 = float16.convert_float_to_float16(model)
    onnx.save(model_fp16, "onnx_models/smaller_vrr_fp16.onnx")


onnx_model_path_float16 = "onnx_models/smaller_vrr_fp16.onnx" # onnx_models vrr_float32_loaded
INFERENCE = True
FLOAT16 = False # True False
reverse_fps_map = {0: 30, 1: 40, 2: 50, 3: 60, 4: 70, 5: 80, 6: 90, 7: 100, 8: 110, 9: 120}
reverse_res_map = {0: 360, 1: 480, 2: 720, 3: 864, 4: 1080}
print(f'============================= FLOAT16 {FLOAT16} =============================')
if INFERENCE:
    path = onnx_model_path_float16 if FLOAT16 else onnx_model_path
    ort_session = onnxruntime.InferenceSession(path) # onnx_model_path
    # for var in ort_session.get_inputs():
    #     print(f'input {var.name}')
    # for var in ort_session.get_outputs():
    #     print(f'output {var.name}')
    _, _, input_h, input_w = ort_session.get_inputs()[0].shape
    print(f'input_h, input_w {input_h, input_w}')
    t = timeit.default_timer()
    img = imageio.v2.imread('0a0c3849_166_1080_500_bistro_path2_seg2_3_496.png')  # 128, 128, 3
    if FLOAT16:
        img = (np.transpose(np.float16(img[:,:,:,np.newaxis]), (3,2,0,1)) )
        fps = np.array([166], dtype=np.int64)  # Example FPS input
        bitrate = np.array([1000], dtype=np.int64)  # Example Bitrate input
        resolution = np.array([1080], dtype=np.int64)  # Example Resolution input
        velocity = np.array([0.5], dtype=np.float16)  # Example Velocity input
    else:
        img = (np.transpose(np.float32(img[:,:,:,np.newaxis]), (3,2,0,1)) )
        fps = np.array([166], dtype=np.int64)  # Example FPS input
        bitrate = np.array([1000], dtype=np.int64)  # Example Bitrate input
        resolution = np.array([1080], dtype=np.int64)  # Example Resolution input
        velocity = np.array([0.5], dtype=np.float32)  # Example Velocity input
    img = np.ascontiguousarray(img)
    
    # t = timeit.default_timer()
    # ort_inputs = {ort_session.get_inputs()[0].name: img}
    ort_inputs = {'images': img, 'fps': fps, 'bitrate': bitrate, 'resolution': resolution, 'velocity': velocity}

    res_out, fps_out = ort_session.run(None, ort_inputs)
    print('onnxruntime infer time:', timeit.default_timer()-t)

    # print(f'res_out {res_out}')
    # print(f'fps_out {fps_out}')
    res_preds = np.argmax(res_out, axis=1)
    fps_preds = np.argmax(fps_out, axis=1)
    predicted_resolution = reverse_res_map[res_preds[0]]
    predicted_fps = reverse_fps_map[fps_preds[0]]
    print(f'res_preds {res_preds}, predicted_resolution {predicted_resolution} p')
    print(f'fps_preds {fps_preds}, predicted_fps {predicted_fps} fps')
