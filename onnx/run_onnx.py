import numpy as np
import sys
sys.path.append('.')
sys.path.append('./lib')
import onnxruntime
import timeit
from DecRefClassification import *
import imageio



onnx_model_path = "onnx_models/vrr_float32.onnx" # onnx_models
ort_session = onnxruntime.InferenceSession(onnx_model_path)
# for var in ort_session.get_inputs():
#     print(f'input {var.name}')
# for var in ort_session.get_outputs():
#     print(f'output {var.name}')
_, _, input_h, input_w = ort_session.get_inputs()[0].shape
print(f'input_h, input_w {input_h, input_w}')

t = timeit.default_timer()

img = imageio.imread('0a0c3849_166_1080_500_bistro_path2_seg2_3_496.png')  # 128, 128, 3
img = (np.transpose(np.float32(img[:,:,:,np.newaxis]), (3,2,0,1)) )
print(f'img {img.shape}')
img = np.ascontiguousarray(img)

fps = np.array([166], dtype=np.int64)  # Example FPS input
bitrate = np.array([1000], dtype=np.int64)  # Example Bitrate input
resolution = np.array([1080], dtype=np.int64)  # Example Resolution input
velocity = np.array([7.5], dtype=np.float32)  # Example Velocity input

# ort_inputs = {ort_session.get_inputs()[0].name: img}
ort_inputs = {'images': img, 'fps': fps, 'bitrate': bitrate, 'resolution': resolution, 'velocity': velocity}

ort_outs = ort_session.run(None, ort_inputs)
print('onnxruntime infer time:', timeit.default_timer()-t)

print(ort_outs[0].shape)
print(ort_outs)
