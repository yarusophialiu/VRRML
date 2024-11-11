from fastapi import FastAPI
from pydantic import BaseModel
from DecRefClassification import *
import base64
from utils import reverse_fps_map, reverse_res_map
import numpy as np
import sys
sys.path.append('.')
sys.path.append('./lib')
import onnxruntime
import timeit
from onnxconverter_common import float16


app = FastAPI()
SAMPLE_DATA = False
SMALL_MODEL = False

num_framerates, num_resolutions = 10, 5
model = DecRefClassification(num_framerates, num_resolutions, VELOCITY=True) # 68,778,031 parameters， 68,777,999
onnx_model_path_float32 = "onnx_models/vrr_fp32.onnx" # onnx_models vrr_float32_loaded smaller_vrr_fp16
onnx_smaller_model_path_float32 = "onnx_models/smaller_vrr_fp32.onnx" # onnx_models vrr_float32_loaded smaller_vrr_fp16
onnx_model_path = onnx_smaller_model_path_float32 if SMALL_MODEL else onnx_model_path_float32
print(f'onnx_model_path {onnx_model_path}')
ort_session = onnxruntime.InferenceSession(onnx_model_path) # onnx_model_path
_, _, input_h, input_w = ort_session.get_inputs()[0].shape


# Define the data structure for incoming requests
# Pydantic is a Python library used for data validation
class PredictionInput(BaseModel):
    patch: str   # Assume the patch is encode binary
    velocity: float   # Scalar velocity


@app.post("/predict")
async def predict(input_data: PredictionInput):
    patch_data = base64.b64decode(input_data.patch)
    img = np.frombuffer(patch_data, dtype=np.uint8).reshape(128, 128, 3)
    print(f'img {img.shape}')
    img = (np.transpose(np.float32(img[:,:,:,np.newaxis]), (3,2,0,1)) )
    fps = np.array([166], dtype=np.int64)
    bitrate = np.array([1000], dtype=np.int64) 
    resolution = np.array([1080], dtype=np.int64) 
    velocity = np.array([input_data.velocity], dtype=np.float32)
    print(f'fps {fps}')
    print(f'velocity {velocity}')

    t = timeit.default_timer()
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
    return {"predicted_fps": predicted_fps.item(), "predicted_resolution": predicted_resolution.item()}



# run the app:
# uvicorn app:app --reload --host 0.0.0.0 --port 8000