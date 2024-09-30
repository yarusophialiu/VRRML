import time
import torch
from fastapi import FastAPI
# from VideoQualityClassifier_velocity import *
from pydantic import BaseModel
from DecRefClassification import *
from VideoSinglePatchDataset import VideoSinglePatchDataset
import base64
from utils import reverse_fps_map, reverse_res_map


app = FastAPI()
SAMPLE_DATA = False

num_framerates, num_resolutions = 10, 5
model = DecRefClassification(num_framerates, num_resolutions, VELOCITY=True) # 68,778,031 parameters， 68,777,999
model_pth_path = f'{VRRML_Project}/models/patch128-256/patch128_batch128.pth'
model.load_state_dict(torch.load(model_pth_path))  # Load the trained model weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if SAMPLE_DATA:
    data_val_directory = f'{VRRML}/ML/reference128x128/validation'
    val_dataset = VideoSinglePatchDataset(directory=data_val_directory, min_bitrate=500, max_bitrate=2000, patch_size=(128, 128), VELOCITY=True, VALIDATION=True) # len 27592
    sample = val_dataset[0]
    print(f'sample size {sample["image"].size()}')
    patch_tensor = sample['image'].unsqueeze(0)
    patch_tensor = patch_tensor.to(device)

if device.type == 'cuda':
    print(f'Loading model to cuda...')
    to_device(model, device)
model.eval()



# Define the data structure for incoming requests
# Pydantic is a Python library used for data validation
class PredictionInput(BaseModel):
    patch: str   # Assume the patch is encode binary
    velocity: float   # Scalar velocity


@app.post("/predict")
async def predict(input_data: PredictionInput):
    start_time = time.time()

    if SAMPLE_DATA:
        patch_tensor = torch.tensor(input_data.patch, dtype=torch.float32)

    # Decode the base64-encoded patch
    patch_data = base64.b64decode(input_data.patch)
    np_patch = np.frombuffer(patch_data, dtype=np.uint8).reshape(128, 128, 3)
    print(f'np_patch {np_patch.shape}')

    tensor_image_manual = torch.tensor(np_patch, dtype=torch.float32).permute(2, 0, 1) # 3, 360, 640
    tensor_image_manual /= 255.0
    # show_patch(tensor_image_manual.permute(1,2,0)) # after permute 360, 640, 3
    patch_tensor = tensor_image_manual.unsqueeze(0) # torch.Size([1, 3, 128, 128])

    velocity_tensor = torch.tensor(input_data.velocity, dtype=torch.float32).unsqueeze(0)
    fps_tensor = torch.tensor(166, dtype=torch.float32).unsqueeze(0) # torch.Size([1])
    resolution_tensor = torch.tensor(1080, dtype=torch.float32).unsqueeze(0)
    bitrate_tensor = torch.tensor(500, dtype=torch.float32).unsqueeze(0) # TODO
    # print(f'velocity_tensor {velocity_tensor.size()}') # torch.Size([1])

    velocity_tensor = velocity_tensor.to(device)  # or .cuda()
    fps_tensor = fps_tensor.to(device)  # or .cuda()
    resolution_tensor = resolution_tensor.to(device)  # or .cuda()
    bitrate_tensor = bitrate_tensor.to(device)  # or .cuda()
    patch_tensor = patch_tensor.to(device)

    end_time = time.time()
    inference_time = end_time - start_time
    print(f'Preprocess before inference time: {inference_time:.4f} seconds')

    # CUDA event-based timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()

    # Forward pass through the model to get predictions
    with torch.no_grad():  # No gradient computation needed
        res_out, fps_out = model(patch_tensor, fps_tensor, bitrate_tensor, resolution_tensor, velocity_tensor)
        _, fps_preds = torch.max(fps_out, dim=1)
        _, res_preds = torch.max(res_out, dim=1)
        predicted_fps = torch.tensor([reverse_fps_map[int(pred)] for pred in fps_preds])
        predicted_resolution = torch.tensor([reverse_res_map[int(pred)] for pred in res_preds])

    end_event.record()
    torch.cuda.synchronize()  # Wait for all operations to finish
    inference_time = start_event.elapsed_time(end_event) / 1000.0  # Convert milliseconds to seconds
    print(f'Inference time: {inference_time:.4f} seconds')

    print(f'"predicted_fps": {predicted_fps.item()}, "predicted_resolution": {predicted_resolution.item()}')
    # pytorch tensor is not JSON serializable, so use
    # .item() converts a 0-dimensional tensor into a Python scalar
    return {"predicted_fps": predicted_fps.item(), "predicted_resolution": predicted_resolution.item()}
    # return {"predicted_fps": 0, "predicted_resolution": 0}

# uvicorn app:app --reload --host 0.0.0.0 --port 8000