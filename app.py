import torch
from fastapi import FastAPI
# from VideoQualityClassifier_velocity import *
from pydantic import BaseModel
from DecRefClassification import *
from typing import List
from VideoSinglePatchDataset import VideoSinglePatchDataset
import struct

# Define your FastAPI app
app = FastAPI()

# # Assume you have your trained neural network model
# class MyNeuralNetwork(torch.nn.Module):
#     def __init__(self):
#         super(MyNeuralNetwork, self).__init__()
#         # Define your layers here
#         # Example: a simple linear model
#         self.fc1 = torch.nn.Linear(64, 32)  # assuming input patch is 64x64
#         self.fc2 = torch.nn.Linear(32, 1)   # output layer, predicting 1 value (e.g., FPS)

#     def forward(self, patch, velocity):
#         x = torch.flatten(patch)  # flatten the patch for fully connected layers
#         x = torch.cat((x, velocity.unsqueeze(0)), dim=0)  # concatenate patch and velocity
#         x = torch.relu(self.fc1(x))
#         output = self.fc2(x)
#         return output

# Load your model (assuming a pre-trained model is available)
# model = MyNeuralNetwork()
num_framerates, num_resolutions = 10, 5
model = DecRefClassification(num_framerates, num_resolutions, VELOCITY=True)
model_pth_path = f'D:/VRRML/VRRML/models/patch128-256/patch128_batch128.pth'
model.load_state_dict(torch.load(model_pth_path))  # Load the trained model weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_val_directory = f'{VRRML}/ML/reference128x128/validation'  
val_dataset = VideoSinglePatchDataset(directory=data_val_directory, min_bitrate=500, max_bitrate=2000, patch_size=(128, 128), VELOCITY=True, VALIDATION=True) # len 27592
sample = val_dataset[0]
print(f'sample size {sample["image"].size()}')
patch_tensor = sample['image'].unsqueeze(0)
patch_tensor = patch_tensor.to(device)
if device.type == 'cuda':
    print(f'Loading model  to cuda...')
    to_device(model, device)
    # to_device(patch_tensor, device)
model.eval()

# print(f'patch_tensor {patch_tensor.size()}')
# if model.is_cuda:
#     print("patch_tensor is on CUDA (GPU).")
# else:
#     print("patch_tensor is on CPU.")

# Define the data structure for incoming requests
# Pydantic is a Python library used for data validation
class PredictionInput(BaseModel):
    # patch: List[List[float]]  # Assume the patch is a 2D array
    velocity: float           # Scalar velocity

fps_map = {30: 0, 40: 1, 50: 2, 60: 3, 70: 4, 80: 5, 90: 6, 100: 7, 110: 8, 120: 9}
res_map = {360: 0, 480: 1, 720: 2, 864: 3, 1080: 4}

reverse_fps_map = {0: 30, 1: 40, 2: 50, 3: 60, 4: 70, 5: 80, 6: 90, 7: 100, 8: 110, 9: 120}
reverse_res_map = {0: 360, 1: 480, 2: 720, 3: 864, 4: 1080}

# Define the route to accept POST requests and return predictions
@app.post("/predict")
async def predict(input_data: PredictionInput):
    # Convert the incoming patch and velocity to PyTorch tensors
    # patch_tensor = torch.tensor(input_data.patch, dtype=torch.float32)

    velocity_tensor = torch.tensor(input_data.velocity, dtype=torch.float32).unsqueeze(0)
    fps_tensor = torch.tensor(166, dtype=torch.float32).unsqueeze(0) # torch.Size([1])
    resolution_tensor = torch.tensor(1080, dtype=torch.float32).unsqueeze(0)
    bitrate_tensor = torch.tensor(500, dtype=torch.float32).unsqueeze(0)

    print(f'velocity_tensor {velocity_tensor.size()}')
    velocity_tensor = velocity_tensor.to(device)  # or .cuda()
    fps_tensor = fps_tensor.to(device)  # or .cuda()
    resolution_tensor = resolution_tensor.to(device)  # or .cuda()
    bitrate_tensor = bitrate_tensor.to(device)  # or .cuda()

    # Forward pass through the model to get predictions
    with torch.no_grad():  # No gradient computation needed
        res_out, fps_out = model(patch_tensor, fps_tensor, bitrate_tensor, resolution_tensor, velocity_tensor)
        _, fps_preds = torch.max(fps_out, dim=1)
        _, res_preds = torch.max(res_out, dim=1)

        predicted_fps = torch.tensor([reverse_fps_map[int(pred)] for pred in fps_preds])
        predicted_resolution = torch.tensor([reverse_res_map[int(pred)] for pred in res_preds])


    print(f'"predicted_fps": {predicted_fps.item()}, "predicted_resolution": {predicted_resolution.item()}')
    # pytorch tensor is not JSON serializable, so use
    # .item() converts a 0-dimensional tensor into a Python scalar
    return {"predicted_fps": predicted_fps.item(), "predicted_resolution": predicted_resolution.item()}

# uvicorn app:app --reload --host 0.0.0.0 --port 8000