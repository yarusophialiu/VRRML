import torch
from fastapi import FastAPI
# from VideoQualityClassifier_velocity import *
from pydantic import BaseModel
from DecRefClassification import *
from typing import List
from VideoSinglePatchDataset import VideoSinglePatchDataset

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
model_pth_path = f'models/patch128_batch128.pth'
model.load_state_dict(torch.load(model_pth_path))  # Load the trained model weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_val_directory = f'{VRRML}/ML/reference128x128/validation'  
val_dataset = VideoSinglePatchDataset(directory=data_val_directory, min_bitrate=500, max_bitrate=2000, patch_size=(128, 128), VELOCITY=True, VALIDATION=True) # len 27592
sample = val_dataset[0]
print(f'sample size {sample["image"].size()}')
if device.type == 'cuda':
    print(f'Loading model  to cuda...')
    to_device(model, device)
    to_device([sample["image"]], device)
model.eval()
# with torch.no_grad():  # Ensure gradients are not computed


# Define the data structure for incoming requests
# Pydantic is a Python library used for data validation
class PredictionInput(BaseModel):
    # patch: List[List[float]]  # Assume the patch is a 2D array
    velocity: float           # Scalar velocity

# Define the route to accept POST requests and return predictions
@app.post("/predict")
async def predict(input_data: PredictionInput):
    # Convert the incoming patch and velocity to PyTorch tensors
    # patch_tensor = torch.tensor(input_data.patch, dtype=torch.float32)

    velocity_tensor = torch.tensor(input_data.velocity, dtype=torch.float32)
    fps_tensor = torch.tensor(166, dtype=torch.float32)
    resolution_tensor = torch.tensor(1080, dtype=torch.float32)
    bitrate_tensor = torch.tensor(8000, dtype=torch.float32)

    # Forward pass through the model to get predictions
    with torch.no_grad():  # No gradient computation needed
        prediction = model(patch_tensor, fps_tensor, bitrate_tensor, resolution_tensor, velocity_tensor)

    # Return the prediction as JSON response
    print(f'prediction {prediction.item()}')
    return {"predicted_fps": prediction.item()}

