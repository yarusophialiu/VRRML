import torch
import torch.nn as nn
import torch.nn.functional as F

# class MyModel(torch.nn.Module):
#     def __init__(self):
#         super(MyModel, self).__init__()
#         self.conv1 = torch.nn.Conv2d(1, 128, 5)

#     def forward(self, x):
#         return torch.relu(self.conv1(x))

# Example model with two inputs: one for CNN, another for MLP
class TwoStageModel(torch.nn.Module):
    def __init__(self):
        super(TwoStageModel, self).__init__()
        # self.cnn = torch.nn.Sequential(...)  # Define CNN layers
        # self.mlp = torch.nn.Sequential(...)  # Define MLP layers
        
        # Define CNN layers
        self.cnn = nn.Sequential(
            # Filter an image for a particular feature (convolution)
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  # assuming 3 channels for RGB input
            # Detect that feature within the filtered image (ReLU)
            nn.ReLU(),
            # Condense the image to enhance the features (maximum pooling)
            nn.MaxPool2d(kernel_size=2, stride=2),                 # downsampling
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))                          # output embedding size 256x1x1
        )

        # Define MLP layers
        self.mlp = nn.Sequential(
            nn.Linear(256 + 4, 128),  # assuming 4 scalar inputs (fps, resolution, bitrate, velocity)
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)          # output layer (adjust size based on your output)
        )

    def forward(self, image_input, mlp_input):
        embedding = self.cnn(image_input)
        # Combine embedding with mlp_input (scalars) before feeding to MLP
        combined_input = torch.cat((embedding, mlp_input), dim=1)
        output = self.mlp(combined_input)
        return embedding, output

# input_tensor = torch.rand((1, 1, 128, 128), dtype=torch.float32)

# model = MyModel()

# print(model.conv1.weight.dtype)  # Should output torch.float32

# torch.onnx.export(
#     model,                  # model to export
#     (input_tensor,),        # inputs of the model,
#     "onnx_demo_model.onnx",        # filename of the ONNX model
#     input_names=["input"],  # Rename inputs for the ONNX model
#     dynamo=True             # True or False to select the exporter to use
# )