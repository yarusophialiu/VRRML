# from onnx_demo import *
from DecRefClassification import *



model = DecRefClassification(num_framerates=10, num_resolutions=5, VELOCITY=True)

# Set the model to evaluation mode
model.eval()

dummy_images = torch.randn(1, 4, 128, 128)        # Example image input of size 128x128
dummy_fps = torch.tensor([30])                  # Single FPS value in batch (or size of batch, e.g., [[30], [60]])
dummy_bitrate = torch.tensor([500])             # Single bitrate value in batch
dummy_resolution = torch.tensor([720])          # Single resolution value in batch
dummy_velocity = torch.tensor([1.2])            # Single velocity value in batch

torch.onnx.export(
    model,
    (dummy_images, dummy_fps, dummy_bitrate, dummy_resolution, dummy_velocity),
    "vrr_classification_float32_4channel.onnx",
    input_names=["images", "fps", "bitrate", "resolution", "velocity"],
    output_names=["res_out", "fps_out"],
    dynamic_axes={
        "images": {0: "batch_size"},       # Variable batch size for images
        "res_out": {0: "batch_size"},      # Variable batch size for outputs
        "fps_out": {0: "batch_size"}
    },
    opset_version=11
)



# model = TwoStageModel()

# # Create dummy inputs to match the model's expected inputs
# dummy_image_input = torch.randn(1, 3, 224, 224)  # Adjust for image input shape
# dummy_mlp_input = torch.randn(1, 36)             # 32 embedding + 4 scalars

# # Export with separate input names for image and mlp inputs
# torch.onnx.export(
#     model,
#     (dummy_image_input, dummy_mlp_input),
#     "two_stage_model.onnx",
#     input_names=["image_input", "mlp_input"],     # Separate names for each input
#     output_names=["embedding", "final_output"]    # Output names for each stage
# )
# #  run inference on different batch sizes without having to re-export the model