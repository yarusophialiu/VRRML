# -*- coding: utf-8 -*-
import torch
from collections import OrderedDict

import sys
sys.path.append('.')
sys.path.append('./lib')

import argparse
# from DecRefClassification import *
from DecRefClassification_smaller import *

parser = argparse.ArgumentParser("ONNX converter")
parser.add_argument('--src_model_path', type=str, default=None, help='src model path')
parser.add_argument('--dst_model_path', type=str, default=None, help='dst model path')
parser.add_argument('--img_size', type=int, default=None, help='img size')
parser.add_argument('--checkpoint', type=int, default=None, help='pth be checkpoint')  # 0 is false
args = parser.parse_args()
    
#device = torch.device('cuda')
model_path = args.src_model_path
onnx_model_path = args.dst_model_path
img_size = args.img_size
checkpoint = args.checkpoint

model = DecRefClassification(num_framerates=10, num_resolutions=5, VELOCITY=True)#.cuda()
new_state_dict = OrderedDict()
if checkpoint:
    print(f'the pth file is checkpoint')
    checkpoint = torch.load(model_path)
    state_dict = checkpoint['model_state_dict']  # Load the trained model weights
    print(f'state_dict {state_dict}')

else:
    state_dict = torch.load(model_path)
    # print(f'state_dict {state_dict}')

# state_dict = torch.load(model_path)['params_ema']
# new_state_dict = OrderedDict()
for k, v in state_dict.items():
    # # stylegan_decoderdotto_rgbsdot1dotmodulated_convdotbias
    # if "stylegan_decoder" in k:
    #     k = k.replace('.', 'dot')
    #     new_state_dict[k] = v
    #     k = k.replace('dotweight', '.weight')
    #     k = k.replace('dotbias', '.bias')
    #     new_state_dict[k] = v
    # else:
    new_state_dict[k] = v
     
model.load_state_dict(new_state_dict, strict=False)
model.eval()

dummy_images = torch.randn(1, 3, 128, 128)        # Example image input of size 128x128
dummy_fps = torch.tensor([30])                  # Single FPS value in batch (or size of batch, e.g., [[30], [60]])
dummy_bitrate = torch.tensor([500])             # Single bitrate value in batch
dummy_resolution = torch.tensor([720])          # Single resolution value in batch
dummy_velocity = torch.tensor([1.2])            # Single velocity value in batch

torch.onnx.export(model, 
                 (dummy_images, dummy_fps, dummy_bitrate, dummy_resolution, dummy_velocity), 
                 onnx_model_path,
                 # whether to store the trained parameters of the model in the exported ONNX model
                 export_params=True, opset_version=11, 
                 do_constant_folding=True,
                #  input_names = ['input'], output_names = []
                input_names=["images", "fps", "bitrate", "resolution", "velocity"],
                output_names=["res_out", "fps_out"],
                dynamic_axes={
                    "images":  {0: "batch_size"},       # Variable batch size for images
                    "res_out": {0: "batch_size"},      # Variable batch size for outputs
                    "fps_out": {0: "batch_size"}},
                )

# python torch2onnx.py --src_model_path models/patch128-256/patch128_batch128.pth --dst_model_path onnx_models/vrr_float32.onnx --img_size 128 --checkpoint 0
# python torch2onnx.py --src_model_path models/smaller_vrr.pth --dst_model_path onnx_models/vrr_ckpt_115.onnx --img_size 128 --checkpoint 0


# # wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth

# # python torch2onnx.py  --src_model_path ./GFPGANv1.4.pth --dst_model_path ./GFPGANv1.4.onnx --img_size 512 

# # python torch2onnx.py  --src_model_path ./GFPGANCleanv1-NoCE-C2.pth --dst_model_path ./GFPGANv1.2.onnx --img_size 512 

# # "models/patch128-256/patch128_batch128.pth"