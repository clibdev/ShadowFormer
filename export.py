from model import ShadowFormer
import torch
import utils
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--weights', default='./shadowformer-istd.pt')
parser.add_argument('--device', default='cpu', type=str, help='cuda or cpu')
parser.add_argument('--dynamic', action='store_true', default=False, help='enable dynamic axis in onnx model')
opt = parser.parse_args()

device = torch.device(opt.device)

win_size = 10
img_multiple_of = 8 * win_size

model_restoration = ShadowFormer(img_size=320, embed_dim=32, win_size=win_size, token_projection='linear', token_mlp='leff')

utils.load_checkpoint(model_restoration, opt.weights)

model_restoration.to(device)
model_restoration.eval()

model_path = os.path.splitext(opt.weights)[0] + '.onnx'

dummy_input_1 = torch.randn(1, 3, 480, 640).to(opt.device)
dummy_input_2 = torch.randn(1, 1, 480, 640).to(opt.device)

dynamic_axes = {'input': {2: '?', 3: '?'}, 'mask': {2: '?', 3: '?'}, 'output': {2: '?', 3: '?'}} if opt.dynamic else None

torch.onnx.export(
    model_restoration,
    (dummy_input_1, dummy_input_2),
    model_path,
    verbose=False,
    input_names=['input', 'mask'],
    output_names=['output'],
    dynamic_axes=dynamic_axes,
    opset_version=17
)
