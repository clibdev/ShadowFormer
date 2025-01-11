from model import ShadowFormer
import torch
import utils
import numpy as np
import torch.nn.functional as F
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', default='./img/noisy_image.png')
parser.add_argument('--mask_path', default='./img/mask.png')
parser.add_argument('--output_path', default='./result.png')
parser.add_argument('--weights', default='./shadowformer-istd.pt')
parser.add_argument('--device', default='cuda', help='cuda or cpu')
opt = parser.parse_args()

device = torch.device(opt.device)

win_size = 10
img_multiple_of = 8 * win_size

model_restoration = ShadowFormer(img_size=320, embed_dim=32, win_size=win_size, token_projection='linear', token_mlp='leff')

utils.load_checkpoint(model_restoration, opt.weights)

model_restoration.to(device)
model_restoration.eval()

with torch.no_grad():
    rgb_noisy = torch.from_numpy(np.float32(utils.load_img(opt.input_path)))
    rgb_noisy = rgb_noisy.permute(2,0,1)
    rgb_noisy = rgb_noisy.unsqueeze(0).to(device)

    mask = utils.load_mask(opt.mask_path)
    mask = torch.from_numpy(np.float32(mask))
    mask = torch.unsqueeze(mask, dim=0)
    mask = mask.unsqueeze(0).to(device)

    # Pad the input if not_multiple_of win_size * 8
    height, width = rgb_noisy.shape[2], rgb_noisy.shape[3]
    H, W = ((height + img_multiple_of) // img_multiple_of) * img_multiple_of, (
            (width + img_multiple_of) // img_multiple_of) * img_multiple_of
    padh = H - height if height % img_multiple_of != 0 else 0
    padw = W - width if width % img_multiple_of != 0 else 0
    rgb_noisy = F.pad(rgb_noisy, (0, padw, 0, padh), 'reflect')
    mask = F.pad(mask, (0, padw, 0, padh), 'reflect')

    rgb_restored = model_restoration(rgb_noisy, mask)
    rgb_restored = torch.clamp(rgb_restored, 0, 1).cpu().numpy().squeeze().transpose((1, 2, 0))

    # Unpad the output
    rgb_restored = rgb_restored[:height, :width, :]

    utils.save_img(rgb_restored*255.0, opt.output_path)
