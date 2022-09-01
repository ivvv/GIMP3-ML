import argparse
import os

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import zoom
from skimage.color import rgb2yuv, yuv2rgb

from model import generator


def parse_args():
    parser = argparse.ArgumentParser(description="Colorize images")
    parser.add_argument("-i",
                        "--input",
                        type=str,
                        required=True,
                        help="input image/input dir")
    parser.add_argument("-o",
                        "--output",
                        type=str,
                        required=True,
                        help="output image/output dir")
    parser.add_argument("-m",
                        "--model",
                        type=str,
                        required=True,
                        help="location for model (Generator)")
    parser.add_argument("--device",
                        type=str,
                        default="cuda",
                        help="which device to use? E.g. 'cpu', 'cuda', or 'cuda:0'")
    args = parser.parse_args()
    return args


args = parse_args()

G = generator()

device = torch.device(args.device)
G.load_state_dict(torch.load(args.model, map_location=device))
G.to(device)


def inference(G, in_path, out_path):
    p = Image.open(in_path).convert('RGB')
    img_yuv = rgb2yuv(p).astype(np.float32)
    H, W, _ = img_yuv.shape
    y = img_yuv[None, None, :, :, 0]
    img_variable = torch.from_numpy(y) - 0.5
    img_variable = img_variable.to(device)
    res = G(img_variable)
    uv = res.cpu().detach().numpy()
    uv[:, 0, :, :] *= 0.436
    uv[:, 1, :, :] *= 0.615
    _, _, H1, W1 = uv.shape
    uv = zoom(uv, (1, 1, H / float(H1), W / float(W1)))
    yuv = np.concatenate([y, uv], axis=1)[0].transpose(1, 2, 0)
    rgb = yuv2rgb(yuv)
    rgb = (rgb.clip(0, 1) * 255).astype(np.uint8)
    Image.fromarray(rgb).save(out_path)


if not os.path.isdir(args.input):
    inference(G, args.input, args.output)
else:
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    for f in os.listdir(args.input):
        inference(G, os.path.join(args.input, f), os.path.join(args.output, f))
