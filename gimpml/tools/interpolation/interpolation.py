import json, os, sys, cv2, torch
from torch.nn import functional as F
import numpy as np

plugin_loc = os.path.join(os.path.dirname(os.path.realpath(__file__)), "RIFE")
sys.path.extend([plugin_loc])
from rife_model import RIFE

from gimpml.plugins.module_utils import *
from gimpml.plugins.interpolation.constants import *


def get_inter(img_s, img_e, string_path, cpu_flag=False, weight_path=None):
    exp = 4
    out_path = string_path

    model = RIFE.Model(cpu_flag)
    model.load_model(os.path.join(weight_path, "interpolateframes"))
    model.eval()
    model.device(cpu_flag)

    img0 = img_s
    img1 = img_e

    img0 = (torch.tensor(img0.transpose(2, 0, 1).copy()) / 255.0).unsqueeze(0)
    img1 = (torch.tensor(img1.transpose(2, 0, 1).copy()) / 255.0).unsqueeze(0)
    if torch.cuda.is_available() and not cpu_flag:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    img0 = img0.to(device)
    img1 = img1.to(device)

    n, c, h, w = img0.shape
    ph = ((h - 1) // 32 + 1) * 32
    pw = ((w - 1) // 32 + 1) * 32
    padding = (0, pw - w, 0, ph - h)
    img0 = F.pad(img0, padding)
    img1 = F.pad(img1, padding)

    img_list = [img0, img1]
    for i in range(exp):
        tmp = []
        for j in range(len(img_list) - 1):
            with torch.no_grad():
                mid = model.inference(img_list[j], img_list[j + 1])
            tmp.append(img_list[j])
            tmp.append(mid)
        tmp.append(img1)
        img_list = tmp

    if not os.path.exists(out_path):
        os.makedirs(out_path)
    for i in range(len(img_list)):
        cv2.imwrite(
            os.path.join(out_path, RESULT_IMG.format(i)),
            (img_list[i][0] * 255)
            .byte()
            .cpu()
            .numpy()
            .transpose(1, 2, 0)[:h, :w, ::-1],
        )


@handle_exceptions(PLUGIN_ID)
def main():
    weight_path = get_weight_path()
    data_output = get_model_config(PLUGIN_ID)
    force_cpu = data_output["force_cpu"]
    gio_file = data_output["gio_file"]
    i1fn = os.path.join(tmp_path, BASE_IMG.format(0))
    image1 = cv2.imread(i1fn)[:, :, ::-1]
    i2fn = os.path.join(tmp_path, BASE_IMG.format(1))
    image2 = cv2.imread(i2fn)[:, :, ::-1]
    get_inter(image1, image2, gio_file, cpu_flag=force_cpu, weight_path=weight_path)
    set_model_config({
        "inference_status": "success",
        "force_cpu": force_cpu
    }, PLUGIN_ID)


if __name__ == "__main__":
    main()

