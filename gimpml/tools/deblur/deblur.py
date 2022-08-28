import os, sys, cv2, torch

plugin_loc = os.path.join(os.path.dirname(os.path.realpath(__file__)), "DeblurGANv2")
sys.path.extend([plugin_loc])
from predictorClass import Predictor

from gimpml.plugins.module_utils import *
from gimpml.plugins.deblur.constants import *


def get_deblur(img, cpu_flag=False, weight_path=None):
    if weight_path is None:
        weight_path = get_weight_path()
    predictor = Predictor(
        weights_path=os.path.join(weight_path, "deblur", "best_fpn.h5"), cf=cpu_flag
    )
    if img.shape[2] == 4:  # get rid of alpha channel
        img = img[:, :, 0:3]
    with torch.no_grad():
        pred = predictor(img, None, cf=cpu_flag)
    return pred


@handle_exceptions(PLUGIN_ID)
def main():
    weight_path = get_weight_path()
    data_output = get_model_config(PLUGIN_ID)
    force_cpu = data_output["force_cpu"]
    image = cv2.imread(os.path.join(tmp_path, BASE_IMG))[:, :, ::-1]
    output = get_deblur(image, cpu_flag=force_cpu, weight_path=weight_path)
    cv2.imwrite(os.path.join(tmp_path, RESULT_IMG), output[:, :, ::-1])
    set_model_config({
        "inference_status": "success",
        "force_cpu": force_cpu
    }, PLUGIN_ID)


if __name__ == "__main__":
    main()

