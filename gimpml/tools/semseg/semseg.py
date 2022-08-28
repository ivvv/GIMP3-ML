import json, os, sys, torch, cv2

plugin_loc = os.path.join(os.path.dirname(os.path.realpath(__file__)), "DPT")
sys.path.extend([plugin_loc])
from semseg_run import run

from gimpml.plugins.module_utils import *
from gimpml.plugins.semseg.constants import *


def get_seg(input_image, cpu_flag=False, weight_path=None):
    with torch.no_grad():
        out = run(
            input_image[:, :, ::-1],
            os.path.join(weight_path, "semseg", "dpt_hybrid-ade20k-53898607.pt"),
            cpu_flag=cpu_flag,
        )
    return out[:, :, ::-1]


@handle_exceptions(PLUGIN_ID)
def main():
    weight_path = get_weight_path()
    data_output = get_model_config(PLUGIN_ID)
    force_cpu = data_output["force_cpu"]
    image = cv2.imread(os.path.join(tmp_path, BASE_IMG))
    output = get_seg(image, cpu_flag=force_cpu, weight_path=weight_path)
    cv2.imwrite(os.path.join(tmp_path, RESULT_IMG), output)
    set_model_config({
        "inference_status": "success",
        "force_cpu": force_cpu
    }, PLUGIN_ID)


if __name__ == "__main__":
    main()
