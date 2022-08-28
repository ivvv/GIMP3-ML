import json, os, sys, cv2, torch
import numpy as np

plugin_loc = os.path.join(os.path.dirname(os.path.realpath(__file__)), "DPT")
sys.path.extend([plugin_loc])
from monodepth_run import run

from gimpml.plugins.module_utils import *
from gimpml.plugins.monodepth.constants import *


def get_mono_depth(input_image, cpu_flag=False, weight_path=None, absolute_depth=False):
    with torch.no_grad():
        out = run(
            input_image,
            os.path.join(weight_path, "MiDaS", "dpt_hybrid-midas-501f0c75.pt"),
            cpu_flag=cpu_flag,
            bits=2,
            absolute_depth=absolute_depth,
        )

    out = np.repeat(out[:, :, np.newaxis], 3, axis=2)
    d1, d2 = input_image.shape[:2]
    out = cv2.resize(out, (d2, d1))
    return out.astype("uint16")


@handle_exceptions(PLUGIN_ID)
def main():
    weight_path = get_weight_path()
    data_output = get_model_config(PLUGIN_ID)
    force_cpu = data_output["force_cpu"]
    image = cv2.imread(os.path.join(tmp_path, BASE_IMG))
    output = get_mono_depth(image[:, :, ::-1], cpu_flag=force_cpu, weight_path=weight_path)
    cv2.imwrite(
        os.path.join(tmp_path, RESULT_IMG),
        output,
        [cv2.IMWRITE_PNG_COMPRESSION, 0],
    )
    set_model_config({
        "inference_status": "success",
        "force_cpu": force_cpu
    }, PLUGIN_ID)


if __name__ == "__main__":
    main()
