import os, sys, json, torch, cv2
import numpy as np

plugin_loc = os.path.join(os.path.dirname(os.path.realpath(__file__)), "PyTorch-Image-Dehazing")
sys.path.extend([plugin_loc])
import net

from gimpml.plugins.module_utils import *
from gimpml.plugins.dehaze.constants import *


def get_dehaze(data_hazy, cpu_flag=False, weight_path=None):
    checkpoint_path=os.path.join(weight_path, "deepdehaze", "dehazer.pth")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint file note found: {checkpoint_path}")

    data_hazy = data_hazy / 255.0
    data_hazy = torch.from_numpy(data_hazy).float()
    data_hazy = data_hazy.permute(2, 0, 1)
    dehaze_net = net.dehaze_net()

    if not cpu_flag and torch.cuda.is_available():
        dehaze_net = dehaze_net.cuda()
        dehaze_net.load_state_dict(torch.load(checkpoint_path))
        data_hazy = data_hazy.cuda()
    else:
        dehaze_net.load_state_dict(torch.load(checkpoint_path, map_location=torch.device("cpu"),))

    data_hazy = data_hazy.unsqueeze(0)
    with torch.no_grad():
        clean_image = dehaze_net(data_hazy)
    out = clean_image.detach().cpu().numpy()[0, :, :, :] * 255
    out = np.clip(np.transpose(out, (1, 2, 0)), 0, 255).astype(np.uint8)
    return out


@handle_exceptions(PLUGIN_ID)
def main():
    weight_path = get_weight_path()
    data_output = get_model_config(PLUGIN_ID)
    force_cpu = data_output["force_cpu"]
    image = cv2.imread(os.path.join(tmp_path, BASE_IMG))[:, :, ::-1]
    output = get_dehaze(image, cpu_flag=force_cpu, weight_path=weight_path)
    cv2.imwrite(os.path.join(tmp_path, RESULT_IMG), output[:, :, ::-1])
    set_model_config({
        "inference_status": "success",
        "force_cpu": force_cpu
    }, PLUGIN_ID)


if __name__ == "__main__":
    main()
