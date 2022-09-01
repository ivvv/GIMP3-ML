import os, sys, torch, cv2
import numpy as np
from PIL import Image

plugin_loc = os.path.join(os.path.dirname(os.path.realpath(__file__)), "ncoloriz")
sys.path.extend([plugin_loc])
from model import generator

from gimpml.plugins.module_utils import *
from gimpml.plugins.coloring.constants import *
from gimpml.tools.tools_helpers import handle_alpha, to_grayscale, yuv2rgb
from gimpml.tools.base_class import ModelBase


class NeuralColorization(ModelBase):
    def __init__(self, config):
        super().__init__(config)
        self.model_name = os.path.join(weight_path, "colorize", "G.pth")

    def load_model(self):
        G = generator()
        #device = torch.device(self.device)
        G.load_state_dict(torch.load(self.model_name, map_location=self.device))
        G.to(self.device)
        return G

    @handle_alpha
    @torch.no_grad()
    def predict(self, img):
        h, w, d = img.shape
        if d == 3:
            img = to_grayscale(img)

        G = self.model

        input_image = img[..., 0]
        H, W = input_image.shape
        infimg = input_image[None, None, ...].astype(np.float32) / 255.
        img_variable = torch.from_numpy(infimg) - 0.5
        img_variable = img_variable.to(self.device)
        res = G(img_variable)
        uv = res.cpu().numpy()[0]
        uv[0, :, :] *= 0.436
        uv[1, :, :] *= 0.615
        u = np.array(Image.fromarray(uv[0]).resize((W, H), Image.BILINEAR))[None, ...]
        v = np.array(Image.fromarray(uv[1]).resize((W, H), Image.BILINEAR))[None, ...]
        yuv = np.concatenate([infimg[0], u, v], axis=0).transpose(1, 2, 0)

        rgb = yuv2rgb(yuv * 255)
        rgb = rgb.clip(0, 255).astype(np.uint8)
        return rgb

@handle_exceptions(PLUGIN_ID)
def main():
    config = get_model_config(PLUGIN_ID)
    model = NeuralColorization(config)
    out = model.predict(cv2.imread(os.path.join(tmp_path, BASE_IMG), cv2.IMREAD_UNCHANGED))
    cv2.imwrite(os.path.join(tmp_path, RESULT_IMG), out)
    config.update({"inference_status": "success", "force_cpu": model.device == "cpu"})
    set_model_config(config, PLUGIN_ID)

if __name__ == "__main__":
    main()