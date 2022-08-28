import os, sys, random, cv2, torch
import numpy as np

plugin_loc = os.path.join(os.path.dirname(os.path.realpath(__file__)), "edge-connect")
sys.path.extend([plugin_loc])
from src.edge_connect import EdgeConnect
from src.config import Config
from skimage.feature import canny

from gimpml.plugins.module_utils import *
from gimpml.plugins.inpainting.constants import *
from gimpml.tools.tools_helpers import add_alpha


def get_inpaint(
    images,
    masks,
    cpu_flag=False,
    model_name="places2"):
    config = Config()
    config._dict = {
        "MODE": 2,
        "MODEL": 3,
        "MASK": 3,
        "EDGE": 1,
        "NMS": 1,
        "SEED": 10,
        "GPU": [0],
        "DEBUG": 0,
        "VERBOSE": 0,
        "LR": 0.0001,
        "D2G_LR": 0.1,
        "BETA1": 0.0,
        "BETA2": 0.9,
        "BATCH_SIZE": 8,
        "INPUT_SIZE": 256,
        "SIGMA": 2,
        "MAX_ITERS": "2e6",
        "EDGE_THRESHOLD": 0.5,
        "L1_LOSS_WEIGHT": 1,
        "FM_LOSS_WEIGHT": 10,
        "STYLE_LOSS_WEIGHT": 250,
        "CONTENT_LOSS_WEIGHT": 0.1,
        "INPAINT_ADV_LOSS_WEIGHT": 0.1,
        "GAN_LOSS": "nsgan",
        "GAN_POOL_SIZE": 0,
        "SAVE_INTERVAL": 1000,
        "SAMPLE_INTERVAL": 1000,
        "SAMPLE_SIZE": 12,
        "EVAL_INTERVAL": 0,
        "LOG_INTERVAL": 10,
        "PATH": os.path.join(weight_path, "edgeconnect", model_name),
    }

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(e) for e in config.GPU)

    # init device
    if torch.cuda.is_available() and not cpu_flag:
        config.DEVICE = torch.device("cuda")
        torch.backends.cudnn.benchmark = True  # cudnn auto-tuner
    else:
        config.DEVICE = torch.device("cpu")

    # set cv2 running threads to 1 (prevents deadlocks with pytorch dataloader)
    cv2.setNumThreads(0)

    # initialize random seed
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)

    # build the model and initialize
    model = EdgeConnect(config)
    model.load()
    images_gray = cv2.cvtColor(images, cv2.COLOR_RGB2GRAY)
    sigma = config.SIGMA

    if sigma == -1:
        sigma = random.randint(1, 4)

    masks = masks / 255
    edge = canny(images_gray, sigma=sigma, mask=(1 - masks).astype(bool)).astype(
        np.float32
    )
    images_gray = images_gray / 255
    images = images / 255

    images = (
        torch.from_numpy(images.astype(np.float32).copy())
        .permute((2, 0, 1))
        .unsqueeze(0)
    )
    images_gray = (
        torch.from_numpy(images_gray.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    )
    masks = torch.from_numpy(masks.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    edges = torch.from_numpy(edge.astype(np.float32)).unsqueeze(0).unsqueeze(0)

    model.edge_model.eval()
    model.inpaint_model.eval()

    if config.DEVICE.type == "cuda":
        images, images_gray, edges, masks = model.cuda(*(images, images_gray, edges, masks))

    # edge model
    if config.MODEL == 1:
        with torch.no_grad():
            outputs = model.edge_model(images_gray, edges, masks)
        outputs_merged = (outputs * masks) + (edges * (1 - masks))

    # inpaint model
    elif config.MODEL == 2:
        with torch.no_grad():
            outputs = model.inpaint_model(images, edges, masks)
        outputs_merged = (outputs * masks) + (images * (1 - masks))

    # inpaint with edge model / joint model
    else:
        with torch.no_grad():
            edges = model.edge_model(images_gray, edges, masks).detach()
            outputs = model.inpaint_model(images, edges, masks)
        outputs_merged = (outputs * masks) + (images * (1 - masks))

    output = model.postprocess(outputs_merged)[0]
    return np.uint8(output.cpu())

@handle_exceptions(PLUGIN_ID)
def main():
    data_output = get_model_config(PLUGIN_ID)

    mask_orig = cv2.imread(os.path.join(tmp_path, MASK_IMG))
    picture = cv2.imread(os.path.join(tmp_path, BASE_IMG))
    force_cpu = data_output["force_cpu"]
    model_name = data_output["model_name"]
    h, w, c = mask_orig.shape

    # invert, mask layer uses white as allow color
    mask_orig = cv2.bitwise_not(mask_orig)

    mask = cv2.resize(mask_orig, (256, 256))
    picture = cv2.resize(picture, (256, 256))

    inpaint = get_inpaint(
        picture[:, :, ::-1],
        mask[:, :, 0],
        cpu_flag=force_cpu,
        model_name=model_name,
    )
    inpaint = cv2.resize(inpaint, (w, h), interpolation = cv2.INTER_LANCZOS4)

    # make only modified pixels non-transparent
    inpaint = add_alpha(inpaint)
    inpaint[...,3] = mask_orig[...,1]

    cv2.imwrite(os.path.join(tmp_path, RESULT_IMG), inpaint)
    set_model_config({
            "inference_status": "success",
            "force_cpu": force_cpu,
            "model_name": model_name
    }, PLUGIN_ID)

if __name__ == "__main__":
    main()
