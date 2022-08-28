import os, sys, random, cv2, torch
import numpy as np

plugin_loc = os.path.join(os.path.dirname(os.path.realpath(__file__)), "edge-connect")
sys.path.extend([plugin_loc])
from src.edge_connect import EdgeConnect

from src.config import Config
from skimage.feature import canny

from gimpml.plugins.module_utils import *
from gimpml.plugins.animeinp.constants import *
from gimpml.tools.tools_helpers import add_alpha


def get_inpaint(
    images, 
    masks, 
    ext_edges, 
    cpu_flag=False,
    model_name="getchu",
    enable_edges=False):
    config = Config()
    config._dict = {
        "MODE":2,             # 1: train, 2: test, 3: eval
        "MODEL":3,            # 1: edge model, 2: inpaint model, 3: edge-inpaint model, 4: joint model
        "MASK":3,             # 1: random block, 2: half, 3: external, 4: (external, random block), 5: (external, random block, half)
        "EDGE":1,             # 1: canny, 2: external
        "NMS":1,              # 0: no non-max-suppression, 1: applies non-max-suppression on the external edges by multiplying by Canny
        "SEED":10,            # random seed
        "DEVICE":0,           # 0: CPU, 1: GPU
        "GPU":[0],            # list of gpu ids
        "DEBUG":0,            # turns on debugging mode
        "VERBOSE":0,          # turns on verbose mode in the output console
        "SKIP_PHASE2": 1,      # Training model, 2nd and 3rd phases by order are needed. We can merge 2nd into the 3rd to speed up (low performance).
        
        "LR": 0.0001,                    # learning rate
        "D2G_LR": 0.1,                   # discriminator/generator learning rate ratio
        "BETA1":  0.0,                    # adam optimizer beta1
        "BETA2": 0.9,                    # adam optimizer beta2
        "BATCH_SIZE": 8,                 # input batch size for training
        "INPUT_SIZE": 128,               # input image size for training 0 for original size
        "SIGMA": 1.5,                      # standard deviation of the Gaussian filter used in Canny edge detector (0: random, -1: no edge)
        "MAX_ITERS":2e7,                # maximum number of iterations to train the model
        
        "EDGE_THRESHOLD":0.5,           # edge detection threshold
        "L1_LOSS_WEIGHT":1,             # l1 loss weight
        "FM_LOSS_WEIGHT":10,            # feature-matching loss weight
        "STYLE_LOSS_WEIGHT":1,          # style loss weight
        "CONTENT_LOSS_WEIGHT":1,        # perceptual loss weight
        "INPAINT_ADV_LOSS_WEIGHT":0.01, # adversarial loss weight
        
        "GAN_LOSS": "nsgan",               # nsgan | lsgan | hinge
        "GAN_POOL_SIZE":0,              # fake images pool size
        "SAVE_INTERVAL":30,           # how many iterations to wait before saving model (0: never)
        "SAMPLE_INTERVAL":200,         # how many iterations to wait before sampling (0: never)
        "SAMPLE_SIZE":12,               # number of images to sample
        "EVAL_INTERVAL":0,              # how many iterations to wait before model evaluation (0: never)
        "LOG_INTERVAL":1000,              # how many iterations to wait before logging training status (0: never)
        "PATH": os.path.join(weight_path, model_name),
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

    # enable the cudnn auto-tuner for hardware.
    torch.backends.cudnn.benchmark = True

    # build the model and initialize
    model = EdgeConnect(config)
    model.load()
   
    """
    if not enable_edges:
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
            torch.from_numpy(
                images.astype(np.float32).copy()).permute((2, 0, 1)
            ).unsqueeze(0)
        )
        images_gray = (
            torch.from_numpy(images_gray.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        )
        masks = torch.from_numpy(masks.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        edges = torch.from_numpy(edge.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        """

    if not enable_edges:
        model.edge_model.eval()
        model.inpaint_model.eval()

        """
        if config.DEVICE.type == "cuda":
            images, images_gray, edges, masks = model.cuda(
                *(images, images_gray, edges, masks)
            )
        """

    with torch.no_grad():
        if ext_edges is None:
            output, edges = model.test_img_with_mask(images, masks)
        else:
            output, edges = model.test_img_with_mask(images, masks, edge_edit=ext_edges)

    #output = model.postprocess(outputs_merged)[0]
    return output#np.uint8(output.cpu())

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
    #mask_orig[mask_orig > 0] = 255

    mask = cv2.resize(mask_orig, (256, 256))[:, :, 0]
    picture = cv2.resize(picture, (256, 256))[:, :, ::-1]
    edge = None

    inpaint = get_inpaint(
        picture,
        mask,
        edge,
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
