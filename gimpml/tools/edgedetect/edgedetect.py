import os, sys, cv2, torch
from torch.autograd import Variable

plugin_loc = os.path.join(os.path.dirname(os.path.realpath(__file__)), "DexiNed")
sys.path.extend([plugin_loc])
from losses import *
from model import DexiNed
from image import tensor_to_image

from gimpml.plugins.module_utils import *
from gimpml.plugins.edgedetect.constants import *
from gimpml.tools.tools_helpers import add_alpha, color_to_alpha


CLASSIC_CONFIG = {
    'img_height': 512,
    'img_width': 512,
    'yita': 0.5,
    'mean_bgr': [103.939,116.779,123.68]#, 137.86],
}

def get_edges(image, cpu_flag=False, weight_path=None):
    w, h = image.shape[:2]
    checkpoint_path=os.path.join(weight_path, "edgedetect", "10_model.pth")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file note found: {checkpoint_path}")

    device = torch.device('cpu' if cpu_flag or torch.cuda.device_count() == 0 else 'cuda')
    # Instantiate model and move it to the computing device
    model = DexiNed().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval() # Put model in evaluation mode

    with torch.no_grad():          
        img_height = CLASSIC_CONFIG["img_height"]
        img_width =CLASSIC_CONFIG["img_width"]
        
        img = cv2.resize(image, (img_width, img_height))
        gt = None
        img = np.array(img, dtype=np.float32)
        img -= CLASSIC_CONFIG["mean_bgr"]
        img = img.transpose((2, 0, 1))
        im_input = (
            torch.from_numpy(img.astype(np.float32).copy())
            #.permute((2, 0, 1))
            .unsqueeze(0)
        )
        #print(im_input.shape)

        if not cpu_flag and device.type == 'cuda':
            model = model.cuda()
            im_input = im_input.cuda()
        else:
            model = model.cpu()
    
        if device.type == 'cuda': torch.cuda.synchronize()
        tensor = model(im_input)
        if device.type == 'cuda': torch.cuda.synchronize()

        edge_maps = []
        for i in tensor:
            tmp = torch.sigmoid(i).cpu().detach().numpy()
            edge_maps.append(tmp)
        tensor = np.array(edge_maps)
        #print(f"tensor shape: {tensor.shape}")

        # (H, W) -> (W, H)
        i_shape = [img_width, img_height]
        tensor2 = None
        tmp_img2 = None

        idx = 0
        tmp = tensor[:, idx, ...]
        tmp2 = tensor2[:, idx, ...] if tensor2 is not None else None
        # tmp = np.transpose(np.squeeze(tmp), [0, 1, 2])
        tmp = np.squeeze(tmp)
        tmp2 = np.squeeze(tmp2) if tensor2 is not None else None

        # Iterate our all 7 NN outputs for a particular image
        preds = []
        for i in range(tmp.shape[0]):
            tmp_img = tmp[i]
            tmp_img = np.uint8(image_normalization(tmp_img))
            tmp_img = cv2.bitwise_not(tmp_img)
            # tmp_img[tmp_img < 0.0] = 0.0
            # tmp_img = 255.0 * (1.0 - tmp_img)
            if tmp2 is not None:
                tmp_img2 = tmp2[i]
                tmp_img2 = np.uint8(image_normalization(tmp_img2))
                tmp_img2 = cv2.bitwise_not(tmp_img2)

            # Resize prediction to match input image size
            if not tmp_img.shape[1] == i_shape[0] or not tmp_img.shape[0] == i_shape[1]:
                tmp_img = cv2.resize(tmp_img, (i_shape[0], i_shape[1]))
                tmp_img2 = cv2.resize(tmp_img2, (i_shape[0], i_shape[1])) if tmp2 is not None else None

            if tmp2 is not None:
                tmp_mask = np.logical_and(tmp_img>128,tmp_img2<128)
                tmp_img= np.where(tmp_mask, tmp_img2, tmp_img)
                preds.append(tmp_img)
            else:
                preds.append(tmp_img)

            if i == 6:
                fuse = tmp_img
                fuse = fuse.astype(np.uint8)
                if tmp_img2 is not None:
                    fuse2 = tmp_img2
                    fuse2 = fuse2.astype(np.uint8)
                    # fuse = fuse-fuse2
                    fuse_mask=np.logical_and(fuse>128,fuse2<128)
                    fuse = np.where(fuse_mask,fuse2, fuse)
                    # print(fuse.shape, fuse_mask.shape)

        # Get the mean prediction of all the 7 outputs
        #average = np.array(preds, dtype=np.float32)
        #average = np.uint8(np.mean(average, axis=0))

    torch.cuda.empty_cache()
    fuse = cv2.resize(fuse, (h, w))
    return fuse#average


@handle_exceptions(PLUGIN_ID)
def main():
    weight_path = get_weight_path()
    data_output = get_model_config(PLUGIN_ID)
    force_cpu = data_output["force_cpu"]
    image = cv2.imread(os.path.join(tmp_path, BASE_IMG))[:, :, ::-1]
    output = get_edges(image, cpu_flag=force_cpu, weight_path=weight_path)
    output = add_alpha(output)
    output = color_to_alpha(output)
    cv2.imwrite(os.path.join(tmp_path, RESULT_IMG), output)
    set_model_config({
        "inference_status": "success",
        "force_cpu": force_cpu
    }, PLUGIN_ID)


if __name__ == "__main__":
    main()
