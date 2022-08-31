import errno, math, sys, os, torch, cv2
import numpy as np
from scipy.signal import argrelextrema
from PIL import Image

plugin_loc = os.path.join(os.path.dirname(os.path.realpath(__file__)), "demosaic_project")
sys.path.extend([plugin_loc])
import architecture as arch

from gimpml.plugins.esrdecensor.constants import *
from gimpml.plugins.module_utils import *
from gimpml.tools.tools_helpers import np_alpha_composite


def create_masks(LowRange, HighRange):
    patterns = {}
    for masksize in range(HighRange+2, LowRange+1, -1):
        maskimg = 2+masksize+masksize-1+2
        pix = np.zeros((maskimg, maskimg, 3), np.uint8)
        pix.fill(255)
        for i in range(2, maskimg, masksize-1):
            for j in range(2, maskimg, masksize-1):
                for k in range(0, maskimg):
                    pix[i, k] = [0, 0, 0]
                    pix[k, j] = [0, 0, 0]
        patterns[masksize-2] = pix
        #print(masksize-2, patterns[masksize-2].shape)
    return patterns

def get_demosaic(
        images, masks, cpu_flag=False,
        model_name="4x_FatalPixels_340000_G.pth"):

    #print("\n-----------------------ESRGAN-Init-----------------------\n")
    if not cpu_flag:
        GPUmem = torch.cuda.get_device_properties(0).total_memory
    else:
        GPUmem = int(MEM_IN_GiB * 1000000000)

        # models/RRDB_ESRGAN_x4.pth OR models/RRDB_PSNR_x4.pth
        pth_model = os.path.join(weight_path, PLUGIN_ID, model_name)
    if not os.path.isfile(pth_model):
        print(f"No model weights found at {pth_model}")
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), pth_model)

    device = torch.device('cpu' if cpu_flag else 'cuda')
    model = arch.RRDB_Net(3, 3, 64, 23, gc=32, upscale=4, norm_type=None, act_type='leakyrelu',
                          mode='CNA', res_scale=1, upsample_mode='upconv')
    model.load_state_dict(torch.load(pth_model), strict=True)
    model.eval()

    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)

    #print("\n-----------------------Logic-----------------------\n")
    # GBlur = 5     CannyTr1 = 20   CannyTr2 = 100
    # LowRange = 2  HighRange = 20
    # DetectionTr = 0.32

    GBlur = 5
    CannyTr1 = 8
    CannyTr2 = 30
    LowRange = 2
    HighRange = 25
    DetectionTr = 0.29
    
    patterns = create_masks(LowRange, HighRange)
    
    #print("\n-----------------------Files-----------------------\n")
    x, y, c = images.shape
    if c < 4:
        alpha = np.full((images.shape[0], images.shape[1]), 255, dtype=np.uint8)
        images= np.dstack((images, alpha))
    
    card = np.zeros((x, y, 4), np.uint8) 
    card[:] = [255, 255, 255, 0]
    cvI = np_alpha_composite(card, images)
    card = cv2.cvtColor(card, cv2.COLOR_BGRA2RGBA)
    #print(images.shape, card.shape)

    img_rgb = cv2.cvtColor(cvI, cv2.COLOR_BGRA2RGBA)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.Canny(img_gray, CannyTr1, CannyTr2)
    img_gray = 255-img_gray
    img_gray = cv2.GaussianBlur(img_gray, (GBlur, GBlur), 0)

    #print("\n-----------------------Detection-----------------------\n")
    resolutions = [-1] * (HighRange+2)
    for masksize in range(HighRange+2, LowRange+1, -1):
        template = cv2.cvtColor(patterns[masksize-2], cv2.COLOR_BGR2GRAY)
        w, h, _ = patterns[masksize-2].shape
        img_detection = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
        loc = np.where(img_detection >= DetectionTr)
        rects = 0
        for pt in zip(*loc[::-1]):
            rects += 1  # increase rectangle count of single resolution
            cv2.rectangle(
                card, pt, (pt[0] + w, pt[1] + h), (0, 0, 0, 255), -1)
        resolutions[masksize-1] = rects

    #print("\n---------------------Calculating-Resolution-----------------------\n")
    resolutions.append(0)
    extremaMIN = argrelextrema(np.array(resolutions), np.less, axis=0)[0]
    extremaMIN = np.insert(extremaMIN, 0, LowRange)
    extremaMIN = np.append(extremaMIN, HighRange+2)

    Extremas = []
    for i, ExtGroup in enumerate(extremaMIN[:-1]):
        Extremas.append(
            (ExtGroup, resolutions[extremaMIN[i]:extremaMIN[i+1]+1]))

    ExtremasSum = []
    BigExtrema = [0, 0, [0, 0]]
    for i, _ in enumerate(Extremas):
        ExtremasSum.append(sum(Extremas[i][1]))
        # 5% precedency for smaller resolution
        if BigExtrema[0] <= sum(Extremas[i][1])+int(sum(Extremas[i][1])*0.05):
            if max(BigExtrema[2]) < max(Extremas[i][1])+max(Extremas[i][1])*0.15:
                BigExtrema = [sum(Extremas[i][1]),
                              Extremas[i][0], Extremas[i][1]]

    MosaicResolutionOfImage = BigExtrema[1] + \
        BigExtrema[2].index(max(BigExtrema[2]))
    if MosaicResolutionOfImage == 0:  # If nothing found - set resolution as smallest
        MosaicResolutionOfImage = HighRange+1
    print('Mosaic resolution is: ' + str(MosaicResolutionOfImage))  # The Resolution of Mosaiced Image

    #print("\n-----------------------ESRGAN-Processing-----------------------\n")
    Sx = int(1.2*x/MosaicResolutionOfImage)
    Sy = int(1.2*y/MosaicResolutionOfImage)
    #print(Sx, Sy)
    shrinkedI = cv2.resize(img_rgb, (Sx, Sy))
    maxres = math.sqrt((Sx*Sy)/(GPUmem*0.00008))
    if maxres > 1:
        shrinkedI = cv2.resize(shrinkedI, (int(Sx/maxres), int(Sy/maxres)))
    # print(maxres)
    while True:
        imgESR = cv2.cvtColor(shrinkedI, cv2.COLOR_RGBA2RGB)
        imgESR = imgESR * 1.0 / 255

        imgESR = torch.from_numpy(np.transpose(
            imgESR[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img_LR = imgESR.unsqueeze(0)
        img_LR = img_LR.to(device)

        output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
        #print(output.shape)
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round()
        if MosaicResolutionOfImage > 7:
            MosaicResolutionOfImage = MosaicResolutionOfImage*0.25
            # print("iter")
            height, width, _ = output.shape
            #print(width, height)
            maxres = math.sqrt((width*height)/(GPUmem*0.00008))
            if maxres > 1:
                shrinkedI = cv2.resize(output, (int(width/maxres), int(height/maxres)))
            else:
                shrinkedI = output
            # print(maxres)
            continue
        break

    #print("\n -----------------------Unification-and-Saving-----------------------\n")
    imgESRbig = cv2.resize(output, (y, x), cv2.INTER_AREA)
    imgESRbig = imgESRbig.astype(np.uint8)
    imgESRbig = cv2.cvtColor(imgESRbig, cv2.COLOR_RGB2RGBA)
    img2gray = cv2.cvtColor(imgESRbig, cv2.COLOR_RGBA2GRAY)
    ret, mask = cv2.threshold(img2gray, 245, 255, cv2.THRESH_BINARY)
    imgESRbig = cv2.bitwise_not(imgESRbig, imgESRbig, mask=mask)
    card = cv2.cvtColor(card, cv2.COLOR_RGBA2GRAY)
    card_inv = cv2.bitwise_not(card)
    #print(imgESRbig.shape, mask.shape, card_inv.shape)
    mosaic_reg = cv2.bitwise_and(imgESRbig, imgESRbig, mask=card_inv)
    mosaic_reg = cv2.cvtColor(mosaic_reg, cv2.COLOR_BGRA2RGBA)
    #output = np_alpha_composite(images, mosaic_reg)
    #print(mosaic_reg.shape)
    return mosaic_reg#output


@handle_exceptions(PLUGIN_ID)
def main():
    data_output = get_model_config(PLUGIN_ID)

    maskfn = os.path.join(tmp_path, MASK_IMG)
    if os.path.isfile(maskfn):
        mask_orig = cv2.imread(os.path.join(tmp_path, MASK_IMG))
    else:
        mask_orig = None
    picture = cv2.imread(os.path.join(tmp_path, BASE_IMG))

    force_cpu = data_output["force_cpu"]
    #model_name = data_output["model_name"]

    # invert, mask layer uses white as allow color
    if mask_orig:
        mask = cv2.bitwise_not(mask_orig)
    else:
        mask = None

    demosaic = get_demosaic(picture, mask, cpu_flag=force_cpu)

    cv2.imwrite(os.path.join(tmp_path, RESULT_IMG), demosaic)
    set_model_config({
        "inference_status": "success",
        "force_cpu": force_cpu,
        # "model_name": model_name
    }, PLUGIN_ID)


if __name__ == "__main__":
    main()
