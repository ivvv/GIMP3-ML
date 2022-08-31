import numpy as np

def remove_alpha(img):
    if len(img.shape) == 3 and img.shape[2] == 4:  # get rid of alpha channel
        return img[:, :, 0:3]
    else:
        return img

def add_alpha(img):
    import cv2
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGBA)
    height, width, channels = img.shape
    if channels < 4:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
    return cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    
def color_to_alpha(img, alpha_color=[255, 255, 255]):
    #img[:, :, 3] = (255 * (img[:, :, :3] != 255).any(axis=2)).astype(np.uint8)
    #img[:, :, 3] = (255 - img[:, :, :3].mean(axis=2)).astype(np.uint8)
    alpha = np.max(
        [
            np.abs(img[..., 0] - alpha_color[0]),
            np.abs(img[..., 1] - alpha_color[1]),
            np.abs(img[..., 2] - alpha_color[2]),
        ],
        axis=0,
    )
    ny, nx, _ = img.shape
    img = np.zeros((ny, nx, 4), dtype=img.dtype)
    for i in range(3):
        img[..., i] = img[..., i]
    img[..., 3] = alpha
    return img

def yuv2rgb(yuv):
    """ Converts YUV numpy array  to RGB. """
    rgb_from_yuv = np.linalg.inv(yuv_from_rgb_mat)
    return np.dot(yuv, rgb_from_yuv.T.copy())

def to_grayscale(rgb):
    """ Converts RGB numpy array  to grayscale. """
    return np.dot(rgb, yuv_from_rgb_mat[0, None].T)

def split_alpha(array: np.ndarray):
    """ Converts numpy array to (RGB, alpha). """
    h, w, d = array.shape
    if d == 1: return array, None
    elif d == 2: return array[:, :, 0:1], array[:, :, 1:2]
    elif d == 3: return array, None
    elif d == 4: return array[:, :, 0:3], array[:, :, 3:4]
    raise ValueError("Image has too many channels ({}), expected <4".format(d))

def merge_alpha(image: np.ndarray, alpha: np.ndarray):
    """ Merges numpy array from (RGB, alpha) to RGBA. """
    h, w, d = image.shape
    if d not in (1, 3): raise ValueError("Incorrect number of channels ({}), expected 1..3".format(d))
    if alpha is None: return image
    return np.concatenate([image, alpha], axis=2)


def combine_alphas(alphas: list[np.ndarray]):
    """ Merges several alpha channels to a single channel. """
    combined_alpha = None
    for alpha in alphas:
        if alpha is not None:
            if combined_alpha is None:
                combined_alpha = alpha
            else:
                combined_alpha = combined_alpha * (alpha / 255.)
    if combined_alpha is not None:
        combined_alpha = combined_alpha.astype(np.uint8)
    return combined_alpha

def handle_alpha(func):
    """ Returns  wrapper for parsing alpha channel of any function. """
    def decorator(*args, **kwargs):
        alphas = []
        args = list(args)
        for i, arg in enumerate(args):
            if isinstance(arg, np.ndarray):
                img, alpha = split_alpha(arg)
                args[i] = img
                alphas.append(alpha)
        for key, arg in list(kwargs.items()):
            if isinstance(arg, np.ndarray):
                img, alpha = split_alpha(arg)
                kwargs[key] = img
                alphas.append(alpha)

        result = func(*args, **kwargs)
        alpha = combine_alphas(alphas)

        # for super-res
        if alpha is not None and result.shape[:2] != alpha.shape[:2]:
            h, w, d = result.shape
            alpha = np.ndarray(Image.fromarray(alpha[..., 0]).resize((w, h), Image.BILINEAR))[..., None]

        result = merge_alpha(result, alpha)
        return result

    return decorator

def to_rgb(image: np.ndarray):
    """ Converts numpy array image to RGB """
    if len(image.shape) == 2:
        image = image[:, :, None]
    return image[:, :, (0, 0, 0)]


def apply_colormap(image: np.ndarray, cmap='magma'):
    """ Applies color map numpy array image """
    # image must be in 0-1 range
    import matplotlib.cm as cm
    mapper = cm.ScalarMappable(norm=lambda x: x, cmap=cmap)
    return mapper.to_rgba(image)[:, :, :3]

def np_alpha_composite(nsrc, ndst):
    """ Mixes two images with alpha channels. """
    srcRGB = nsrc[...,:3]
    dstRGB = ndst[...,:3]
    srcA = nsrc[...,3]/255.0
    dstA = ndst[...,3]/255.0
    outA = srcA + dstA*(1-srcA)
    outRGB = (srcRGB*srcA[...,np.newaxis] + dstRGB*dstA[...,np.newaxis]*(1-srcA[...,np.newaxis])) / outA[...,np.newaxis]
    return np.dstack((outRGB, outA*255)).astype(np.uint8)
