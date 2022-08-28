import numpy as np

def add_alpha(img):
    import cv2
    height, width, channels = img.shape
    if channels < 4:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
    return cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)

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
