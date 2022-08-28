import os, sys, cv2, torch
import numpy as np
from scipy.cluster.vq import kmeans2

from gimpml.plugins.module_utils import *
from gimpml.plugins.kmeans.constants import *


def get_kmeans(image, loc_flag=False, n_clusters=3):
    if image.shape[2] == 4:  # get rid of alpha channel
        image = image[:, :, 0:3]
    h, w, d = image.shape
    # reshape the image to a 2D array of pixels and 3 color values (RGB)
    pixel_values = image.reshape((-1, 3))

    if loc_flag:
        xx, yy = np.meshgrid(range(w), range(h))
        x = xx.reshape(-1, 1)
        y = yy.reshape(-1, 1)
        pixel_values = np.concatenate((pixel_values, x, y), axis=1)

    pixel_values = np.float32(pixel_values)
    c, out = kmeans2(pixel_values, n_clusters)

    if loc_flag:
        c = np.uint8(c[:, 0:3])
    else:
        c = np.uint8(c)
    segmented_image = c[out.flatten()]
    segmented_image = segmented_image.reshape((h, w, d))
    return segmented_image


@handle_exceptions(PLUGIN_ID)
def main():
    weight_path = get_weight_path()
    data_output = get_model_config(PLUGIN_ID)
    n_cluster = data_output["n_cluster"]
    position = data_output["position"]
    image = cv2.imread(os.path.join(tmp_path, BASE_IMG))[:, :, ::-1]
    output = get_kmeans(image, loc_flag=position, n_clusters=n_cluster)
    cv2.imwrite(os.path.join(tmp_path, RESULT_IMG), output[:, :, ::-1])
    set_model_config({
        "inference_status": "success"
    }, PLUGIN_ID)


if __name__ == "__main__":
    main()

