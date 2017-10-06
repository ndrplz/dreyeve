import numpy as np
import cv2


def blend_map(img, map, factor, colormap=cv2.COLORMAP_JET):
    """
    Function to blend an image and a probability map.

    :param img: The image
    :param map: The map
    :param factor: factor * img + (1-factor) * map
    :param colormap: a cv2 colormap.
    :return: The blended image.
    """

    assert 0 < factor < 1, 'factor must satisfy 0 < factor < 1'

    map = np.float32(map)
    map /= map.max()
    map *= 255
    map = map.astype(np.uint8)

    blend = cv2.addWeighted(src1=img, alpha=factor,
                            src2=cv2.applyColorMap(map, colormap), beta=(1-factor),
                            gamma=0)

    return blend


palette = np.array([[128, 64, 128],
                    [244, 35, 232],
                    [70, 70, 70],
                    [102, 102, 156],
                    [190, 153, 153],
                    [153, 153, 153],
                    [250, 170, 30],
                    [220, 220, 0],
                    [107, 142, 35],
                    [152, 251, 152],
                    [70, 130, 180],
                    [220, 20, 60],
                    [255, 0, 0],
                    [0, 0, 142],
                    [0, 0, 70],
                    [0, 60, 100],
                    [0, 80, 100],
                    [0, 0, 230],
                    [119, 11, 32]], dtype=np.uint8)


def seg_to_rgb(segm):

    prediction = np.argmax(segm, axis=0)

    h, w = prediction.shape

    rgb = np.reshape(palette[prediction.ravel()], newshape=(h, w, 3))
    return rgb
