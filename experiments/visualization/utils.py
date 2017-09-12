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
