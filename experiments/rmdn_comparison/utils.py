import numpy as np
import scipy.stats


def gmm_to_probability_map(gmm, image_size):
    h, w = image_size

    y, x = np.mgrid[0:h:1, 0:w:1]
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = y
    pos[:, :, 1] = x

    out = np.zeros(shape=(h, w))

    for g in range(0, gmm.shape[0]):
        w = gmm[g, 0]
        normal = scipy.stats.multivariate_normal(mean=gmm[g, 1:3], cov=[[gmm[g, 3], gmm[g, 5]], [gmm[g, 5], gmm[g, 4]]])
        out += w * normal.pdf(pos)

    out /= out.sum()

    return out


if __name__ == '__main__':

    gmm = np.array([[0.5, 50, 0, 100, 100, 0], [0.5, 100, 100, 10, 10, -1]], dtype='float32')
    map = gmm_to_probability_map(gmm, image_size=(128, 171))

    from computer_vision_utils.io_helper import normalize
    import cv2
    cv2.imshow('GMM', normalize(map))
    cv2.waitKey()
