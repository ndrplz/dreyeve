import numpy as np

# cityscapes dataset palette
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
                    [119, 11, 32]], dtype='uint8')


def seg_to_colormap(seg, channels_first):
    """
    Function to turn segmentation PREDICTION (not probabilities) to colormap.

    :param seg: the prediction image, having shape (h,w)
    :param channels_first: if true, returns (c,h,w) rather than (h,w,c)
    :return: the colormap image, having shape (h,w,3)
    """
    h, w = seg.shape
    color_image = palette[seg.ravel()].reshape(h, w, 3)

    if channels_first:
        color_image = color_image.transpose(2, 0, 1)

    return color_image


def read_lines_from_file(filename):
    """
    Function to read lines from file

    :param filename: The text file to be read.
    :return: content: A list of strings
    """
    with open(filename) as f:
        content = f.readlines()

    content = [x.strip() for x in content]
    return content


def get_branch_from_experiment_id(experiment_id):
    """
    Function to return model branch name given experiment_id.
    :param experiment_id: experiment id
    :return: a string among ['all','image','optical_flow','semseg']
    """

    assert isinstance(experiment_id, basestring), "Experiment ID must be a string."

    branch = None
    if experiment_id.lower().startswith('dreyeve'):
        branch = "all"
    elif experiment_id.lower().startswith('color'):
        branch = "image"
    elif experiment_id.lower().startswith('flow'):
        branch = "optical_flow"
    elif experiment_id.lower().startswith('segm'):
        branch = "semseg"

    return branch
