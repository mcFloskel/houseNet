import cv2
import numpy as np

from analysis.effective_receptive_field import get_effective_receptive_field
from networks.dnet import DNet

""" Example of calculating the effective receptive field and visualizing it with opencv.
"""


def prepare_receptive_field(receptive_field: np.ndarray,
                            up_scale_factor: int = 1,
                            grey_scale: bool = False):
    """Generate an numpy array which can be visualized or saved with cv2 functions.

    # Arguments:
        receptive_field: Numpy array
            array which represents the effective receptive field (channels last)
        up_scale_factor: int
            factor to use for up-scaling
        grey_scale: bool
            Convert array to gray values (mean over all color channels)

    # Returns:
        A Numpy array
    """

    shape = receptive_field.shape

    shape_up = (shape[0] * up_scale_factor, shape[1] * up_scale_factor)
    image = cv2.resize(receptive_field, shape_up)

    if grey_scale:
        image = image.mean(axis=2)
    return image


def visualize_receptive_field(receptive_field, up_scale_factor=1, grey_scale=False):
    """Shows the effective receptive field

    # Arguments:
        receptive_field: Numpy array
            array which represents the effective receptive field (channels last)
        up_scale_factor: int
            factor to use for up-scaling
        grey_scale: bool
            convert array to gray values (mean over all color channels)
    """

    image = prepare_receptive_field(receptive_field, up_scale_factor, grey_scale)
    cv2.imshow('Effective Receptive Field', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    net = DNet()
    rf = get_effective_receptive_field(net)
    visualize_receptive_field(rf, up_scale_factor=4)
