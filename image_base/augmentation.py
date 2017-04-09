import matplotlib.path as mplPath
import numpy as np
import cv2


def make_poly_mask(shape, polygon, background=0):

    array = np.ones(shape,dtype=np.float32)*background
    cv2.fillConvexPoly(array, polygon, color=1)

    return array


