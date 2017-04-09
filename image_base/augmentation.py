import matplotlib.path as mplPath
import numpy as np
import cv2


def make_poly_mask(shape, polygon):

    array = np.zeros(shape,dtype=np.float32)
    cv2.fillConvexPoly(array, polygon, color=1)

    return array


