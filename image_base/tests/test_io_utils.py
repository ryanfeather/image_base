from image_base import  io_utils as IU

import os
import inspect

def my_data_path(p):




def test_get_im_cv2():
    im  = IU.get_im_cv2('test_data/test100x1000.jpg')
    assert im.shape == (100,1000,3)

    im = IU.get_im_cv2('test_data/test100x1000.jpg',upscale=False,downscale=False,size=(100,100))
    assert im.shape == (100,1000,3)


    im = IU.get_im_cv2('test_data/test100x1000.jpg',upscale=False,downscale=True,size=(100,100))
    assert im.shape == (100,100,3)


