import logging
import abc
from keras.applications.imagenet_utils import preprocess_input as preprocess_center

from keras.applications.inception_v3 import  InceptionV3, preprocess_input as preprocess_scale
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50

import sys
sys.setrecursionlimit(20000) # needed for compiling/saving some keras models

def make_keras_pretrained(inputs, model_class, weights='imagenet', lock=True):

    model = model_class(include_top=include_top, weights=weights,input_tensor=inputs)

    if lock:
        for layer in model.layers:
            layer.trainable = False

    logging.info('{0} output shape {1:}'.format(model_class, model.output_shape))
    return model.output

class Pretrained():

    def __init__(self,inputs, model_class, weights='imagenet', lock=True, preprocessing=preprocess_center):
        self.model = make_keras_pretrained(inputs, model_class, weights=weights, lock=lock)
        self.preprocessor = preprocessing

class PreInceptionV3:
    def __init__(self, inputs, weights='imagenet', lock=True):
        super().__init__(inputs, InceptionV3, weights=weights, lock=lock, preprocessing=preprocess_scale)

class PreResNet50:
    def __init__(self, inputs, weights='imagenet', lock=True):
        super().__init__(inputs, ResNet50, weights=weights, lock=lock)

class PreVGG19:
    def __init__(self, inputs, weights='imagenet', lock=True):
        super().__init__(inputs, VGG19, weights=weights, lock=lock)

