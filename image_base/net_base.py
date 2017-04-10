import logging
import abc
from keras.applications.imagenet_utils import preprocess_input as preprocess_center

from keras.applications.inception_v3 import  InceptionV3, preprocess_input as preprocess_scale
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
import numpy as np
from keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint
from keras.models import load_model, Model
from keras.layers import Conv2D, MaxPooling2D

import sys
sys.setrecursionlimit(20000) # needed for compiling/saving some keras models

def make_keras_pretrained(inputs, model_class, weights='imagenet', lock=True):

    model = model_class(include_top=False, weights=weights,input_tensor=inputs)

    if lock:
        for layer in model.layers:
            layer.trainable = False

    logging.info('{0} output shape {1:}'.format(model_class, model.output_shape))
    return model.output

def make_small_vgg19(inputs, dim1=64):
    """ Make a configurable top vgg-19 like architecture

    :param inputs:
    :param dim1:
    :return:
    """
    x = Conv2D(dim1, (3, 3), activation='relu', padding='same', name='block1_conv1')(inputs)
    x = Conv2D(dim1, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(dim1*2, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(dim1*2, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(dim1*4, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(dim1*4, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(dim1*4, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = Conv2D(dim1*4, (3, 3), activation='relu', padding='same', name='block3_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(dim1*8, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(dim1*8, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(dim1*8, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = Conv2D(dim1*8, (3, 3), activation='relu', padding='same', name='block4_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(dim1*8, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(dim1*8, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(dim1*8, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = Conv2D(dim1*8, (3, 3), activation='relu', padding='same', name='block5_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    model = Model(inputs, x, name='vgg19')
    return model

class Architecture():

    def __init__(self, inputs, make_func, preprocessing=preprocess_center):

        self.model = make_func(inputs)
        self.preprocessor = preprocessing

class Pretrained(Architecture):

    def __init__(self,inputs, model_class, weights='imagenet', lock=True, preprocessing=preprocess_center):
        def make_func(inputs_in):
            return  make_keras_pretrained(inputs_in, model_class, weights=weights, lock=lock)
        super(Pretrained, self).__init__(inputs, make_func, preprocessing=preprocessing)

class PreInceptionV3(Pretrained):
    def __init__(self, inputs, weights='imagenet', lock=True):
        super(PreInceptionV3, self).__init__(inputs, InceptionV3, weights=weights, lock=lock, preprocessing=preprocess_scale)

class PreResNet50(Pretrained):
    def __init__(self, inputs, weights='imagenet', lock=True):
        super(PreResNet50, self).__init__(inputs, ResNet50, weights=weights, lock=lock)

class PreVGG19(Pretrained):
    def __init__(self, inputs, weights='imagenet', lock=True):
        super(PreVGG19,self).__init__(inputs, VGG19, weights=weights, lock=lock)

def make_trainable(model):
    for layer in model.layers:
        layer.trainable = True

def fit_model(X_train, X_valid, Y_train, Y_valid, batch_size, model, nb_epoch, num_fold, patience,
              save_pattern, schedule, datagen=None, model_name=''):
    save_file = save_pattern.format(iter=num_fold, mod=model_name)
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=patience, verbose=0),
        LearningRateScheduler(schedule),
        ModelCheckpoint(filepath=save_file, monitor='val_loss', verbose=0, save_best_only=True)
    ]
    if datagen is None:
        model.fit(X_train, Y_train, batch_size=batch_size,
                  nb_epoch=nb_epoch,
                  shuffle=True, verbose=2, validation_data=(X_valid, Y_valid),
                  callbacks=callbacks)
    if datagen is not None:
        model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size, shuffle=True),
                                nb_epoch=nb_epoch, samples_per_epoch=len(X_train), verbose=2,
                                validation_data=(X_valid, Y_valid),
                                callbacks=callbacks)

    model = load_model(save_file)
    return model, save_file
