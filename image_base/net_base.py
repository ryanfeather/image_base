import logging
import abc
from keras.applications.imagenet_utils import preprocess_input as preprocess_center

from keras.applications.inception_v3 import  InceptionV3, preprocess_input as preprocess_scale
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
import numpy as np
from keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint
from keras.models import load_model

import sys
sys.setrecursionlimit(20000) # needed for compiling/saving some keras models

def make_keras_pretrained(inputs, model_class, weights='imagenet', lock=True):

    model = model_class(include_top=False, weights=weights,input_tensor=inputs)

    if lock:
        for layer in model.layers:
            layer.trainable = False

    logging.info('{0} output shape {1:}'.format(model_class, model.output_shape))
    return model.output

class Architecture():

    def __init__(self, inputs, make_func, preprocessing=preprocess_center):

        self.model = make_func(inputs)
        self.preprocessor = preprocessing

class Pretrained(Architecture):

    def __init__(self,inputs, model_class, weights='imagenet', lock=True, preprocessing=preprocess_center):
        def make_func(inputs_in):
            return  make_keras_pretrained(inputs_in, model_class, weights=weights, lock=lock)
        super(Pretrained, self).__init_(inputs, make_func, preprocessing=preprocessing)

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
