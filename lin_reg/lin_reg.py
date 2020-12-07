import tensorflow as tf
import keras
from keras.layers import Input, Conv2D, Flatten, Dense
from keras.layers import Reshape, Concatenate, Add
from keras.models import Model
import random
import keras.layers as layers
import sherpa
#import sherpa
import sys

import numpy as np
import os
import matplotlib.pyplot as plt
#from ssim_loss import *
from keras.initializers import Initializer, Constant
import keras.backend as K



#Amin Cuda settings
#os.environ["CUDA_VISIBLE_DEVICES"]="2"
#Alex Cuda settings 
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

class AlphaInit(Initializer):
    def __init__(self, alpha, **kwargs):
        super(AlphaInit, self).__init__(**kwargs)
        self.constant = alpha
    def __call__(self):
        return K.constant(self.constant)
class Mult(layers.Layer):
    def __init__(self, init, **kwargs):
        super(Mult, self).__init__(**kwargs)
        self.init = init
    def build(self, input_shape):
        output_dim = input_shape
        if self.init == "constant":
            self.kernel = self.add_weight(
                shape=[1],
                initializer=Constant(0.5),
                name="kernel",
                trainable=True,
            )
        else:
            self.kernel = self.add_weight(
                shape=[1],
                initializer=keras.initializers.get("he_normal"),
                name="kernel",
                trainable=True,
            )
    def call(self, inputs):
        out = inputs*self.kernel
        return out
    def get_config(self):
        # Implement get_config to enable serialization. This is optional.
        config = super(Mult, self).get_config()
        config.update({"init": self.init})
        return dict(list(config.items()))
def build_model(ssim=False):
    """
    Build Keras model with different hyperparameters
    """
    
    input_prev = Input(shape=(48, 48, 1))
    input_next = Input(shape=(48, 48, 1))
    x_prev = Mult("he")(input_prev)
    x_next = Mult("he")(input_next)  
    h = Add()([x_prev, x_next])
    
    model = Model([input_prev, input_next], h)
    # add loss function
    if ssim:
        model.compile(optimizer='adam', loss=ssim_loss, metrics=[ssim_loss, 'mse'])
    else:
        model.compile(optimizer='adam', loss='mse')
    return model
    
def train(model, x_train, y_train, model_name):
    #model.fit(x_train, y_train, epochs=100, batch_size=64, validation_data=(x_val, y_val))
    x_prev = x_train[:,:,:, 0]  
    x_next = x_train[:,:,:, 1]
    X_SHAPE = (*x_prev.shape,1)
    x_prev = x_prev.reshape(X_SHAPE)
    x_next = x_next.reshape(X_SHAPE)
    print(f'x_next shape: {x_next.shape}, x_prev shape: {x_prev.shape}')
    print(f'y_shape: {y_train.shape}')
    history = model.fit([x_prev, x_next], y_train, epochs=25, batch_size=64)
    model.save(model_name)

if __name__=="__main__":
    import h5py
    # we can do grid search here
    
  
    


    train_file = sys.argv[1]

    model_name = train_file[:-5]+"_lin_reg.h5"

    f  = h5py.File(train_file, 'r')
    x_train = f['x_train']
    x_train = np.array(x_train)
    print(x_train.shape)
    shape_train = x_train.shape[0]
    shape_train = int(shape_train/64.)*64 # ssim needs to have the same batch size for all the mini batches
    x_train = x_train[:shape_train,:,:,:]
    print(x_train.shape)
    y_train = f['y_train']
    y_train = np.array(y_train)
    y_train = y_train[:shape_train,:,:]
    Y_SHAPE = (*y_train.shape, 1)
    y_train = y_train.reshape(Y_SHAPE)
    
    
    model = build_model()
    train(model, x_train, y_train, model_name)
    
