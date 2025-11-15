import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
import hyperopt as hp
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

sys.path.append("../../python/")
import general_purpose_libs as gpl
import regression_libs as rl
from models.dune_cvn import *

def create_and_train_model(n_outputs, model_parameters, train, validation, output_folder, model_name):
    input_shape = model_parameters['input_shape']
    if model_parameters['loss_function'] == 'my_loss_function':
        loss_function = rl.my_loss_function
    elif model_parameters['loss_function'] == 'my_loss_function_both_dir':
        loss_function = rl.my_loss_function_both_dir
    else:
        loss_function = model_parameters['loss_function']

    print("Building the model...")
    model = DUNECVNModel(initial_conv_filters=model_parameters['initial_conv_filters'],
                            depth=model_parameters['depth'],
                            filters=model_parameters['filters'],
                            width=model_parameters['width'],
                            weight_decay=model_parameters['weight_decay'],
                            input_shapes=[input_shape],
                            output_neurons=n_outputs)
    # Compile the model
    print("Compiling the model...")
    # add learning ratescheduler
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=model_parameters['learning_rate'],
        decay_steps=10000,
        decay_rate=model_parameters['decay_rate'])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
        loss=loss_function,
        metrics=['accuracy'])   

    # Stop training when `val_loss` is no longer improving
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=6,
            verbose=1)
    ]    

    history = model.fit(train, 
                        epochs=200, 
                        validation_data=validation, 
                        callbacks=callbacks,
                        verbose=1)
    
    return model, history




