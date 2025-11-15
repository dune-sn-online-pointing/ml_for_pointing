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


def build_model(n_outputs, optimizable_parameters, train, validation, output_folder, input_shape, loss_function='mean_squared_error'):
    
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(optimizable_parameters['n_filters'], (optimizable_parameters['kernel_size'], 1), activation='relu', input_shape=input_shape))

    for i in range(optimizable_parameters['n_conv_layers']):
        model.add(layers.Conv2D(optimizable_parameters['n_filters']//(i+1), (optimizable_parameters['kernel_size'], 1), activation='relu'))
        model.add(layers.LeakyReLU(alpha=0.05))
        model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Flatten())
    for i in range(optimizable_parameters['n_dense_layers']):
        model.add(layers.Dense(optimizable_parameters['n_dense_units']//(i+1), activation='relu'))
        model.add(layers.LeakyReLU(alpha=0.05))
    
    model.add(layers.Dense(n_outputs, activation='linear'))  

    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=optimizable_parameters['learning_rate'],
        decay_steps=10000,
        decay_rate=optimizable_parameters['decay_rate'])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
        loss=loss_function,
        metrics=['accuracy'])   

    # Stop training when `val_loss` is no longer improving
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=9,
            verbose=1)
    ]    

    history = model.fit(train, 
                        epochs=200, 
                        validation_data=validation, 
                        callbacks=callbacks,
                        verbose=1)

    return model, history



def create_and_train_model(n_outputs, model_parameters, train, validation, output_folder, model_name):
    if model_parameters['loss_function'] == 'my_loss_function':
        loss_function = rl.my_loss_function
    elif model_parameters['loss_function'] == 'my_loss_function_both_dir':
        loss_function = rl.my_loss_function_both_dir
    else:
        loss_function = model_parameters['loss_function']

    input_shape = model_parameters['input_shape']
    build_parameters = model_parameters['build_parameters'] 
    model, history = build_model(n_outputs, build_parameters, train, validation, output_folder, input_shape, loss_function=loss_function)

    return model, history



