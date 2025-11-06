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
    
    # model = tf.keras.Sequential()
    input_layer_u = layers.Input(shape=input_shape)
    input_layer_v = layers.Input(shape=input_shape)
    input_layer_x = layers.Input(shape=input_shape)

    x_u = layers.Conv2D(optimizable_parameters['n_filters'], (optimizable_parameters['kernel_size'], 1))(input_layer_u)
    x_u = layers.LeakyReLU(alpha=0.05)(x_u)
    x_u = layers.MaxPooling2D((2, 2))(x_u)

    x_v = layers.Conv2D(optimizable_parameters['n_filters'], (optimizable_parameters['kernel_size'], 1))(input_layer_v)
    x_v = layers.LeakyReLU(alpha=0.05)(x_v)
    x_v = layers.MaxPooling2D((2, 2))(x_v)

    x_x = layers.Conv2D(optimizable_parameters['n_filters'], (optimizable_parameters['kernel_size'], 1))(input_layer_x)
    x_x = layers.LeakyReLU(alpha=0.05)(x_x)
    x_x = layers.MaxPooling2D((2, 2))(x_x)

    # Convolutional layers
    for i in range(optimizable_parameters['n_conv_layers']):
        x_u = layers.Conv2D(optimizable_parameters['n_filters']//(i+1), (optimizable_parameters['kernel_size'], 1))(x_u)
        x_u = layers.LeakyReLU(alpha=0.05)(x_u)
        x_u = layers.MaxPooling2D((2, 2))(x_u)

        x_v = layers.Conv2D(optimizable_parameters['n_filters']//(i+1), (optimizable_parameters['kernel_size'], 1))(x_v)
        x_v = layers.LeakyReLU(alpha=0.05)(x_v)
        x_v = layers.MaxPooling2D((2, 2))(x_v)

        x_x = layers.Conv2D(optimizable_parameters['n_filters']//(i+1), (optimizable_parameters['kernel_size'], 1))(x_x)
        x_x = layers.LeakyReLU(alpha=0.05)(x_x)
        x_x = layers.MaxPooling2D((2, 2))(x_x)

    # Flatten
    x_u = layers.Flatten()(x_u)
    x_v = layers.Flatten()(x_v)
    x_x = layers.Flatten()(x_x)

    x = layers.concatenate([x_u, x_v, x_x])

    # Dense layers
    for i in range(optimizable_parameters['n_dense_layers']):
        x = layers.Dense(optimizable_parameters['n_dense_units']//(i+1), activation='relu')(x)
        x = layers.LeakyReLU(alpha=0.05)(x)

        output_layer = layers.Dense(n_outputs, activation='linear')(x)
        # normalize the output
        # output_layer = layers.Lambda(lambda x: x / tf.norm(x, axis=1, keepdims=True))(output_layer)

    model = keras.Model(inputs=[input_layer_u, input_layer_v, input_layer_x], outputs=output_layer)
    
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

    # print(train.shape)

    # the input shape is [200, 50, 3]. Unpack the input shape to get the number of channels
    history = model.fit(train,
                        epochs=200, 
                        validation_data=validation, 
                        callbacks=callbacks,
                        verbose=1)

    return model, history



def create_and_train_model(n_outputs, model_parameters, train, validation, output_folder, model_name):
    space_options = model_parameters['space_options']
    if model_parameters['loss_function'] == 'my_loss_function':
        loss_function = rl.my_loss_function
    elif model_parameters['loss_function'] == 'my_loss_function_both_dir':
        loss_function = rl.my_loss_function_both_dir
    else:
        loss_function = model_parameters['loss_function']

    input_shape = model_parameters['input_shape']
    hp_max_evals = model_parameters['hp_max_evals']

    space = {
        'n_conv_layers': hp.hp.choice('n_conv_layers', space_options['n_conv_layers']),
        'n_dense_layers': hp.hp.choice('n_dense_layers', space_options['n_dense_layers']),
        'n_filters': hp.hp.choice('n_filters', space_options['n_filters']),
        'kernel_size': hp.hp.choice('kernel_size', space_options['kernel_size']),
        'n_dense_units': hp.hp.choice('n_dense_units', space_options['n_dense_units']),
        'learning_rate': hp.hp.uniform('learning_rate', space_options['learning_rate'][0], space_options['learning_rate'][1]),
        'decay_rate': hp.hp.uniform('decay_rate', space_options['decay_rate'][0], space_options['decay_rate'][1]),
    }
    trials = hp.Trials()
    # Run the hyperparameter search
    print("Running the hyperparameter search...")
    best = hp.fmin(
        fn=lambda x: hypertest_model(  optimizable_parameters=x, 
                                        train=train, 
                                        validation=validation, 
                                        n_outputs=n_outputs, 
                                        output_folder=output_folder,
                                        input_shape=input_shape,
                                        loss_function=loss_function
                                        ),
        space=space,
        algo=hp.tpe.suggest,
        max_evals=hp_max_evals,
        trials=trials)
    print("Hyperparameter search completed.")

    best_dict = {
        'n_conv_layers': space_options['n_conv_layers'][best['n_conv_layers']],
        'n_dense_layers': space_options['n_dense_layers'][best['n_dense_layers']],
        'n_filters': space_options['n_filters'][best['n_filters']],
        'kernel_size': space_options['kernel_size'][best['kernel_size']],
        'n_dense_units': space_options['n_dense_units'][best['n_dense_units']],
        'learning_rate': best['learning_rate'],
        'decay_rate': best['decay_rate'],
    }

    print("Best parameters: ", best_dict)
    print("Best loss: ", -trials.best_trial['result']['loss'])

    print("Best parameters saved.")
    # Save the trials
    print("Saving the trials...")
    np.save(output_folder+model_name+"_trials.npy", trials)
    print("Trials saved.")
    # Save the best parameters
    print("Saving the best parameters...")
    np.save(output_folder+model_name+"_best.npy", best)
    print("Best parameters saved.")
    # Save the best model
    print("Saving the best model...")
    model, history = build_model( n_outputs=n_outputs, 
                                    optimizable_parameters=best_dict, 
                                    train=train, 
                                    validation=validation,
                                    output_folder=output_folder, 
                                    input_shape=input_shape,
                                    loss_function=loss_function
                                    )
    print("Model built.")
    print("Plotting the hyperparameter search...")
    plot_hyperparameter_search(trials, output_folder, model_name)
    print("Plot saved.")
    return model, history

def plot_hyperparameter_search(trials, output_folder, model_name):
    trials = [t for t in trials.trials if t['result']['loss'] != 9999]    
    plt.figure(figsize=(15, 15))
    plt.suptitle('Hyperparameters tuning')
    plt.subplot(3, 3, 1)
    plt.title('Loss vs n_conv_layers')
    plt.scatter([t['misc']['vals']['n_conv_layers'] for t in trials], [-t['result']['loss'] for t in trials], s=20, alpha=0.3, label='loss')
    plt.xlabel('n_conv_layers')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(3, 3, 2)
    plt.title('Loss vs n_dense_layers')
    plt.scatter([t['misc']['vals']['n_dense_layers'] for t in trials], [-t['result']['loss'] for t in trials], s=20, alpha=0.3, label='loss')
    plt.xlabel('n_dense_layers')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(3, 3, 3)
    plt.title('Loss vs n_filters')
    plt.scatter([t['misc']['vals']['n_filters'] for t in trials], [-t['result']['loss'] for t in trials], s=20, alpha=0.3, label='loss')
    plt.xlabel('n_filters')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(3, 3, 4)
    plt.title('Loss vs kernel_size')
    plt.scatter([t['misc']['vals']['kernel_size'] for t in trials], [-t['result']['loss'] for t in trials], s=20, alpha=0.3, label='loss')
    plt.xlabel('kernel_size')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(3, 3, 5)
    plt.title('Loss vs n_dense_units')
    plt.scatter([t['misc']['vals']['n_dense_units'] for t in trials], [-t['result']['loss'] for t in trials], s=20, alpha=0.3, label='loss')
    plt.xlabel('n_dense_units')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(3, 3, 6)
    plt.title('Loss vs learning_rate')
    plt.scatter([t['misc']['vals']['learning_rate'] for t in trials], [-t['result']['loss'] for t in trials], s=20, alpha=0.3, label='loss')
    plt.xlabel('learning_rate')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(3, 3, 7)
    plt.title('Loss vs decay_rate')
    plt.scatter([t['misc']['vals']['decay_rate'] for t in trials], [-t['result']['loss'] for t in trials], s=20, alpha=0.3, label='loss')
    plt.xlabel('decay_rate')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(3, 3, 9)
    plt.title('Loss vs Trial ID')
    plt.scatter([t['tid'] for t in trials], [-t['result']['loss'] for t in trials], s=20, alpha=0.3, label='loss')
    plt.xlabel('Trial ID')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(output_folder+model_name+"_hyperopt_evolution.png")
    plt.clf()
    plt.close()
    # fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    # sns.heatmap([t['result']['loss'] for t in trials], ax=ax[0], annot=True)
    # sns.heatmap([t['result']['accuracy'] for t in trials], ax=ax[1], annot=True)
    # ax[0].set_title('Loss')
    # ax[1].set_title('Loss')
    # fig.suptitle('Hyperparameter search')
    # plt.savefig(output_folder + model_name + '_hyperparameter_search.png')
    # plt.close()

def hypertest_model(optimizable_parameters, train, validation, n_outputs, output_folder, input_shape, loss_function='mean_squared_error'):
    is_comp=is_compatible(optimizable_parameters, input_shape)
    if not is_comp:
        return {'loss': 9999, 'status': hp.STATUS_FAIL}
    else:
        print(is_comp)
    with open(output_folder+"hyperopt_progression.txt", "a") as f:
        f.write(str(optimizable_parameters)+"\n")

    model, history = build_model(n_outputs=n_outputs, 
                                optimizable_parameters=optimizable_parameters, 
                                train=train, 
                                validation=validation, 
                                output_folder=output_folder, 
                                input_shape=input_shape, 
                                loss_function=loss_function)
    loss, accuracy=model.evaluate(validation)
    print("loss: ", loss, " accuracy: ", accuracy)
    with open(output_folder+"hyperopt_progression.txt", "a") as f:
        f.write("loss: "+str(loss)+"\n")
        f.write("accuracy: "+str(accuracy)+"\n")
        f.write("\n")
    return {'loss': loss, 'status': hp.STATUS_OK}

def is_compatible(parameters, input_shape):
    shape = input_shape
    # account for the first conv2d layer
    shape = (shape[0] - parameters['kernel_size'] + 1, shape[1] - 1, shape[2])
    for i in range(parameters['n_conv_layers']):
        # account for the conv2d layer
        shape = (shape[0] - parameters['kernel_size'] + 1, shape[1] - 1, shape[2])
        # account for the maxpooling layer
        shape = (shape[0]//5, shape[1]//2, shape[2])
    
    # account for the flatten layer
    shape = (shape[0]*shape[1]*shape[2],)
    if shape[0] <= 0:
        return False
    else:
        return shape

