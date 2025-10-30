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
from mpl_toolkits.axes_grid1 import ImageGrid
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
import healpy as healpy

import general_purpose_libs as gpl


def prepare_data(input_data, input_label, dataset_parameters, output_folder):
    # Load the data
    remove_y_direction = dataset_parameters["remove_y_direction"]
    train_fraction = dataset_parameters["train_fraction"]
    val_fraction = dataset_parameters["val_fraction"]
    test_fraction = dataset_parameters["test_fraction"]
    aug_coefficient = dataset_parameters["aug_coefficient"]
    prob_per_flip = dataset_parameters["prob_per_flip"]

    if not os.path.exists(input_data) or not os.path.exists(input_label):
        print(input_data)
        print(input_label)
        print("Exists input data: ", os.path.exists(input_data))
        print("Exists input label: ", os.path.exists(input_label))
        print("Input file not found.")
        exit()

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load the dataset
    print("Loading the dataset...")
    dataset_img = np.load(input_data, allow_pickle=True)
    dataset_label = np.load(input_label)
    print("Dataset loaded.")
    print("Dataset_img shape: ", dataset_img.shape)
    print("Dataset_lab shape: ", dataset_label.shape)
    print("dataset_img type: ", type(dataset_img))
    print("dataset_lab type: ", type(dataset_label))


    # Remove the direction
    if remove_y_direction:
        print("Removing the direction...")
        dataset_label = np.delete(dataset_label, 1, axis=1)
    
    # Normalize the labels
    print("Normalizing the labels...")
    if dataset_label.shape[1] == 3:
        dataset_label = dataset_label/np.linalg.norm(dataset_label, axis=1)[:, np.newaxis]
    elif dataset_label.shape[1] == 2:
        r = np.sqrt(dataset_label[:,0]**2 + dataset_label[:,1]**2)
        dataset_label[:,0] = dataset_label[:,0]/r
        dataset_label[:,1] = dataset_label[:,1]/r
    print("Labels normalized.")


    if not os.path.exists(output_folder+"samples/"):
        os.makedirs(output_folder+"samples/")
    
    for i in range(dataset_label.shape[1]):
        plt.hist(dataset_label[:,i], bins=100, label='dir '+str(i))
    plt.legend()
    plt.savefig(output_folder+"samples/label_hist.png")

    # Check if the dimension of images and labels are the same
    if dataset_img.shape[0] != dataset_label.shape[0]:
        print("Error: the dimension of images and labels are not the same.")
        exit()
    # shuffle the dataset
    print("Shuffling the dataset...")
    index = np.arange(dataset_img.shape[0])
    np.random.shuffle(index)
    dataset_img = dataset_img[index]
    dataset_label = dataset_label[index]
    print("Dataset shuffled.")

    # Save some images
    print("Saving some images...")
    save_samples_from_ds(dataset_img, dataset_label, output_folder+"samples/", name="img", n_samples=10)
    save_labels_in_a_map(dataset_label, output_folder+"samples/", name="full_ds")
    print("Images saved.")
    
    # Split the dataset in training, validation and test
    print("Splitting the dataset...")
    training_fraction = train_fraction
    validation_fraction = val_fraction
    test_fraction = test_fraction

    train_images = dataset_img[:int(dataset_img.shape[0]*training_fraction)]
    validation_images = dataset_img[int(dataset_img.shape[0]*training_fraction):int(dataset_img.shape[0]*training_fraction)+int(dataset_img.shape[0]*validation_fraction)]
    test_images = dataset_img[int(dataset_img.shape[0]*training_fraction)+int(dataset_img.shape[0]*validation_fraction):]
    train_labels = dataset_label[:int(dataset_label.shape[0]*training_fraction)]
    validation_labels = dataset_label[int(dataset_label.shape[0]*training_fraction):int(dataset_label.shape[0]*training_fraction)+int(dataset_label.shape[0]*validation_fraction)]
    test_labels = dataset_label[int(dataset_label.shape[0]*training_fraction)+int(dataset_label.shape[0]*validation_fraction):]
    # save test set for further analysis
    np.save(output_folder+"test_images.npy", test_images)
    np.save(output_folder+"test_labels.npy", test_labels)
    print("Dataset splitted.")

    if aug_coefficient>1:
        print("Data augmentation...")
        print("Train images shape before: ", train_images.shape)
        train_images, train_labels = data_augmentation(train_images, train_labels, coefficient=aug_coefficient, prob_per_flip=prob_per_flip)
        print("Train images shape after: ", train_images.shape)
        print("Data augmented.")

    # train = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(10000).batch(32)
    # validation = tf.data.Dataset.from_tensor_slices((validation_images, validation_labels)).batch(32)
    # test = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(32)
    train = tf.data.Dataset.from_tensor_slices(((train_images[:, :, :, 0], train_images[:, :, :, 1], train_images[:, :, :, 2]), train_labels)).shuffle(10000).batch(32)
    validation = tf.data.Dataset.from_tensor_slices(((validation_images[:, :, :, 0], validation_images[:, :, :, 1], validation_images[:, :, :, 2]), validation_labels)).batch(32)
    test = tf.data.Dataset.from_tensor_slices(((test_images[:, :, :, 0], test_images[:, :, :, 1], test_images[:, :, :, 2]), test_labels)).batch(32)

    return train, validation, test

def test_model(model, test, output_folder):
    # Test the model
    print("Doing some test...")
    predictions = model.predict(test)      
    # Calculate metrics
    print("Calculating metrics...")
    # get the test labels from the test dataset
    test_labels = np.array([label for _, label in test], dtype=object)
    test_labels = np.concatenate(test_labels, axis=0)
    log_metrics(test_labels, predictions, output_folder=output_folder)
    print("Metrics calculated.")
    print("Drawing model...")
    keras.utils.plot_model(model, output_folder+"architecture.png", show_shapes=True)
    print("Model drawn.")
    print("Test done.")

def log_metrics(test_labels, predictions, output_folder):
    save_labels_in_a_map(test_labels, output_folder, name="map_true")
    save_labels_in_a_map(predictions, output_folder, name="map_predictions")
    plot_diff(test_labels, predictions, output_folder)

def plot_diff(test_labels, predictions, output_folder, final_theta=None, final_phi=None, final_theta_std=None, final_phi_std=None):
    pred_x, pred_y, pred_z = predictions[:, 0], predictions[:, 1], predictions[:, 2]
    true_x, true_y, true_z = test_labels[:, 0], test_labels[:, 1], test_labels[:, 2]

    # norm predictions
    r = np.sqrt(pred_x**2 + pred_z**2 + pred_y**2)
    pred_x = pred_x/r
    pred_y = pred_y/r
    pred_z = pred_z/r

    true_angles = from_coordinate_to_theta_phi(test_labels)
    true_theta, true_phi = true_angles[:, 0], true_angles[:, 1]
    pred_angles = from_coordinate_to_theta_phi(predictions) 
    pred_theta, pred_phi = pred_angles[:, 0], pred_angles[:, 1]
   
    plt.hist(pred_theta, bins=50, alpha=0.5, label="Predicted", range=(-np.pi, np.pi))
    unique_true_theta = np.unique(true_theta)
    if len(unique_true_theta) > 1:
        plt.hist(true_theta, bins=50, alpha=0.5, label="True", range=(-np.pi, np.pi))
    else:
        plt.axvline(unique_true_theta[0], color='b', linestyle='solid', linewidth=2, label="True Theta")

    if final_theta is not None:
        # add a line with the final theta 
        plt.axvline(final_theta, color='r', linestyle='dashed', linewidth=2, label="Final Theta")
        upper_bound = (final_theta + final_theta_std + np.pi) % (2 * np.pi) - np.pi
        lower_bound = (final_theta - final_theta_std + np.pi) % (2 * np.pi) - np.pi
        plt.axvline(upper_bound, color='r', linestyle='dotted', linewidth=2, label="Final Theta + Std")
        plt.axvline(lower_bound, color='r', linestyle='dotted', linewidth=2, label="Final Theta - Std")

    plt.title("True and Predicted thetas")
    plt.legend()
    plt.xlabel("Theta [Rad]")
    plt.legend(loc='upper right')
    plt.savefig(output_folder+'thetas.png')
    plt.clf()


    plt.hist(pred_phi, bins=50, alpha=0.5, label="Predicted", range=(0, np.pi))
    unique_true_phi = np.unique(true_phi)
    if len(unique_true_phi) > 1:
        plt.hist(true_phi, bins=50, alpha=0.5, label="True", range=(0, np.pi))
    else:
        plt.axvline(unique_true_phi[0], color='b', linestyle='solid', linewidth=2, label="True Phi")

    if final_phi is not None:
        # add a line with the final phi 
        plt.axvline(final_phi, color='r', linestyle='dashed', linewidth=2, label="Final Phi")
        plt.axvline(np.min([final_phi + final_phi_std, np.pi]), color='r', linestyle='dotted', linewidth=2, label="Final Phi + Std")
        plt.axvline(np.max([final_phi - final_phi_std, 0]), color='r', linestyle='dotted', linewidth=2, label="Final Phi - Std")       

    plt.title("True and Predicted phis")
    plt.legend()
    plt.xlabel("Phi [Rad]")
    plt.legend(loc='upper right')
    plt.savefig(output_folder+'phis.png')
    plt.clf()

    plt.hist(true_x, bins=50, alpha=0.5, label="True")
    plt.hist(pred_x, bins=50, alpha=0.5, label="Predicted")
    plt.title("X")
    plt.legend(loc='upper right')
    plt.savefig(output_folder+'x.png')
    plt.clf()

    plt.hist(true_y, bins=50, alpha=0.5, label="True")
    plt.hist(pred_y, bins=50, alpha=0.5, label="Predicted")
    plt.title("Y")
    plt.legend(loc='upper right')
    plt.savefig(output_folder+'y.png')
    plt.clf()

    plt.hist(true_z, bins=50, alpha=0.5, label="True")
    plt.hist(pred_z, bins=50, alpha=0.5, label="Predicted")
    plt.title("Z")
    plt.legend(loc='upper right')
    plt.savefig(output_folder+'z.png')
    plt.clf()

    # flat distribution
    theta_flat = np.random.uniform(-np.pi, np.pi, len(true_x))
    cosines_flat = np.cos(theta_flat)
    cosines = (true_x * pred_x + true_z * pred_z) / (np.sqrt(true_x**2 + true_z**2) * np.sqrt(pred_x**2 + pred_z**2))

    plt.hist(cosines, bins=50, alpha=1, label="Cos($\\theta_{true}-\\theta_{pred}$)")
    plt.hist(cosines_flat, bins=50, alpha=0.2, label="Cos($\\theta_{true}-\\theta_{flat}$)", hatch='xx')
    plt.xlabel("Cos($\\theta$)")
    plt.title("Cosine between true and predicted direction")
    plt.legend()
    plt.savefig(output_folder+'cosine.png')
    plt.clf()

    # plot the difference
    mins = np.minimum(true_theta, pred_theta)
    maxs = np.maximum(true_theta, pred_theta)
    true_diff = maxs - mins
    true_diff = np.where(true_diff > np.pi, 2*np.pi - true_diff, true_diff)

    mins_flat = np.minimum(true_theta, theta_flat)
    maxs_flat = np.maximum(true_theta, theta_flat)
    flat_diff = maxs_flat - mins_flat
    flat_diff = np.where(flat_diff > np.pi, 2*np.pi - flat_diff, flat_diff)

    plt.hist(true_diff, bins=50, alpha=1, label="$\\theta_{true}-\\theta_{pred}$")
    plt.hist(flat_diff, bins=50, alpha=0.2, label="$\\theta_{true}-\\theta_{flat}$", hatch='xx')
    plt.xlabel("Angle Difference [Rad]")
    plt.title("Angle difference between true and predicted direction")
    plt.legend()
    plt.savefig(output_folder+'difference.png')
    plt.clf()

    plt.hist(true_theta-pred_theta, bins=50, alpha=0.5, label="True")
    plt.hist(true_theta-theta_flat, bins=50, alpha=0.5, label="Flat")
    plt.xlabel("True - Predicted")
    plt.legend()
    plt.savefig(output_folder+'difference_not_correct.png')
    plt.clf()

    # include also the y coordinate in the angle studies
    flat_omega = np.random.uniform(0, np.pi, len(true_x))
    complete_cos_omega = (true_x * pred_x + true_y * pred_y + true_z * pred_z) / (np.sqrt(true_x**2 + true_y**2 + true_z**2) * np.sqrt(pred_x**2 + pred_y**2 + pred_z**2))
    complete_omega = np.arccos(complete_cos_omega)
    plt.hist(complete_cos_omega, bins=50, alpha=1, label="Cos($\\omega$)")
    plt.hist(np.cos(flat_omega), bins=50, alpha=0.2, label="Cos($\\omega_{flat}$)", hatch='xx')
    plt.xlabel("Cos($\\omega$)")
    plt.title("Cosine between true and predicted direction in the 3D space")
    plt.legend()
    plt.savefig(output_folder+'cosine_3D.png')
    plt.clf()

    plt.hist(complete_omega, bins=50, alpha=1, label="$\\omega$")
    plt.hist(flat_omega, bins=50, alpha=0.2, label="$\\omega_{flat}$", hatch='xx')
    plt.xlabel("$\\omega$")
    plt.title("Angle between true and predicted direction in the 3D space")
    plt.legend()
    plt.savefig(output_folder+'angle_3D.png')
    plt.clf()





def save_labels_in_a_map(dataset_label, output_folder, name="map"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    nside = 64
    npix = healpy.nside2npix(nside)
    # create a map with the number of pixels
    map_hp = np.zeros(npix)
    angles = from_coordinate_to_theta_phi(dataset_label)
    thetas, phis = angles[:,0], angles[:,1]

    plt.figure(figsize=(10, 10))
    plt.title("Labels map")
    thetas = np.mod(thetas, 2*np.pi)
    phis = np.mod(phis, np.pi)
    # get the indices
    indices = healpy.ang2pix(nside, phis, thetas)
    # fill the map
    for index in indices:
        map_hp[index] += 1

    map_hp = healpy.smoothing(map_hp, fwhm=0.1)
    healpy.mollview(map_hp, title="Predictions", cmap="viridis")
    healpy.graticule()
    plt.savefig(output_folder+name+".png")
    plt.close()

def save_samples_from_ds(dataset, labels, output_folder, name="img", n_samples=10):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # get the labels
    fig= plt.figure(figsize=(20, 20))
    for i in range(n_samples):
        # get the sample
        sample = dataset[i]
        # get the label
        label = labels[i]
        # save the sample
        gpl.save_sample_img(sample, output_folder, name+"_"+str(i))
        plt.clf()
    plt.close()
            
def from_coordinate_to_theta_phi(coords):
    # nomalize the coordinates
    coords = coords/np.linalg.norm(coords, axis=1)[:, np.newaxis]
    x, y, z = coords[:,0], coords[:,1], coords[:,2]
    r = np.sqrt(x**2 + y**2 + z**2)

    phi = np.arccos(y/r)
    theta = np.arctan2(z, x)

    return np.array([theta, phi]).T

def data_augmentation(dataset, labels, coefficient=2, prob_per_flip=0.5):
    # create the augmented dataset
    augmented_dataset = []
    augmented_labels = []
    # for each sample
    if (labels.shape[1]) == 3:
        for i in range(int(coefficient*len(dataset))):
            index = np.random.randint(0, len(dataset))
            # get the sample
            sample = dataset[index]
            # get the label
            label = labels[index]
            new_label = [label[0], label[1], label[2]]
            if np.random.rand() > prob_per_flip:
                # flip the sample
                sample = np.flipud(sample)
                new_label[0] = -label[0]
            if np.random.rand() > prob_per_flip:
                sample = np.fliplr(sample)
                new_label[2] = -label[2]
            augmented_dataset.append(sample)
            augmented_labels.append(new_label)
    elif (labels.shape[1]) == 2:
        for i in range(int(coefficient*len(dataset))):
            index = np.random.randint(0, len(dataset))
            # get the sample
            sample = dataset[index]
            # get the label
            label = labels[index]
            new_label = [label[0], label[1]]
            if np.random.rand() > prob_per_flip:
                # flip the sample
                sample = np.flipud(sample)
                new_label[0] = -label[0]
            if np.random.rand() > prob_per_flip:
                sample = np.fliplr(sample)
                new_label[1] = -label[1]
            augmented_dataset.append(sample)
            augmented_labels.append(new_label)

    return np.array(augmented_dataset), np.array(augmented_labels)

def my_loss_function(y_true, y_pred):
    # the loss is the dot product of the true and predicted values, divided by the product of the norms of the two vectors
    return 1 - tf.reduce_sum(y_true * y_pred, axis=-1) / (tf.norm(y_true, axis=-1) * tf.norm(y_pred, axis=-1))

def my_loss_function_both_dir(y_true, y_pred):
    # the loss accepts two directions, the true and the predicted one
    return tf.reduce_sum(tf.minimum(1 - tf.reduce_sum(y_true * y_pred, axis=-1) / (tf.norm(y_true, axis=-1) * tf.norm(y_pred, axis=-1)), 1 + tf.reduce_sum(y_true * y_pred, axis=-1) / (tf.norm(y_true, axis=-1) * tf.norm(y_pred, axis=-1))), axis=-1)


def prepare_data_from_npz_regression(data_dir, plane, dataset_parameters, output_folder):
    """
    Prepare data from NPZ batch files for regression (electron direction).
    Similar to classification version but extracts direction vectors instead of binary labels.
    
    Args:
        data_dir: Directory containing NPZ batch files
        plane: Plane to use ('U', 'V', or 'X')
        dataset_parameters: Dictionary with dataset parameters
        output_folder: Output folder for saving samples and plots
        
    Returns:
        train, validation, test: Tuples of (images, direction_labels)
    """
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import data_loader as dl
    import numpy as np
    
    train_fraction = dataset_parameters.get("train_fraction", 0.8)
    val_fraction = dataset_parameters.get("val_fraction", 0.1)
    test_fraction = dataset_parameters.get("test_fraction", 0.1)
    max_samples = dataset_parameters.get("max_samples", None)
    
    print("\n" + "="*60)
    print("LOADING DATA FROM NPZ FILES (REGRESSION)")
    print("="*60)
    print(f"Data directory: {data_dir}")
    print(f"Plane: {plane}")
    print(f"Max samples: {max_samples if max_samples else 'All'}")
    
    # Create output folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Load dataset from NPZ files
    dataset_img, metadata = dl.load_dataset_from_directory(
        data_dir=data_dir,
        plane=plane,
        max_samples=max_samples,
        verbose=True
    )
    
    # Extract direction labels (columns 3-5: x, y, z)
    dataset_label = dl.extract_direction_labels(metadata)
    
    print(f"Dataset shape: {dataset_img.shape}")
    print(f"Direction labels shape: {dataset_label.shape}")
    print(f"Direction labels range: [{dataset_label.min():.3f}, {dataset_label.max():.3f}]")
    
    # Add channel dimension
    if len(dataset_img.shape) == 3:
        dataset_img = np.expand_dims(dataset_img, axis=-1)
    
    # Split dataset
    n_samples = len(dataset_img)
    n_train = int(n_samples * train_fraction)
    n_val = int(n_samples * val_fraction)
    
    # Shuffle
    indices = np.random.permutation(n_samples)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train+n_val]
    test_idx = indices[n_train+n_val:]
    
    train = (dataset_img[train_idx], dataset_label[train_idx])
    validation = (dataset_img[val_idx], dataset_label[val_idx])
    test = (dataset_img[test_idx], dataset_label[test_idx])
    
    print(f"\nData split:")
    print(f"  Train: {len(train[0])} samples, labels shape: {train[1].shape}")
    print(f"  Val: {len(validation[0])} samples, labels shape: {validation[1].shape}")
    print(f"  Test: {len(test[0])} samples, labels shape: {test[1].shape}")
    
    return train, validation, test
