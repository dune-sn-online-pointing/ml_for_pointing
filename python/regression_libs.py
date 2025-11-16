import sys
import os
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.pylab as pylab
import seaborn as sns
try:
    import hyperopt as hp
except ImportError:
    hp = None
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from mpl_toolkits.axes_grid1 import ImageGrid
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
try:
    import healpy as healpy
except ImportError:
    healpy = None

import general_purpose_libs as gpl


def compute_direction_metrics(true_dirs, pred_dirs):
    """Compute angular error statistics between true and predicted directions."""
    if true_dirs.shape != pred_dirs.shape:
        raise ValueError("Prediction and label shapes do not match for metric computation")

    def _normalize(vectors):
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return vectors / norms

    true_unit = _normalize(true_dirs)
    pred_unit = _normalize(pred_dirs)
    cosine = np.clip(np.sum(true_unit * pred_unit, axis=1), -1.0, 1.0)
    angular_errors = np.degrees(np.arccos(cosine))

    metrics = {
        "angular_error_mean": float(np.mean(angular_errors)),
        "angular_error_median": float(np.median(angular_errors)),
        "angular_error_std": float(np.std(angular_errors)),
        "angular_error_25th": float(np.percentile(angular_errors, 25)),
        "angular_error_68th": float(np.percentile(angular_errors, 68)),
        "angular_error_75th": float(np.percentile(angular_errors, 75)),
        "angular_error_90th": float(np.percentile(angular_errors, 90)),
        "angular_error_95th": float(np.percentile(angular_errors, 95)),
        "angular_error_max": float(np.max(angular_errors)),
        "angular_error_min": float(np.min(angular_errors)),
        "cosine_mean": float(np.mean(cosine)),
        "num_samples": int(len(angular_errors))
    }
    return metrics, angular_errors


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
    # Test the model and extract labels
    # Note: We need to extract labels BEFORE model.predict() consumes the dataset
    print("Extracting test labels...")

    output_folder = output_folder if output_folder.endswith(os.sep) else output_folder + os.sep

    def _gather_dataset(dataset):
        data_buffers = None
        labels_buffer = []
        for batch in dataset:
            try:
                data, labels, _ = batch
            except (ValueError, TypeError):
                try:
                    data, labels = batch
                except (ValueError, TypeError):
                    data = batch
                    labels = None
            if isinstance(data, (list, tuple)):
                if data_buffers is None:
                    data_buffers = [[] for _ in range(len(data))]
                for idx, component in enumerate(data):
                    data_buffers[idx].append(component.numpy() if hasattr(component, "numpy") else component)
            else:
                if data_buffers is None:
                    data_buffers = []
                data_buffers.append(data.numpy() if hasattr(data, "numpy") else data)
            if labels is not None:
                labels_buffer.append(labels.numpy() if hasattr(labels, "numpy") else labels)
        if isinstance(data_buffers, list) and data_buffers and isinstance(data_buffers[0], list):
            test_inputs = [np.concatenate(buf, axis=0) for buf in data_buffers]
        else:
            test_inputs = np.concatenate(data_buffers, axis=0)
        test_labels = np.concatenate(labels_buffer, axis=0) if labels_buffer else None
        return test_inputs, test_labels

    if isinstance(test, tuple) and len(test) >= 2:
        test_data = test[0]
        test_labels = test[1]
        print(f"Using tuple format: data shape {test_data.shape if isinstance(test_data, np.ndarray) else 'multi-input'}, labels shape {test_labels.shape}")
    else:
        test_data, test_labels = _gather_dataset(test)

    if test_labels is None:
        raise ValueError("Test labels are required for regression evaluation")

    print("Doing some test...")
    predictions = model.predict(test_data)
    # Calculate metrics
    print("Calculating metrics...")
    metrics, angular_errors = compute_direction_metrics(test_labels, predictions)
    log_metrics(test_labels, predictions, output_folder=output_folder)
    print("Metrics calculated.")
    print("Drawing model...")
    keras.utils.plot_model(model, output_folder+"architecture.png", show_shapes=True)
    print("Model drawn.")

    predictions_path = output_folder + "predictions.npy"
    labels_path = output_folder + "test_labels.npy"
    val_predictions_path = output_folder + "val_predictions.npz"

    np.save(predictions_path, predictions)
    np.save(labels_path, test_labels)
    np.savez(
        val_predictions_path,
        predictions=predictions,
        true_directions=test_labels,
        angular_errors=angular_errors
    )

    print("Test done.")

    artifacts = {
        "predictions": os.path.basename(predictions_path),
        "labels": os.path.basename(labels_path),
        "val_predictions": os.path.basename(val_predictions_path),
        "architecture": "architecture.png"
    }

    return {
        "metrics": metrics,
        "artifacts": artifacts
    }

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
    # thetas and phis are already in correct ranges from conversion function
    # get the indices (healpy.ang2pix expects theta first, then phi)
    indices = healpy.ang2pix(nside, thetas, phis)
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
    # normalize the coordinates
    norms = np.linalg.norm(coords, axis=1, keepdims=True)
    # Avoid division by zero - replace zero norms with 1
    norms = np.where(norms == 0, 1, norms)
    coords = coords / norms
    
    x, y, z = coords[:,0], coords[:,1], coords[:,2]
    
    # Healpy convention: theta = polar angle from z-axis [0, pi], phi = azimuthal angle [0, 2*pi]
    theta = np.arccos(np.clip(z, -1, 1))  # Polar angle from z-axis
    phi = np.arctan2(y, x)  # Azimuthal angle in x-y plane
    phi = np.mod(phi, 2*np.pi)  # Ensure phi is in [0, 2*pi]

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
    
    # CRITICAL: Filter for main tracks only (is_main_track=1 in column 1+offset)
    offset, _ = dl._metadata_layout(metadata.shape[1])
    is_main_track = metadata[:, 1 + offset].astype(bool)
    
    print(f"\nFiltering for main tracks:")
    print(f"  Total samples: {len(metadata)}")
    print(f"  Main tracks: {np.sum(is_main_track)} ({100*np.sum(is_main_track)/len(metadata):.1f}%)")
    print(f"  Background: {np.sum(~is_main_track)} ({100*np.sum(~is_main_track)/len(metadata):.1f}%)")
    
    # Apply filter
    dataset_img = dataset_img[is_main_track]
    metadata = metadata[is_main_track]
    
    print(f"  After filtering: {len(metadata)} main track samples")
    
    # Extract direction labels (columns 6-8+offset: px, py, pz, normalized)
    dataset_label = dl.extract_direction_labels(metadata)
    
    print(f"\nDataset shape: {dataset_img.shape}")
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


def prepare_data_from_npz_3planes(data_dir, dataset_parameters, output_folder):
    """
    Prepare data from NPZ batch files for 3-plane regression (electron direction).
    Loads matched clusters from all three planes (U, V, X) and combines them.
    Only includes samples where ALL THREE planes have a match at the same index within each batch.
    
    Args:
        data_dir: Directory (or list of directories) containing NPZ batch files
        dataset_parameters: Dictionary with dataset parameters
        output_folder: Output folder for saving samples and plots
        
    Returns:
        train, validation, test: Tuples of (dict of plane images, direction_labels)
            where images is a dict: {'U': array, 'V': array, 'X': array}
    """
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import data_loader as dl
    import numpy as np
    import glob
    
    train_fraction = dataset_parameters.get("train_fraction", 0.6)
    val_fraction = dataset_parameters.get("val_fraction", 0.2)
    test_fraction = dataset_parameters.get("test_fraction", 0.2)
    max_samples = dataset_parameters.get("max_samples", None)
    batch_pattern = dataset_parameters.get("batch_pattern", "*_matched_plane{plane}.npz")
    shuffle_data = dataset_parameters.get("shuffle_data", True)
    random_seed = dataset_parameters.get("random_seed", 42)
    
    print("\n" + "="*60)
    print("LOADING DATA FROM NPZ FILES (3-PLANE REGRESSION WITH MATCHING)")
    print("="*60)
    print(f"Data directory: {data_dir}")
    print(f"Batch pattern: {batch_pattern}")
    print(f"Max samples: {max_samples if max_samples else 'All'}")
    
    # Create output folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Handle single directory or list of directories
    data_dirs = [data_dir] if isinstance(data_dir, str) else data_dir
    
    # Collect all matched batch files
    print("\nFinding matched batch files...")
    batch_base_files = []
    for dir_path in data_dirs:
        # Find X plane files as reference
        x_pattern = batch_pattern.replace("{plane}", "X")
        x_files = sorted(glob.glob(os.path.join(dir_path, x_pattern)))
        
        for x_file in x_files:
            # Derive base filename
            base_file = x_file.replace("_planeX.npz", "")
            
            # Check if corresponding U and V files exist
            u_file = base_file + "_planeU.npz"
            v_file = base_file + "_planeV.npz"
            
            if os.path.exists(u_file) and os.path.exists(v_file):
                batch_base_files.append(base_file)
    
    print(f"Found {len(batch_base_files)} matched batch files")
    
    # Load data from matched batches, keeping only samples with all 3 planes
    all_images_u = []
    all_images_v = []
    all_images_x = []
    all_labels = []
    
    total_samples_loaded = 0
    total_samples_kept = 0
    
    for i, base_file in enumerate(batch_base_files):
        if i % 100 == 0 and i > 0:
            print(f"  Processed {i}/{len(batch_base_files)} batches, kept {total_samples_kept}/{total_samples_loaded} samples ({100*total_samples_kept/total_samples_loaded:.1f}%)")
        
        # Load all three planes
        u_data = np.load(base_file + "_planeU.npz", allow_pickle=True)
        v_data = np.load(base_file + "_planeV.npz", allow_pickle=True)
        x_data = np.load(base_file + "_planeX.npz", allow_pickle=True)
        
        u_imgs = u_data['images']
        v_imgs = v_data['images']
        x_imgs = x_data['images']
        x_metadata = x_data['metadata']
        
        # Only keep samples where all three planes have data at that index
        min_samples = min(len(u_imgs), len(v_imgs), len(x_imgs))
        
        total_samples_loaded += max(len(u_imgs), len(v_imgs), len(x_imgs))
        total_samples_kept += min_samples
        
        if min_samples > 0:
            all_images_u.append(u_imgs[:min_samples])
            all_images_v.append(v_imgs[:min_samples])
            all_images_x.append(x_imgs[:min_samples])
            all_labels.append(x_metadata[:min_samples])
    
    print(f"\nFinished loading {len(batch_base_files)} batches")
    print(f"Total samples loaded: {total_samples_loaded}")
    print(f"Total samples kept (matched): {total_samples_kept}")
    print(f"Filtering ratio: {100*total_samples_kept/total_samples_loaded:.1f}%")
    
    # Concatenate all batches
    dataset_img_u = np.concatenate(all_images_u, axis=0)
    dataset_img_v = np.concatenate(all_images_v, axis=0)
    dataset_img_x = np.concatenate(all_images_x, axis=0)
    metadata = np.concatenate(all_labels, axis=0)
    
    print(f"\nFinal shapes:")
    print(f"  U: {dataset_img_u.shape}")
    print(f"  V: {dataset_img_v.shape}")
    print(f"  X: {dataset_img_x.shape}")
    print(f"  Metadata: {metadata.shape}")
    
    # Extract direction labels (columns 3-5: x, y, z)
    dataset_label = dl.extract_direction_labels(metadata)
    
    print(f"\nDirection labels shape: {dataset_label.shape}")
    print(f"Direction labels range: [{dataset_label.min():.3f}, {dataset_label.max():.3f}]")
    
    # Add channel dimension to each plane
    dataset_img_u = dataset_img_u[..., np.newaxis]
    dataset_img_v = dataset_img_v[..., np.newaxis]
    dataset_img_x = dataset_img_x[..., np.newaxis]
    
    print(f"\nFinal shapes with channel dimension:")
    print(f"  U: {dataset_img_u.shape}")
    print(f"  V: {dataset_img_v.shape}")
    print(f"  X: {dataset_img_x.shape}")
    
    # Determine number of samples and splits
    n_samples = len(dataset_label)
    
    # Apply max_samples limit if specified
    if max_samples is not None and n_samples > max_samples:
        print(f"\nLimiting dataset to {max_samples} samples (from {n_samples})")
        dataset_img_u = dataset_img_u[:max_samples]
        dataset_img_v = dataset_img_v[:max_samples]
        dataset_img_x = dataset_img_x[:max_samples]
        dataset_label = dataset_label[:max_samples]
        n_samples = max_samples
    
    n_train = int(n_samples * train_fraction)
    n_val = int(n_samples * val_fraction)
    
    # Shuffle
    if shuffle_data:
        np.random.seed(random_seed)
        indices = np.random.permutation(n_samples)
    else:
        indices = np.arange(n_samples)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train+n_val]
    test_idx = indices[n_train+n_val:]
    
    # Create train/val/test splits for each plane
    train_imgs = {
        'U': dataset_img_u[train_idx],
        'V': dataset_img_v[train_idx],
        'X': dataset_img_x[train_idx]
    }
    val_imgs = {
        'U': dataset_img_u[val_idx],
        'V': dataset_img_v[val_idx],
        'X': dataset_img_x[val_idx]
    }
    test_imgs = {
        'U': dataset_img_u[test_idx],
        'V': dataset_img_v[test_idx],
        'X': dataset_img_x[test_idx]
    }
    
    train = (train_imgs, dataset_label[train_idx])
    validation = (val_imgs, dataset_label[val_idx])
    test = (test_imgs, dataset_label[test_idx])
    
    print(f"\nData split:")
    print(f"  Train: {len(train[1])} samples")
    print(f"  Validation: {len(validation[1])} samples")
    print(f"  Test: {len(test[1])} samples")
    
    return train, validation, test

