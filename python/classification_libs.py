import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
# import hyperopt as hp  # Optional, only for hyperopt models
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from mpl_toolkits.axes_grid1 import ImageGrid
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
# import healpy as healpy  # Optional

import general_purpose_libs as gpl
import data_loader as dl


def prepare_data_from_npz(data_dir, plane, dataset_parameters, output_folder):
    """
    Prepare data from NPZ batch files for training.
    
    Args:
        data_dir: Directory containing NPZ batch files
        plane: Plane to use ('U', 'V', or 'X')
        dataset_parameters: Dictionary with dataset parameters
        output_folder: Output folder for saving samples and plots
        
    Returns:
        train, validation, test: Tuples of (images, labels)
    """
    train_fraction = dataset_parameters.get("train_fraction", 0.8)
    val_fraction = dataset_parameters.get("val_fraction", 0.1)
    test_fraction = dataset_parameters.get("test_fraction", 0.1)
    aug_coefficient = dataset_parameters.get("aug_coefficient", 1)
    prob_per_flip = dataset_parameters.get("prob_per_flip", 0.5)
    balance_data = dataset_parameters.get("balance_data", False)
    balance_method = dataset_parameters.get("balance_method", "undersample")
    max_samples = dataset_parameters.get("max_samples", None)
    
    print("\n" + "="*60)
    print("LOADING DATA FROM NPZ FILES")
    print("="*60)
    print(f"Data directory: {data_dir}")
    print(f"Plane: {plane}")
    print(f"Max samples: {max_samples if max_samples else 'All'}")
    
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Load dataset from NPZ files
    batch_pattern = dataset_parameters.get(
        "batch_pattern", 'clusters_plane{plane}_batch*.npz'
    )

    dataset_img, metadata = dl.load_dataset_from_directory(
        data_dir=data_dir,
        plane=plane,
        batch_pattern=batch_pattern,
        max_samples=max_samples,
        verbose=True
    )
    
    # Extract labels for main track identification
    dataset_label = dl.extract_labels_for_mt_identification(metadata)
    
    # Get dataset statistics
    stats = dl.get_dataset_statistics(metadata, verbose=True)
    
    # Add channel dimension if needed (for CNN compatibility)
    if len(dataset_img.shape) == 3:
        dataset_img = np.expand_dims(dataset_img, axis=-1)
    
    print(f"Dataset shape after preprocessing: {dataset_img.shape}")
    print(f"Labels shape: {dataset_label.shape}")
    print(f"Unique labels: {np.unique(dataset_label)}")
    
    # Balance dataset if requested
    if balance_data:
        print(f"\nBalancing dataset using {balance_method}...")
        dataset_img, dataset_label = dl.balance_dataset(
            dataset_img, dataset_label, 
            method=balance_method
        )
    
    # Shuffle the dataset
    print("\nShuffling the dataset...")
    index = np.arange(dataset_img.shape[0])
    np.random.shuffle(index)
    dataset_img = dataset_img[index]
    dataset_label = dataset_label[index]
    print("Dataset shuffled.")
    
    # Save some sample images
    samples_dir = os.path.join(output_folder, "samples")
    if not os.path.exists(samples_dir):
        os.makedirs(samples_dir)
    print(f"\nSaving sample images to {samples_dir}...")
    save_samples_from_ds(dataset_img, dataset_label, samples_dir + "/")
    print("Sample images saved.")
    
    # Split the dataset into training, validation and test
    print("\nSplitting the dataset...")
    n_total = dataset_img.shape[0]
    n_train = int(n_total * train_fraction)
    n_val = int(n_total * val_fraction)
    
    train_images = dataset_img[:n_train]
    validation_images = dataset_img[n_train:n_train+n_val]
    test_images = dataset_img[n_train+n_val:]
    
    train_labels = dataset_label[:n_train]
    validation_labels = dataset_label[n_train:n_train+n_val]
    test_labels = dataset_label[n_train+n_val:]
    
    print(f"Training set: {train_images.shape[0]} samples")
    print(f"Validation set: {validation_images.shape[0]} samples")
    print(f"Test set: {test_images.shape[0]} samples")
    
    # Data augmentation
    if aug_coefficient > 1:
        print(f"\nApplying data augmentation (coefficient={aug_coefficient})...")
        print(f"Train images shape before: {train_images.shape}")
        train_images, train_labels = data_augmentation(
            train_images, train_labels, 
            coefficient=aug_coefficient, 
            prob_per_flip=prob_per_flip
        )
        print(f"Train images shape after: {train_images.shape}")
        print("Data augmentation completed.")
    
    # Prepare return tuples
    train = (train_images, train_labels)
    validation = (validation_images, validation_labels)
    test = (test_images, test_labels)
    
    print("="*60)
    print("DATA PREPARATION COMPLETE")
    print("="*60 + "\n")
    
    return train, validation, test


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
    # remove images where np.sum is 655350000
    print("Removing corrupted images...")
    corrupted_images = []
    for i in range(dataset_img.shape[0]):
        if (np.sum(dataset_img[i])==655350000):
            corrupted_images.append(i)
    print("Corrupted images: ", len(corrupted_images))
    dataset_img = np.delete(dataset_img, corrupted_images, axis=0)
    dataset_label = np.delete(dataset_label, corrupted_images, axis=0)
    

    print("Dataset loaded.")
    print("Dataset_img shape: ", dataset_img.shape)
    print("Dataset_lab shape: ", dataset_label.shape)
    print("dataset_img type: ", type(dataset_img))
    print("dataset_lab type: ", type(dataset_label))

    if not os.path.exists(output_folder+"samples/"):
        os.makedirs(output_folder+"samples/")
    # Check if the dimension of images and labels are the same
    if dataset_img.shape[0] != dataset_label.shape[0]:
        print("Error: the dimension of images and labels are not the same.")
    # Handle legacy metadata without explicit plane: fallback to requested plane ID
    plane_ids = metadata[:, -1] if metadata.shape[1] > 11 else np.full(len(metadata), {'U': 0, 'V': 1, 'X': 2}[plane])
    # shuffle the dataset
    print("Shuffling the dataset...")
    index = np.arange(dataset_img.shape[0])
    np.random.shuffle(index)
    dataset_img = dataset_img[index]

    # Save some images
    print("Saving some images...")
    save_samples_from_ds(dataset_img, dataset_label, output_folder+"samples/")
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
    print("Dataset splitted.")

    if aug_coefficient>1:
        print("Data augmentation...")
        print("Train images shape before: ", train_images.shape)
        train_images, train_labels = data_augmentation(train_images, train_labels, coefficient=aug_coefficient, prob_per_flip=prob_per_flip)
        print("Train images shape after: ", train_images.shape)
        print("Data augmented.")
    
    print("Unique labels: ", np.unique(train_labels, return_counts=True))

    # Create the datasets
    print("Creating the dataset objects...")
    with tf.device("CPU"):
        train = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(10000).batch(32)
        # CRITICAL: Add .repeat() to validation dataset to allow multiple passes during hyperopt
        validation = tf.data.Dataset.from_tensor_slices((validation_images, validation_labels)).batch(32).repeat()
        test = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(32)
    print("Datasets created.")
    return train, validation, test

def data_augmentation(dataset, labels, coefficient=2, prob_per_flip=0.5):
    # create the augmented dataset
    augmented_dataset = []
    augmented_labels = []
    # for each sample
    for i in range(int(coefficient*len(dataset))):
        index = np.random.randint(0, len(dataset))
        # get the sample
        sample = dataset[index]
        # get the label
        label = labels[index]
        if np.random.rand() > prob_per_flip:
            # flip the sample
            sample = np.flipud(sample)
        if np.random.rand() > prob_per_flip:
            sample = np.fliplr(sample)
        augmented_dataset.append(sample)
        augmented_labels.append(label)
    return np.array(augmented_dataset), np.array(augmented_labels)

def calculate_metrics(y_true, y_pred,):
    # calculate the confusion matrix, the accuracy, and the precision and recall 
    # binary trick
    y_pred_am = np.where(y_pred > 0.5, 1, 0)
    cm = confusion_matrix(y_true, y_pred_am, normalize='true')
    # compute precision matrix
    

    accuracy = accuracy_score(y_true, y_pred_am)
    precision = precision_score(y_true, y_pred_am, average='macro')
    recall = recall_score(y_true, y_pred_am, average='macro')
    f1 = f1_score(y_true, y_pred_am, average='macro')

    return cm, accuracy, precision, recall, f1
    
def log_metrics(y_true, y_pred, output_folder="", label_names=["CC", "ES"]):
    cm, accuracy, precision, recall, f1 = calculate_metrics(y_true, y_pred)
    print("Confusion Matrix")
    print(cm)
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1: ", f1)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    with open(output_folder+f"metrics.txt", "a") as f:
        f.write("Confusion Matrix\n")
        f.write(str(cm)+"\n")
        f.write("Accuracy: "+str(accuracy)+"\n")
        f.write("Precision: "+str(precision)+"\n")
        f.write("Recall: "+str(recall)+"\n")
        f.write("F1: "+str(f1)+"\n")
    # save confusion matrix 
    plt.figure(figsize=(10,10))
    plt.title("Confusion matrix", fontsize=28)
    sns.heatmap(cm, annot=True, cmap="YlGnBu", xticklabels=label_names, yticklabels=label_names, annot_kws={"fontsize": 20})
    plt.ylabel('True label', fontsize=28)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('Predicted label', fontsize=28)
    plt.savefig(output_folder+f"confusion_matrix.png")
    plt.clf()
    # Binarize the output
    y_test = label_binarize(y_true, classes=np.arange(len(label_names)))
    n_classes = y_test.shape[1]
    
    plt.figure()

    fpr, tpr, _ = roc_curve(y_true[:], y_pred[:])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label='ROC curve (area = {0:0.2f})'.format(roc_auc))

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate', fontsize=28)
    plt.ylabel('True Positive Rate', fontsize=28)
    plt.title('ROC curve', fontsize=28)
    plt.legend(loc="lower right", fontsize=20)
    plt.savefig(output_folder+"roc_curve.png")
    plt.clf()
    # create an histogram of the predictions
    print(y_pred.shape)
    print(y_true.shape)
    y_true = np.reshape(y_true, (y_true.shape[0],))
    bkg_preds = y_pred[y_true < 0.5]
    sig_preds = y_pred[y_true > 0.5]
    print("Background predictions: ", bkg_preds.shape)
    print("Signal predictions: ", sig_preds.shape)

    plt.hist(bkg_preds, bins=50, alpha=0.5, label=f'{label_names[0]} (n={bkg_preds.shape[0]})')
    plt.hist(sig_preds, bins=50, alpha=0.5, label=f'{label_names[1]} (n={sig_preds.shape[0]})')
    plt.legend(loc='upper right')
    plt.xlabel('Prediction')
    plt.ylabel('Counts')
    plt.title('Predictions')
    plt.savefig(output_folder+f"predictions.png")
    plt.clf()
    
    # Create log-scale version
    plt.hist(bkg_preds, bins=50, alpha=0.5, label=f'{label_names[0]} (n={bkg_preds.shape[0]})')
    plt.hist(sig_preds, bins=50, alpha=0.5, label=f'{label_names[1]} (n={sig_preds.shape[0]})')
    plt.legend(loc='upper right')
    plt.xlabel('Prediction')
    plt.ylabel('Counts')
    plt.title('Predictions (log scale)')
    plt.yscale('log')
    plt.savefig(output_folder+f"predictions_log.png")
    plt.clf()
    
    # Save predictions and labels for later plot regeneration
    np.save(output_folder + "predictions.npy", y_pred)
    np.save(output_folder + "test_labels.npy", y_true)
    print(f"Saved predictions and labels to {output_folder}")

def save_sample_img(ds_item, output_folder, img_name):
    if ds_item.shape[2] == 1:
        img = ds_item[:,:,0]
        plt.figure(figsize=(10, 26))
        plt.title(img_name)
        plt.imshow(img)
        # plt.colorbar()
        # add x and y labels
        plt.xlabel("Channel")
        plt.ylabel("Time (ticks)")
        # save the image, with a bbox in inches smaller than the default but bigger than tight
        plt.savefig(output_folder+img_name+".png", bbox_inches='tight', pad_inches=1)
        plt.close()

    else:
        img_u = ds_item[:,:,0]
        img_v = ds_item[:,:,1]
        img_x = ds_item[:,:,2]
        fig = plt.figure(figsize=(8, 20))
        grid = ImageGrid(fig, 111,          # as in plt.subplot(111)
                        nrows_ncols=(1,3),
                        axes_pad=0.5,
                        share_all=True,
                        cbar_location="right",
                        cbar_mode="single",
                        cbar_size="30%",
                        cbar_pad=0.25,
                        )   


        if img_u[0, 0] != -1:
            im = grid[0].imshow(img_u)
            grid[0].set_title('U plane')
        if img_v[0, 0] != -1:
            im = grid[1].imshow(img_v)
            grid[1].set_title('V plane')
        if img_x[0, 0] != -1:
            im = grid[2].imshow(img_x)
            grid[2].set_title('X plane')
        grid.cbar_axes[0].colorbar(im)
        grid.axes_llc.set_yticks(np.arange(0, img_u.shape[0], 100))
        # save the image
        plt.savefig(output_folder+ 'multiview_' + img_name + '.png')
        plt.close()

def save_samples_from_ds(dataset, labels, output_folder, name="img", n_samples_per_label=10):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # get the labels
    labels_unique = np.unique(labels, return_counts=True)
    # get the samples
    for label in labels_unique[0]:
        # get the indices
        indices = np.where(labels == label)[0]
        indices = indices[:np.minimum(n_samples_per_label, indices.shape[0])]
        samples = dataset[indices]
        # save the images
        for i, sample in enumerate(samples):
            save_sample_img(sample, output_folder, name+"_"+str(label)+"_"+str(i))
    # make one image for each label containing n_samples_per_label images, using plt suplot
    fig= plt.figure(figsize=(20, 20))
    for i, label in enumerate(labels_unique[0]):
        grid = ImageGrid(fig, 111,          # as in plt.subplot(111)
                nrows_ncols=(1,10),
                axes_pad=0.5,
                share_all=True,
                cbar_location="right",
                cbar_mode="single",
                cbar_size="30%",
                cbar_pad=0.25,
                )   
        indices = np.where(labels == label)[0]
        indices = indices[:np.minimum(n_samples_per_label, indices.shape[0])]
        samples = dataset[indices]
        # save the images
        plt.suptitle("Label: "+str(label), fontsize=25)

        for j, sample in enumerate(samples):
            im = grid[j].imshow(sample[:,:,0])
            grid[j].set_title(j)

        grid.cbar_axes[0].colorbar(im)
        grid.axes_llc.set_yticks(np.arange(0, sample.shape[0], 100))
        plt.savefig(output_folder+ 'all_'+str(label)+'.png')
        plt.clf()

def test_model(model, test, output_folder, label_names=["CC", "ES"]):
    print("Doing some test...")
    predictions = model.predict(test)      
    # Calculate metrics
    print("Calculating metrics...")
    # get the test labels from the test dataset
    test_labels = test[1]
    log_metrics(test_labels, predictions, output_folder=output_folder, label_names=label_names)
    print("Metrics calculated.")
    print("Drawing model...")
    keras.utils.plot_model(model, output_folder+"architecture.png", show_shapes=True)
    print("Model drawn.")
    print("Drawing histogram of energies...")
    test_img = test[0]
    
    # Extract metadata if available (test tuple has 3 elements: img, labels, metadata)
    test_metadata = test[2] if len(test) > 2 else None
    
    histogram_of_enegies(test_labels, predictions, test_img, limit=0.5, output_folder=output_folder)
    print("Drawing prediction vs energy scatter plot...")
    plot_prediction_vs_energy(test_labels, predictions, test_img, metadata=test_metadata, output_folder=output_folder)

    print("Test done.")

def histogram_of_enegies(test_labels, predictions, images, limit=0.5, output_folder=""):
    # check if some images are corrupted
    corrupted_images = []
    for i in range(images.shape[0]):
        if (np.sum(images[i])==0):
            corrupted_images.append(i)
    print("Corrupted images: ", len(corrupted_images))


    true_positives = []
    true_negatives = []
    false_positives = []
    false_negatives = []
    all_images = []
    for i in range(len(test_labels)):
        if test_labels[i] == 1 and predictions[i] > limit:
            true_positives.append(np.sum(images[i]))
        elif test_labels[i] == 0 and predictions[i] < limit:
            true_negatives.append(np.sum(images[i]))
        elif test_labels[i] == 0 and predictions[i] > limit:
            false_positives.append(np.sum(images[i]))
        elif test_labels[i] == 1 and predictions[i] < limit:
            false_negatives.append(np.sum(images[i]))
        all_images.append(np.sum(images[i]))
    
    print("True Positives: ", len(true_positives))
    print("True Negatives: ", len(true_negatives))
    print("False Positives: ", len(false_positives))
    print("False Negatives: ", len(false_negatives))
    print("All images: ", len(all_images))
    print(np.unique(np.array(all_images), return_counts=True))

    # sum the pixel values (ADC sums)
    # Auto-calculate range from actual data
    all_values = true_positives + true_negatives + false_positives + false_negatives
    if len(all_values) > 0:
        val_min, val_max = np.min(all_values), np.max(all_values)
        # Add 5% margin
        margin = 0.05 * (val_max - val_min)
        hist_range = (max(0, val_min - margin), val_max + margin)
    else:
        hist_range = (0, 3e5)  # Fallback to reasonable ADC range
    
    plt.figure(figsize=(10, 6))
    plt.hist(true_positives, range=hist_range, bins=100, alpha=0.5, label='True Positives (n='+str(len(true_positives))+')')
    plt.hist(true_negatives, range=hist_range, bins=100, alpha=0.5, label='True Negatives (n='+str(len(true_negatives))+')')
    plt.hist(false_positives, range=hist_range, bins=100, alpha=0.5, label='False Positives (n='+str(len(false_positives))+')')
    plt.hist(false_negatives, range=hist_range, bins=100, alpha=0.5, label='False Negatives (n='+str(len(false_negatives))+')')

    plt.legend(loc='upper right')
    plt.xlabel('Pixel value')
    plt.ylabel('Counts')
    plt.title('Pixel value histogram')
    plt.savefig(output_folder+"pixel_value_histogram.png")
    plt.clf()

def plot_prediction_vs_energy(test_labels, predictions, images, metadata=None, output_folder=""):
    """
    Create scatter plot of NN prediction vs cluster energy.
    Colors separate background (blue) and main track (red) clusters.
    
    Args:
        test_labels: True labels (0=background, 1=main track)
        predictions: NN prediction outputs
        images: Image data (used as fallback if metadata not provided)
        metadata: Cluster metadata array where column 10 contains energy in MeV
        output_folder: Where to save plots
    """
    # Get cluster energies from metadata (column 10) or compute from ADC sum
    if metadata is not None and metadata.shape[1] > 10:
        # Column 10 contains energy in MeV (already converted from ADC using proper factors)
        cluster_energies = metadata[:, 10]
        energy_unit = "MeV"
        print(f"Using cluster energy from metadata (column 10, in MeV)")
    else:
        # Fallback: sum ADC values (this is less accurate as it doesn't use proper conversion factors)
        cluster_energies = np.sum(images, axis=(1, 2))
        energy_unit = "ADC sum"
        print(f"Warning: Using ADC sum as energy (metadata not provided)")
    
    # Flatten predictions if needed
    if len(predictions.shape) > 1:
        predictions = predictions.flatten()
    
    # Separate by true label
    bkg_mask = test_labels < 0.5
    mt_mask = test_labels > 0.5
    
    bkg_energies = cluster_energies[bkg_mask]
    bkg_preds = predictions[bkg_mask]
    
    mt_energies = cluster_energies[mt_mask]
    mt_preds = predictions[mt_mask]
    
    print(f"Creating prediction vs energy plot...")
    print(f"  Background clusters: {len(bkg_energies)}")
    print(f"  Main track clusters: {len(mt_energies)}")
    print(f"  Energy range: {cluster_energies.min():.2f} - {cluster_energies.max():.2f} {energy_unit}")
    
    # Linear scale plot
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(bkg_energies, bkg_preds, alpha=0.3, s=2, c='blue', label=f'Background (n={len(bkg_energies)})')
    ax.scatter(mt_energies, mt_preds, alpha=0.3, s=2, c='red', label=f'Main Track (n={len(mt_energies)})')
    ax.set_xlabel(f'Cluster Energy ({energy_unit})', fontsize=14)
    ax.set_ylabel('NN Prediction', fontsize=14)
    ax.set_title('NN Prediction vs Cluster Energy', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig(output_folder + "prediction_vs_energy.png", dpi=150)
    plt.clf()
    
    # Log scale plot
    fig2, ax2 = plt.subplots(figsize=(10, 7))
    ax2.scatter(bkg_energies, bkg_preds, alpha=0.3, s=2, c='blue', label=f'Background (n={len(bkg_energies)})')
    ax2.scatter(mt_energies, mt_preds, alpha=0.3, s=2, c='red', label=f'Main Track (n={len(mt_energies)})')
    ax2.set_xlabel(f'Cluster Energy ({energy_unit})', fontsize=14)
    ax2.set_ylabel('NN Prediction', fontsize=14)
    ax2.set_title('NN Prediction vs Cluster Energy (log scale)', fontsize=16)
    ax2.set_xscale('log')
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig(output_folder + "prediction_vs_energy_log.png", dpi=150)
    plt.clf()
    
    print(f"  Saved: prediction_vs_energy.png")
    print(f"  Saved: prediction_vs_energy_log.png")


def prepare_data_from_multiple_npz(data_dirs, plane, dataset_parameters, output_folder):
    """
    Prepare data from multiple NPZ batch file directories for training.
    
    Args:
        data_dirs: List of directories containing NPZ batch files
        plane: Plane to use ('U', 'V', or 'X')
        dataset_parameters: Dictionary with dataset parameters
        output_folder: Output folder for saving samples and plots
        
    Returns:
        train, validation, test: Tuples of (images, labels)
        history_dict: Dictionary to store performance metrics
    """
    train_fraction = dataset_parameters.get("train_fraction", 0.7)
    val_fraction = dataset_parameters.get("val_fraction", 0.15)
    test_fraction = dataset_parameters.get("test_fraction", 0.15)
    balance_data = dataset_parameters.get("balance_data", False)
    balance_method = dataset_parameters.get("balance_method", "undersample")
    max_samples = dataset_parameters.get("max_samples", None)
    batch_pattern = dataset_parameters.get(
        "batch_pattern", 'clusters_plane{plane}_batch*.npz'
    )
    shuffle_data = dataset_parameters.get("shuffle_data", True)
    random_seed = dataset_parameters.get("random_seed", 42)
    
    print("\n" + "="*60)
    print("LOADING DATA FROM MULTIPLE NPZ DIRECTORIES")
    print("="*60)
    print(f"Data directories:")
    for dir in data_dirs:
        print(f"  - {dir}")
    print(f"Plane: {plane}")
    print(f"Shuffle: {shuffle_data}")
    print(f"Max samples: {max_samples if max_samples else 'All'}")
    
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Load dataset from multiple NPZ directories
    dataset_img, metadata = dl.load_dataset_from_multiple_directories(
        data_dirs=data_dirs,
        plane=plane,
        batch_pattern=batch_pattern,
        max_samples=max_samples,
        shuffle=shuffle_data,
        random_seed=random_seed,
        verbose=True
    )
    
    # Extract labels for main track identification
    dataset_label = dl.extract_labels_for_mt_identification(metadata)
    
    # Get dataset statistics
    stats = dl.get_dataset_statistics(metadata, verbose=True)
    
    # Add channel dimension if needed (for CNN compatibility)
    if len(dataset_img.shape) == 3:
        dataset_img = np.expand_dims(dataset_img, axis=-1)
    
    print(f"\nFinal dataset shape: {dataset_img.shape}")
    print(f"Labels shape: {dataset_label.shape}")
    
    # Balance dataset if requested
    if balance_data:
        print(f"\nBalancing dataset using {balance_method} method...")
        dataset_img, dataset_label = dl.balance_dataset(
            dataset_img, dataset_label, 
            method=balance_method,
            random_state=random_seed
        )
    
    # Split into train/val/test
    print("\n" + "="*60)
    print("SPLITTING DATASET")
    print("="*60)
    
    n_samples = len(dataset_img)
    n_train = int(n_samples * train_fraction)
    n_val = int(n_samples * val_fraction)
    n_test = n_samples - n_train - n_val
    
    print(f"Total samples: {n_samples}")
    print(f"Train: {n_train} ({train_fraction*100:.1f}%)")
    print(f"Validation: {n_val} ({val_fraction*100:.1f}%)")
    print(f"Test: {n_test} ({test_fraction*100:.1f}%)")
    
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Split data
    train_img = dataset_img[:n_train]
    train_label = dataset_label[:n_train]
    train_metadata = metadata[:n_train]
    
    val_img = dataset_img[n_train:n_train+n_val]
    val_label = dataset_label[n_train:n_train+n_val]
    val_metadata = metadata[n_train:n_train+n_val]
    
    test_img = dataset_img[n_train+n_val:]
    test_label = dataset_label[n_train+n_val:]
    test_metadata = metadata[n_train+n_val:]
    
    print(f"\nTrain distribution: {np.sum(train_label==1)} main tracks, {np.sum(train_label==0)} background")
    print(f"Val distribution: {np.sum(val_label==1)} main tracks, {np.sum(val_label==0)} background")
    print(f"Test distribution: {np.sum(test_label==1)} main tracks, {np.sum(test_label==0)} background")
    
    # Save sample images
    print("\nSaving sample images...")
    save_sample_images(train_img, train_label, output_folder, n_samples=10)
    
    # Initialize history dictionary for saving performance metrics
    history_dict = {
        'config': {
            'data_dirs': data_dirs,
            'plane': plane,
            'dataset_parameters': dataset_parameters,
            'n_train': n_train,
            'n_val': n_val,
            'n_test': n_test,
            'random_seed': random_seed
        },
        'dataset_stats': stats
    }
    
    # Return tuples with metadata included for test set
    return (train_img, train_label), (val_img, val_label), (test_img, test_label, test_metadata), history_dict
