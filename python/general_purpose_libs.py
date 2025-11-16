import matplotlib.pyplot as plt
import numpy as np
import json
import sys
import os

from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pylab as pylab

params = {'legend.fontsize': 'x-large',
          'figure.figsize': (10, 10),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)

def create_report(output_folder, model_name, input_json, model_function):
    # create the report
    dataset_lab = np.load(input_json["input_label"])
    with open(output_folder+"report.txt", "w") as f:
        f.write("Model name: "+model_name+"\n")
        f.write("Input label: "+input_json["input_label"]+"\n")
        f.write("Input data: "+input_json["input_data"]+"\n")
        f.write("Unique labels: "+str(np.unique(dataset_lab, return_counts=True))+"\n")
        f.write("Input json: \n"+json.dumps(input_json, indent=4)+"\n")
        f.write("Model function: \n"+model_function+"\n")


def save_sample_img(ds_item, output_folder, img_name):
    if ds_item.shape[2] == 1:
        img = ds_item[:,:,0]
        # where pixels are 0, set them to np.nan
        img= np.where(img == 0, np.nan, img)
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

def save_history(history, output_folder):
    """Persist training history in both human-readable and structured formats."""
    if history is None:
        return

    os.makedirs(output_folder, exist_ok=True)

    if hasattr(history, 'history'):
        raw_history = history.history
    elif isinstance(history, dict):
        raw_history = history
    else:
        return

    history_dict = {k: [float(v) for v in values] for k, values in raw_history.items()}

    text_path = os.path.join(output_folder, "history.txt")
    with open(text_path, "w") as f:
        json.dump(history_dict, f, indent=2)

    json_path = os.path.join(output_folder, "training_history.json")
    with open(json_path, "w") as f:
        json.dump(history_dict, f, indent=2)

    def _plot_metric(metric_key, ylabel, filename):
        train_values = history_dict.get(metric_key)
        val_values = history_dict.get(f"val_{metric_key}")
        if not train_values:
            return
        plt.figure()
        plt.plot(train_values, label=metric_key)
        if val_values:
            plt.plot(val_values, label=f"val_{metric_key}")
        plt.title(f"Model {metric_key}")
        plt.ylabel(ylabel)
        plt.xlabel('Epoch')
        plt.legend(loc='upper left')
        plt.savefig(os.path.join(output_folder, filename))
        plt.close()

    _plot_metric('loss', 'Loss', 'loss.png')
    if 'accuracy' in history_dict or 'acc' in history_dict:
        metric_name = 'accuracy' if 'accuracy' in history_dict else 'acc'
        _plot_metric(metric_name, 'Accuracy', 'accuracy.png')


def history_to_serializable(history):
    """Return a JSON-serializable history dictionary."""
    if history is None:
        return {}
    if hasattr(history, 'history'):
        source = history.history
    elif isinstance(history, dict):
        source = history
    else:
        return {}
    serializable = {}
    for key, values in source.items():
        serializable[key] = [float(v) for v in values]
    return serializable


def write_results_json(output_folder, payload):
    """Persist a results.json file and return its path."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
    results_path = os.path.join(output_folder, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"✓ Results saved to: {results_path}")
    return results_path

