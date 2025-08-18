import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.lines import Line2D

def plot_decision_boundary(model, features, labels, title, test_idxs=None):
    # plot feature space and decision boundary for a trained classifier
    # inputs: model -> trained classifier, features -> numpy array of shape (n_samples, n_features) ie. 36 epochs, each with alpha/theta power features, 
    # labels -> associated labels (0 = eyes open), title -> plot title string

    # create grid for feature space
    x_min, x_max = features[:, 0].min() - 0.5, features[:, 0].max() + 0.5
    y_min, y_max = features[:, 1].min() - 0.5, features[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200)) # corresponds to a 200x200 coordinate pairs (grid points) -> 2 layers, one for x and one for y coordinates, overlayed

    # turn coordinate layers into a list of cartesian points
    grid_points = np.c_[xx.ravel(), yy.ravel()] # .ravel() -> 'flatten' grid ie. 2D to 1D, .c_[] -> stacks arrays as columns ie. now (x, y) coordinate pairs
    Z = model.predict(grid_points).reshape(xx.shape) # classify each grid point, reshape to the 2D grid

    # plot feature space and decision boundary
    plt.figure(figsize=(7, 6))
    plt.contourf(xx, yy, Z, alpha=0.2, cmap='bwr') # colours decision regions based on Z

    # identify training vs test set
    train_mask = np.ones(len(labels), dtype=bool)
    test_mask = np.zeros(len(labels), dtype=bool)
    if test_idxs is not None:
        train_mask[test_idxs] = False
        test_mask[test_idxs] = True
    # train_mask stores indices of training epochs, test_mask stores indices of testing epochs

    # held-in -> training
    plt.scatter(features[train_mask & (labels == 0), 0], features[train_mask & (labels == 0), 1], color='blue', edgecolor='k', marker='o', label='Eyes open')
    plt.scatter(features[train_mask & (labels == 1), 0], features[train_mask & (labels == 1), 1], color='red', edgecolor='k', marker='o', label='Eyes closed')

    # held-out -> testing 
    if test_idxs is not None:
        plt.scatter(features[test_mask & (labels == 0), 0], features[test_mask & (labels == 0), 1], color='blue', edgecolor='k', marker='^', label='Eyes open')
        plt.scatter(features[test_mask & (labels == 1), 0], features[test_mask & (labels == 1), 1], color='red', edgecolor='k', marker='^', label='Eyes closed')

    plt.xlabel("Mean alpha band power [dB]")
    plt.ylabel("Mean theta band power [dB]")
    # plt.title(title)
    if test_idxs is not None:
        # manual legend
        legend_elements = [
            Line2D([0], [0], linestyle='None', label='Eyes open'),
            Line2D([0], [0], linestyle='None', label='Eyes closed'),
            Line2D([0], [0], color='k', marker='o', linestyle='None', label='Train'),
            Line2D([0], [0], color='k', marker='^', linestyle='None', label='Test')
        ]
        leg = plt.legend(handles=legend_elements, loc='best')
        for text, col in zip(leg.get_texts(), ['blue', 'red']):
            text.set_color(col)
    else:
        plt.legend(loc='upper left')
        
    # save plot
    file_name = f"plots/{title.replace(' ', '_').lower()}.png"
    plt.savefig(file_name, dpi=300)

    plt.close()

    return
