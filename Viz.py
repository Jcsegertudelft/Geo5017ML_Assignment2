from Loader import load_one_file, load_all_files
from Features import *
import numpy as np
from matplotlib import pyplot as plt

def plot_pc(data, classes = None):
    "Plots a point cloud, with or without labels"
    plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')
    if classes.any() != None:
        assert data.shape[0] == len(classes)
        num_classes = len(np.unique(classes))
        colors = np.array([
            [np.random.randint(0, 255),
             np.random.randint(0, 255),
             np.random.randint(0, 255)]
        for _ in range(num_classes)])/255
        for cls, color in zip(np.unique(classes), colors):
            print(color)
            ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=color)

    else:
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='black')
    plt.show()
    return

def plot_feature(data, feature_func):
    "Plot the distribution of features per class"
    feature_vals = standardize(np.array([feature_1(data_point) for data_point in data]))
    plt.figure(figsize=(10, 10))
    plt.scatter(feature_vals[:100], np.full(100,1), c='black', alpha=0.5)
    plt.scatter(feature_vals[100:200],np.full(100, 2),c='red', alpha=0.5)
    plt.scatter(feature_vals[200:300],np.full(100, 3),c='green', alpha=0.5)
    plt.scatter(feature_vals[300:400],np.full(100, 4),c='blue', alpha=0.5)
    plt.scatter(feature_vals[400:],np.full(100, 5),c='yellow', alpha=0.5)
    plt.xlim([-5,5])
    plt.show()


if __name__ == '__main__':
    data = load_all_files()
    plot_feature(data, feature_1)