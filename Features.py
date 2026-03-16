from Loader import load_one_file, load_all_files
import numpy as np
from scipy.spatial import KDTree, ConvexHull

def calc_features(data):
    """
    Main fucntion of this file
    calculating the features for each point cloud by which they are to be classified

    Returns an nxm array of these features values, n number of point clouds, m number of features
    """
    features = []
    for data_point in data:
        #KDTree per pointcloud
        KDT = KDTree(data_point)

        #Calculate Feature vector for each pointcloud
        feature_vector = [
            feature_1(data_point, KDT),
            feature_2(data_point, KDT),
            feature_3(data_point, KDT),
            feature_4(data_point, KDT),
            feature_5(data_point, KDT),
            feature_6(data_point, KDT)
        ]
        features.append(feature_vector)

    features = np.array(features)
    return features

def standardize(features):
    """
    Standardizing the features which are calculate above
    """
    features = np.array(features)
    if features.ndim == 1:
        mean = np.mean(features)
        std = np.std(features)
    else:
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0)
    features = (features - mean) / std
    return list(features)

# TODO: Implement features, return value
# Narrowness: variance in Z (or height zmax - zmin) over Variance X and Y
# More size related
def feature_1(data):
    '''
    2D point density: Number of points divided by area of 2D convex hull
    '''
    Convex_2d = ConvexHull(data[:,:2])
    return data.shape[0] / Convex_2d.volume


def feature_2(data):
    '''
    Mean Planarity: Planarity of neighbourhood of 10 closest + point itself
    '''
    KDT = KDTree(data)
    plan_list = []
    for point in data:
        subset_dists, subset_inds = KDT.query(point,11)
        subset_pts = data[subset_inds]
        eigvals = calc_eigvals(subset_pts)
        eigvals.sort()
        planarity = (eigvals[1]-eigvals[0])/eigvals[2]
        plan_list.append(planarity)
    return np.mean(plan_list)


def feature_3(data):
    """
    Mean linearity: Linearity of neighbourhood of 10 closest + point itself
    Distinguishes poles from other tall objects like buildings.
    """
    KDT = KDTree(data)
    linearity_list = []
    for point in data:
        subset_dists, subset_inds = KDT.query(point,11)
        subset_pts = data[subset_inds]
        eigvals = calc_eigvals(subset_pts)
        eigvals.sort()
        linearity = (eigvals[2]-eigvals[1])/eigvals[2]
        linearity_list.append(linearity)
    return np.mean(linearity_list)


def feature_4(data):
    """
    Height above ground: current point - estimated ground height using 5th percentile.
    Distinguishes tall objects like trees from low objects like cars.
    """
    height = data[:,2]
    perc_5_point = np.percentile(data[height],5)
    return height-perc_5_point


def feature_5(data):
    return False

def feature_6(data):
    return False

def calc_eigvals(data):
    cov_matr = np.cov(data.T)
    eig_vals = np.linalg.eigvals(cov_matr)
    return eig_vals

if __name__ == '__main__':
    data = load_one_file(4)
    calc_eigvals(data)
    feature_1(data)
