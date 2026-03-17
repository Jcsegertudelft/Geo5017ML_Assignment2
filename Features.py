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
            feature_1(data_point),
            feature_2(data_point),
            feature_3(data_point),
            feature_4(data_point),
            feature_5(data_point),
            feature_6(data_point)
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
        mean = np.mean(features, axis=1)
        std = np.std(features, axis=1)

    features = (features.T - mean) /std
    return features.T

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
    Height above ground: highest point - estimated ground height using 5th percentile.
    Distinguishes tall objects like trees from low objects like cars.
    """
    height = data[:,2]
    max_height = np.max(height)
    ground_height = np.percentile(height,5)
    height_above_ground = max_height - ground_height
    return height_above_ground


def feature_5(data):
    """
    3D point density: number of points / 3D convex hull volume. Example: A building has large volume of points
    Trees have irregular but large distribution/volume
    """
    hull=ConvexHull(data)
    return data.shape[0]/hull.volume

def feature_6(data):
    """
    Bounding box aspect ratio: height / max horizontal width
    Measures how tall vs wide an object is. Good seperation of poles
    """
    x_range = np.max(data[:,0]) - np.min(data[:,0])
    y_range = np.max(data[:,1]) - np.min(data[:,1])
    z_range = np.max(data[:,2]) - np.min(data[:,2])

    width = max(x_range, y_range)
    return z_range / width if width > 0 else 0

def calc_eigvals(data):
    cov_matr = np.cov(data.T)
    eig_vals = np.linalg.eigvals(cov_matr)
    return eig_vals

if __name__ == '__main__':
    data = load_one_file(4)
    calc_eigvals(data)
    feature_1(data)
