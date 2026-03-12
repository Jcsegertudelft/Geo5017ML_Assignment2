from Loader import load_one_file, load_all_files
import numpy as np
from scipy.spatial import KDTree

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
    if features.ndim == 1:
        mean = np.mean(features)
        std = np.std(features)
    else:
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0)
    features = (features - mean) / std
    return features

# TODO: Implement features: give suiting name, return value

# Ideas features : Mean planarity of neighbourhood of 20~ points
# mean Z-length / XY-length of normal vector (fences have no little normal facing up)
# some size stuff
def feature_1(data, KDT = None):
    #Temporary test function
    return len(data)

def feature_2(data, KDT):
    return False

def feature_3(data, KDT):
    return False

def feature_4(data, KDT):
    return False

def feature_5(data, KDT):
    return False

def feature_6(data, KDT):
    return False

if __name__ == '__main__':
    features = calc_features(load_all_files())
    features = standardize(features)
    print(features)
