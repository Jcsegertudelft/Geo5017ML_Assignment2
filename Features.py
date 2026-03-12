from Loader import load_one_file, load_all_files
import numpy as np
from scipy.spatial import kdtree

def main(data):
    features = []
    for data_point in data:

        KDT = kdtree.KDTree(data_point)

        feature_vector = [
            feature_1(data_point, KDT),
            feature_2(data_point, KDT),
            feature_3(data_point, KDT),
            feature_4(data_point, KDT),
            feature_5(data_point, KDT),
            feature_6(data_point, KDT)
        ]
        features.append(feature_vector)

# TODO: Implement features: give suiting name, return value
def feature_1(data, KDT):
    return False

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


