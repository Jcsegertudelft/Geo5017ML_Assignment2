'''
Main file
'''

import numpy as np
from Loader import load_one_file, load_all_files
from Features import *

class pc_object:
    def __init__(self, number):
        self.number = number

        self.label = np.floor(number / 100)

        self.points = load_one_file(number)

        self.feature = []

    def compute_features(self):
        self.feature.append(feature_1(self.points))
        self.feature.append(feature_2(self.points))
        self.feature.append(feature_3(self.points))
        self.feature.append(feature_4(self.points))
        self.feature.append(feature_5(self.points))
        self.feature.append(feature_6(self.points))


def forward_search(training_set, features, n_wanted):
    selected_features = []
    while len(selected_features) < n_wanted:
        break
    pass


def main():
    features = np.array([pc_object(i) for i in range(500)])
    rng = np.random.default_rng(42) #Set the rng consistently
    selection = rng.choice(np.arange(100),size = 30, replace = False) #Random selection of test set
    test_set_ind = []
    for i in range(5):
        test_set_ind += list(selection + i*100)
    test_set_ind = np.array(test_set_ind)
    test_set = features[test_set_ind]
    train_set = np.delete(features,test_set_ind)



if __name__ == '__main__':
    main()
