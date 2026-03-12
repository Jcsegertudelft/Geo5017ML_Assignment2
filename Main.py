'''
Main file
'''

import numpy as np
from Loader import load_one_file, load_all_files
from Features import *

class object:
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