'''
Main file
'''

import numpy as np
import pandas as pd

from Loader import load_one_file, load_all_files
from Features import *
from typing import Tuple
from pandas import DataFrame

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


def forward_search(training_set_df,n_considered, n_wanted):
    selected_features = []
    candidate_features = list(range(1,n_considered+1))
    while len(selected_features) < n_wanted:
        selection_value = []
        for feature in candidate_features:
            feature_set = tuple(selected_features + [feature])
            within = within_scatter(training_set_df, feature_set)
            between = between_scatter(training_set_df, feature_set)
            J = np.linalg.trace(between) / np.linalg.trace(within)
            selection_value.append(J)
        selected_feature = candidate_features.pop(np.argmax(selection_value))
        selected_features.append(selected_feature)
    return selected_features

def create_feature_df(pc_objects):
    df = pd.DataFrame()
    df['number'] = [obj.number for obj in pc_objects]
    df['class'] = [obj.label for obj in pc_objects]
    all_features = np.array([obj.feature for obj in pc_objects]).T
    all_features = standardize(all_features)
    for i in range(all_features.shape[0]):
        df[f'feature_{i+1}'] = all_features[i]
    return df

def within_scatter(df, feature_numbers:Tuple):
    feat_column_names = [f'feature_{i}' for i in feature_numbers]
    scatter_matr = np.zeros((len(feature_numbers),len(feature_numbers)))
    for cls in range(5):
        df_class = df[df['class'] == cls]
        feature_columns = np.array(df_class[feat_column_names].values).T
        scatter_matr += feature_columns.shape[1]/df.shape[0] * np.cov(feature_columns)
    return scatter_matr

def between_scatter(df, feature_numbers:Tuple):
    feat_column_names = [f'feature_{i}' for i in feature_numbers]
    scatter_matr = np.zeros((len(feature_numbers),len(feature_numbers)))
    total_means = df[feat_column_names].mean()
    for cls in range(5):
        df_class = df[df['class'] == cls]
        feature_columns = np.array(df_class[feat_column_names].values).T
        mean_diff = np.mean(feature_columns, axis = 1) - total_means
        mean_diff = np.reshape(mean_diff, (1,len(feature_numbers)))
        scatter_matr += feature_columns.shape[1]/df.shape[0] * np.linalg.matmul(mean_diff.T, mean_diff)
    return scatter_matr


def main():
    print('Reading data')
    pc_objects = np.array([pc_object(i) for i in range(500)])
    rng = np.random.default_rng(42) #Set the rng consistently
    print('Computing features')
    for obj in pc_objects:
        obj.compute_features()
    print('Selecting Training/Test split')
    selection = rng.choice(np.arange(100),size = 30, replace = False) #Random selection of test set
    test_set_ind = []
    for i in range(5):
        test_set_ind += list(selection + i*100)
    test_set_ind = np.array(test_set_ind)
    test_set = pc_objects[test_set_ind]
    train_set = np.delete(pc_objects,test_set_ind)
    print('Put features in Dataframe')
    train_set_df = create_feature_df(train_set)
    test_set_df = create_feature_df(test_set)
    print('Selecting features based on forward search')
    selected_features = forward_search(train_set_df, 6, 4)
    print(selected_features)
    columns_exported = ['number', 'class'] + [f'feature_{i}' for i in selected_features]
    print(train_set_df)
    train_set_df.to_csv('train_set.csv', index = False, columns = columns_exported)
    test_set_df.to_csv('test_set.csv', index = False, columns = columns_exported)



if __name__ == '__main__':
    main()
