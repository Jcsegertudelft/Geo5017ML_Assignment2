'''
File with functions that load the point cloud data
'''

import numpy as np
from os import listdir

def load_one_file(number):
    """
    Loads a single point cloud file
    """
    number = str(number)
    # Need file names of format 'xxx.xyz'
    while len(number) <3:
        number = '0' + number
    file_path = "Data/Data/" + number + ".xyz"
    data = np.loadtxt(file_path)
    return data

def load_all_files():
    """
    Loads point cloud files and puts them in a list
    """
    all_data = []
    for file in listdir("Data/Data"):
        file_path = "Data/Data/" + file
        data = np.loadtxt(file_path)
        all_data.append(data)
    return all_data

if __name__ == "__main__":
    load_all_files()