'''
File with functions that load the pointcloud data
'''

import numpy as np
from os import listdir

def load_one_file(number):
    number = str(number)
    while len(number) <3:
        number = '0' + number
    file_path = "Data/Data/" + number + ".xyz"
    data = np.loadtxt(file_path)
    return data

def load_all_files():
    all_data = []
    for file in listdir("Data/Data"):
        file_path = "Data/Data/" + file
        data = np.loadtxt(file_path)
        all_data.append(data)
    return all_data

if __name__ == "__main__":
    load_all_files()