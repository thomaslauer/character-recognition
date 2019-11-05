import matplotlib
matplotlib.use('qt5agg')

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys
import torch
import zipfile

class EmnistDataset:

    """
    Extracts and loads the EMNIST datasets

    TODO: Make a custom PyTorch Dataset class to load larger splits. Currently it all has to be in ram at the same time. 
    """

    def __init__(self):
        self.mnist_zip = 'emnist.zip'
        self.data_folder = 'emnist'

        self.splits = ['balanced', 'byclass', 'bymerge', 'digits', 'letters', 'mnist']
        self.stages = ['train', 'test']
    
    def prep_data(self):

        """
        Checks if the data folder exists. If it doesn't, extract the zip.
        """

        if not os.path.isdir(self.data_folder):
            with zipfile.ZipFile(self.mnist_zip, 'r') as zip_ref:
                zip_ref.extractall(self.data_folder)

    def load_split_numpy(self, split, stage):
        """
        Loads the same splits, but in numpy arrays instead of tensors.
        """

        if not split in self.splits:
            print(split, "is not a valid EMNIST split")
            return

        if not stage in self.stages:
            print(stage, "is not a valid stage")
            return

        csv_filename = self.data_folder + "/emnist-" + split + "-" + stage + ".csv"
        print("Loading", csv_filename)
        data_frame = pd.read_csv(csv_filename, header=None, dtype=np.int32)

        np_x = data_frame.iloc[:,1:].values.reshape((-1, 1, 28, 28)) # Load the images
        np_y = data_frame.iloc[:,0].values  # Load the labels

        return np_x, np_y

    def load_split(self, split, stage):
        """
        Loads a specific dataset file from the csv. 

        Split refers to the EMNIST split in the dataset.
        See https://www.kaggle.com/crawford/emnist/ and 
        https://www.nist.gov/node/1298471/emnist-dataset for more info. 

        Stage refers to testing or training data. 
        """

        if not split in self.splits:
            print(split, "is not a valid EMNIST split")
            return

        if not stage in self.stages:
            print(stage, "is not a valid stage")
            return

        csv_filename = self.data_folder + "/emnist-" + split + "-" + stage + ".csv"
        print("Loading", csv_filename)
        data_frame = pd.read_csv(csv_filename, header=None, dtype=np.int8)

        tensor_x = torch.Tensor(data_frame.iloc[:,1:].values.reshape((-1, 1, 28, 28))).float() # Load the images
        tensor_y = torch.Tensor(data_frame.iloc[:,0].values).long()  # Load the labels

        return torch.utils.data.TensorDataset(tensor_x, tensor_y)

    def load_mapping(self, split):
        mapping = {}
        with open(self.data_folder + "/emnist-" + split + "-mapping.txt") as f:
            for line in f:
                (key, val) = (line.split()[0:2])
                mapping[int(key)] = chr(int(val))
        
        return mapping

    def show_image(self, image):
        reshaped_image = np.reshape(image, (28, 28)).transpose()

        plt.imshow(reshaped_image)
        plt.show()
