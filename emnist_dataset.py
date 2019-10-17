import matplotlib
matplotlib.use('qt5agg')

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys
import zipfile

class EmnistDataset:

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
        data_frame = pd.read_csv(csv_filename, header=None)

        # first column is the label, the rest is the image data
        labels = data_frame.iloc[:,0].values
        images = data_frame.iloc[:,1:].values

        # load the mapping for this split (converts labels to actual ascii codes)
        self.mapping = self.load_mapping(split)

        for i in range(10):
            print("Label is ", self.mapping[int(labels[i])])
            self.show_image(images[i])


    def load_mapping(self, split):
        d = {}
        with open(self.data_folder + "/emnist-" + split + "-mapping.txt") as f:
            for line in f:
                (key, val) = line.split()
                d[int(key)] = chr(int(val))
        
        return d

    def show_image(self, image):
        reshaped_image = np.reshape(image, (28, 28)).transpose()

        plt.imshow(reshaped_image)
        plt.show()


if __name__ == '__main__':
    dataset = EmnistDataset()

    dataset.prep_data()
    dataset.load_split('bymerge', 'test')

