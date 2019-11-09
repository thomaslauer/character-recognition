from emnist_dataset import EmnistDataset

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def rotateImage(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def process_emnist(split, stage):
    emnist_dataset = EmnistDataset()

    data_x, data_y = emnist_dataset.load_split_numpy(split, stage)
    print(data_y)

    output_dir = os.path.join("../output", split, stage)


    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    

    for i in range(np.shape(data_x)[0]):
        if i % 100 == 0:
            print(i, "/", np.shape(data_x)[0])
        image = np.uint8(np.transpose(data_x[i][0]))
        image = rotateImage(image, np.random.uniform() * 360)

        if not os.path.isdir(os.path.join(output_dir, "{0:03d}".format(data_y[i]))):
            os.mkdir(os.path.join(output_dir, "{0:03d}".format(data_y[i])))

        cv2.imwrite(os.path.join(output_dir, "{0:03d}".format(data_y[i]), "{0:09d}.jpg".format(i)), image)

def process_char74k(split):
    print("asdf")


if __name__ == "__main__":
    """
    process_emnist("mnist", "train")
    process_emnist("mnist", "test")
    """
    process_char74k("asdf")