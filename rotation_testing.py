from emnist_dataset import EmnistDataset

import cv2
import numpy as np
import matplotlib.pyplot as plt

def rotateImage(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


emnist_dataset = EmnistDataset()

data_x, data_y = emnist_dataset.load_split_numpy("byclass", "train")

print(np.shape(data_x[0]))

for i in range(np.shape(data_x)[0]):
    if i % 100 == 0:
        print(i, "/", np.shape(data_x)[0])
    image = np.uint8(np.transpose(data_x[i][0]))
    image = rotateImage(image, np.random.uniform() * 360)
    cv2.imwrite("output/{0:09d}.jpg".format(i), image)
