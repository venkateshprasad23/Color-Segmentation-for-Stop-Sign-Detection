#!/usr/bin/python

import numpy as np
import os
from matplotlib import pyplot as plt


def main():
    # these images do not exist
    excludes = [7, 34, 35]

    # these are the images with STOP signs
    stops = [i for i in range(1, 101)]
    for i in excludes:
        stops.remove(i)

    # random permutation is good
    stops = np.random.permutation(stops)

    # Set aside 19 for validation set
    stop_val = stops[:19]
    # the rest are for training purposes
    stop_train = stops[19:]

    # those images that do NOT have stop signs
    other_ims = [i for i in range(101, 201)]
    other_ims = np.random.permutation(other_ims)

    # set aside 20 images for validation
    other_val = other_ims[:20]
    # the rest of the images are for training purposes
    other_train = other_ims[20:]

    print(stop_train, stop_val)
    print(other_train, other_val)

    # ALL folder name initializations go here
    img_folder = "./trainset/"
    mask_folder = "./data/red_masks/"
    train_img_folder = "./data/train/orig/"
    train_mask_folder = "./data/train/masks/"
    val_img_folder = "./data/val/orig/"
    val_mask_folder = "./data/val/masks/"

    for file_name in os.listdir(img_folder):
        print("Processing Image with name: ", file_name)
        fn, ext = file_name.split(".")
        nu = int(fn)
        # read image
        img = plt.imread(os.path.join(img_folder, file_name))
        mask = plt.imread(os.path.join(mask_folder, fn + ".png"))

        # STOP train
        if nu in stop_train or nu in other_train:
            plt.imsave(train_img_folder + fn + ".png", img)
            plt.imsave(train_mask_folder + fn + ".png", mask)

        elif nu in stop_val or nu in other_val:
            plt.imsave(val_img_folder + fn + ".png", img)
            plt.imsave(val_mask_folder + fn + ".png", mask)
    print("Done!")


if __name__ == "__main__":
    main()
