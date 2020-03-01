from matplotlib import pyplot as plt
from roipoly import RoiPoly
import os

import numpy as np

directory = './trainset'

savingdirectory = './maskedimages/'

def normalize_image(x):
    x = (x)/255
    return x

for filename in os.listdir(directory):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        imagepath = os.path.join(directory, filename)
        img = plt.imread(imagepath)
        # resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

        img = img.astype(np.float32)

        img = normalize_image(img)

        fig = plt.figure()
        plt.imshow(img)
        # plt.colorbar()
        plt.title("left click: line segment         right click or double click: close region")
        plt.show(block=False)

        roi1 = RoiPoly(color='b', fig=fig)

        fig = plt.figure()
        plt.imshow(img)
        # plt.colorbar()
        plt.title("left click: line segment         right click or double click: close region")
        plt.show(block=False)

        roi2 = RoiPoly(color='b', fig=fig)

        fig = plt.figure()
        plt.imshow(img)
        # plt.colorbar()
        plt.title("left click: line segment         right click or double click: close region")
        plt.show(block=False)

        roi3 = RoiPoly(color='b', fig=fig)

        totalmasked = roi1.get_mask(img[:, :, 0]) + roi2.get_mask(img[:, :, 0]) + roi3.get_mask(img[:, :, 0])


        fig = plt.figure()
        plt.imshow(totalmasked, cmap=plt.cm.gray)
        plt.show()

        savefilename, extension = filename.split('.')

        np.save(savingdirectory + savefilename, totalmasked)











