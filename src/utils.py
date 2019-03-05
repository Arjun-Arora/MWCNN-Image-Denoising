import numpy as np
import torch
import csv
import skimage as sk
import os
import sys
import pkgutil
# search_path = ['.'] # set to None to see all modules importable from sys.path
# all_modules = [x[1] for x in pkgutil.iter_modules(path=search_path)]
# print(all_modules)
# print(sys.path)
import glob
from tqdm import tqdm
from sklearn.feature_extraction import image
from numpy.lib.stride_tricks import as_strided


def load_imgs(directory_path,n = -1):
    """
    :param directory_path: path to directory containing images
    :param n: number of images from dataset to load (-1 for all)
    :return: list of images in grayscale
    """
    imgs_list = []
    file_paths  = directory_path + "*.*"
    # print(file_paths)
    print("loading images... ")
    with tqdm(total = len(glob.glob(file_paths)[:n])) as pbar:
        for filename in glob.glob(file_paths)[:n]:
            # print(filename)
            imgs_list.append(sk.io.imread(filename,as_gray=True))
            # print(imgs_list[-1].shape)
            # print("shape of current image: {}".format(imgs_list[-1].shape))
            pbar.update(1)
    print("completed loading images!")
    return imgs_list
def load_patches(patches_path=None,Train=True,patch_sz =(240,240) ):
    """
    :param patches_path: path to patches of images. If path is None, then load images and create patches from scratch
    :param Train: whether loading patches for train or test. This is just in case we need to create patches from scratch and want to split these patches up
    :return: (concatenated patches, path to patches)
    """
    if patches_path is None:
        if Train:
            imgs = load_imgs("./data/Train/")
            patches_path = "./data/patches_Train/"
        else:
            imgs = load_imgs("./data/Test/")
            patches_path = "./data/patches_Test/"

        patches = []
        print("loading patches")
        with tqdm(total = len(imgs)) as pbar:
            for img in imgs:
                patch = image.extract_patches_2d(img,patch_sz,max_patches=24)
                # patch = as_strided(img, shape=(img.shape[0] - (patch_sz[0]-1), img.shape[1] - (patch_sz[1]-1), patch_sz[0], patch_sz[1]),
                #                          strides=img.strides + img.strides, writeable=True)
                # for patch_img in patch.reshape(-1, patch_sz[0],patch_sz[1]):
                for j in range(patch.shape[0]):
                    patches.append(patch[j,:,:])
                pbar.update(1)
        print("completed loading patches!")
        if Train and not os.path.isdir(patches_path):
            os.makedirs(patches_path)
        elif not Train and not patches_path:
            os.makedirs(patches_path)
        print("writing patches")
        with tqdm(total=len(patches)) as pbar:
            for i in range(len(patches)):
                fname = patches_path + str(i)+".png"
                sk.io.imsave(fname,patches[i])
                pbar.update(1)
        print("finished writing patches to directory!")
        patches = np.vstack(patches)
    else:
        print("loading patches from patches directory")
        patches = []
        file_paths = patches_path + "*.*"
        with tqdm(total=len(glob.glob(file_paths))) as pbar:
            for filename in glob.glob(file_paths):
                # print(filename)
                patches.append(sk.io.imread(filename, as_gray=True))
                # print("shape of current image: {}".format(imgs_list[-1].shape))
                pbar.update(1)
        print("completed loading patches from directory!")
        patches = np.vstack(patches)
    return patches, patches_path


# load_imgs("./data/Train/")
load_patches("./data/patches_Train/")