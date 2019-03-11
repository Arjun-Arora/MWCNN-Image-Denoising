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
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import matplotlib.pyplot as plt

from numpy.lib.stride_tricks import as_strided


def show_images(images, cols=1, titles=None):
    """Display a list of images in a single figure with matplotlib.

    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.

    cols (Default = 1): Number of columns in figure (number of rows is
                        set to np.ceil(n_images/float(cols))).

    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert ((titles is None) or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images / float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()


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
def load_patches(patches_path=None,Train=True,patch_sz =(240,240),n=-1 ):
    """
    :param patches_path: path to patches of images. If path is None, then load images and create patches from scratch
    :param Train: whether loading patches for train or test. This is just in case we need to create patches from scratch and want to split these patches up
    :param n: numbe of images from original datast to load
    :return: (concatenated patches, path to patches)
    """
    if patches_path is None:
        if Train:
            imgs = load_imgs("./data/Train/",n=n)
            patches_path = "./data/patches/"
        else:
            imgs = load_imgs("./data/Test/",n=n)
            patches_path = "./data/patches/"

        patches = []
        print("making patches")
        with tqdm(total = len(imgs)) as pbar:
            for img in imgs:
                patch = image.extract_patches_2d(img,patch_sz,max_patches=24)
                for j in range(patch.shape[0]):
                    patches.append(patch[j,:,:])
                pbar.update(1)
        print("completed making patches!")
        if Train and not os.path.isdir(patches_path):
            os.makedirs(patches_path)
        elif not Train and not os.path.isdir(patches_path):
            os.makedirs(patches_path)
        print("writing patches")
        with tqdm(total=len(patches)) as pbar:
            for i in range(len(patches)):
                fname = patches_path + str(i)+".png"
                sk.io.imsave(fname,patches[i])
                pbar.update(1)
        print("finished writing patches to directory!")
        patches = np.array(patches)

    else:
        print("loading patches from patches directory")
        patches = []
        file_paths = patches_path + "*.*"
        with tqdm(total=len(glob.glob(file_paths)[:n])) as pbar:
            for filename in glob.glob(file_paths)[:n]:
                # print(filename)
                patch = plt.imread(filename)
                patch = sk.img_as_float(patch)
                patches.append(patch)
                # print("shape of current image: {}".format(imgs_list[-1].shape))
                # plt.show(patch)
                # plt.show()
                pbar.update(1)
        print("completed loading patches from directory!")
        patches = np.array(patches)
    return patches, patches_path

class patchesDataset(Dataset):
    def __init__(self, patches_path=None, patch_sz=(240,240),noise_level=15,noise_type='gaussian',n=-1):
        """
        :param patches_path: path to patches
        :param patch_sz: size of patches to load
        :param noise_level: level of noise
        :param n: how many patches to load if patches_path is filled, otherwise n original images to load (approx ~24 * n)
        """
        self.patches_target,patches_path = load_patches(patches_path=patches_path, Train=True, patch_sz=patch_sz,n=n)
        print(self.patches_target.shape)
        self.patches_target = self.patches_target[:,np.newaxis,:,:]
        # noise = np.random.normal(np.mean(self.patches_target,axis=0), noise_level,(n,patch_sz[0],patch_sz[1]))
        # noise = np.random.normal(0, noise_level, (n, patch_sz[0], patch_sz[1]))
        # print(noise)
        # print(self.patches_target.shape)
        # print(noise/255)
        print("noise_type: ",noise_type )
        if noise_type == 'gaussian':
            std_dev = noise_level
        elif noise_type == 'mixture': #mixture gaussian and poisson shot noise based on pixel values
            std_dev = np.sqrt(np.power(noise_level,2) + np.power(self.patches_target,2))

        self.patches_noisy = self.patches_target + std_dev / 255 * np.random.randn(*patch_sz).astype('f')
        print("shape of target: {} shape of noisy: {}".format(self.patches_noisy.shape,self.patches_target.shape))


        # print(np.mean(self.patches_target[0,:,:]))
        rand_idx = np.random.randint(0,self.patches_target.shape[0])
        sample_target = self.patches_target[rand_idx,0,:,:]
        sample_noise = self.patches_noisy[rand_idx,0,:,:]

        # print(sample_noise)


        show_images([sample_noise,sample_target], cols=1, titles=['Noisy image', 'Target image'])

    def __len__(self):
        return self.patches_noisy.shape[0]

    def __getitem__(self, idx):

        patch_target = self.patches_target[idx,:,:]
        patch_noisy = self.patches_noisy[idx,:,:]
        # img_name = os.path.join(self.root_dir,
        #                         self.landmarks_frame.iloc[idx, 0])
        # image = io.imread(img_name)
        # landmarks = self.landmarks_frame.iloc[idx, 1:].as_matrix()
        # landmarks = landmarks.astype('float').reshape(-1, 2)
        # sample = {'image': image, 'landmarks': landmarks}

        # if self.transform:
        #     sample = self.transform(sample)

        sample = {'target':patch_target, 'input':patch_noisy}
        return sample


def make_plots(experiment_dir,epochs = 20):
    """
    :param experiment_dir: directory where .npy files are kept
    :param epochs: number of epochs for each plot for val and train
    :return: saves plots for Val and Train Loss  vs. Epoch and Train and Val PSNR vs. Epoch
    """
    train_PSNRs_path = experiment_dir + '/train_PSNRs.npy'
    train_losses_path = experiment_dir + '/train_losses.npy'
    val_PSNRs_path = experiment_dir + '/val_PSNRs.npy'
    val_losses_path = experiment_dir + '/val_losses.npy'

    train_PSNRs = np.load(train_PSNRs_path)
    val_PSNRs = np.load(val_PSNRs_path)

    train_losses = np.load(train_losses_path)
    val_losses = np.load(val_losses_path)

    x = np.linspace(1,epochs,len(train_PSNRs))
    loss_fig = plt.figure()
    plt.xlabel('epochs')
    plt.ylabel('Log MSE loss')
    plt.title('Log MSE per Epoch ')
    train_plt, = plt.plot(x,np.log(train_losses),alpha=0.5)
    val_plt, = plt.plot(x,np.log(val_losses),alpha=0.5)
    plt.legend([train_plt, val_plt],['train_loss','val_loss'])

    loss_fig.savefig(experiment_dir + '/loss_vs_epoch.png')

    PSNR_fig = plt.figure()
    plt.xlabel('epochs')
    plt.ylabel("PSNR (dB)")
    plt.title('PSNR per Epoch')
    train_PSNR_plt, = plt.plot(x, train_PSNRs,alpha=0.5)
    val_PSNR_plt, = plt.plot(x, val_PSNRs,alpha=0.5)
    plt.legend([train_PSNR_plt, val_PSNR_plt], ['train_PSNR', 'val_PSNR'])

    PSNR_fig.savefig(experiment_dir + '/PSNR_vs_epoch.png')








# load_imgs("./data/Train/")
# load_patches("./data/patches/")
if __name__ == "__main__":
    # patchesDataset(patches_path="./data/patches/",n=1000,noise_type='mixture')
    make_plots('./experiments/baseline')
# patchesDataset(patches_path=None,n=-1)
# patchesDataset(patches_path=None,n=-1)
