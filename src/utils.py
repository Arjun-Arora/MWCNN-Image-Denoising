import numpy as np
import torch
import csv
import skimage as sk
import os
import sys
sys.path.append("./src/models/")
sys.path.append("./src/")
import glob
from tqdm import tqdm
from sklearn.feature_extraction import image
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from MWU_CNN import MW_Unet
import torch.nn as nn
from skimage.measure import compare_ssim as ssim
import torchvision


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
def load_patches(patches_path=None,Train=True,patch_sz =(240,240),n=24000):
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
        print(patch_sz)
        patches = np.zeros((n,patch_sz[0],patch_sz[1]))
        file_paths = patches_path + "*.*"
        ctr = 0
        with tqdm(total=len(glob.glob(file_paths)[:n])) as pbar:
            for filename in glob.glob(file_paths)[:n]:
                # print(filename)
                patch = plt.imread(filename)
                patch = sk.img_as_float(patch)
                # patches.append(patch)
                patches[ctr,:,:] = patch
                # print("shape of current image: {}".format(imgs_list[-1].shape))
                # plt.show(patch)
                # plt.show()
                pbar.update(1)
                ctr += 1
        print("completed loading patches from directory!")
        # patches = np.array(patches)
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
        elif noise_type == 's&p':
            N,C,H,W  = self.patches_target.shape

            s_vs_p = 0.5
            amount = 0.04
            self.patches_noisy = np.copy(self.patches_target)
            num_salt = np.ceil(amount * self.patches_target.size * s_vs_p)
            coords = [np.random.randint(0, i, int(num_salt)) for i in self.patches_target.shape]
            self.patches_noisy[coords] = 1

            # Pepper mode
            num_pepper = np.ceil(amount * self.patches_target.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i, int(num_pepper)) for i in self.patches_target.shape]
            self.patches_noisy[coords] = 0


        if noise_type == 'gaussian' or noise_type == 'mixture':
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


def imshow(img):
    #     print('Image device and mean')
    #     print(img.device)
    #     print(img.mean())
    output_image = img.cpu().numpy().transpose((1, 2, 0))
    npimg = output_image.astype(np.uint8)
    #     print('Mean of image: {}'.format(npimg.mean()))
    # format H,W,C
    plt.imshow(npimg)
    plt.show()

def save_image(img, path):
    #     img = img * 0.5 + 0.5
    torchvision.utils.save_image(img, path)


def backprop(optimizer, model_output, target):
    optimizer.zero_grad()
    loss_fn = nn.MSELoss()
    loss = loss_fn(model_output, target)
    loss.backward()
    optimizer.step()
    return loss


def get_PSNR(model_output, target):
    I_hat = model_output.cpu().detach().numpy()
    I = target.cpu().detach().numpy()
    mse = (np.square(I - I_hat)).mean(axis=None)
    PSNR = 10 * np.log10(1.0 / mse)
    return PSNR


def get_SSIM(model_output, target):
    I_hat = model_output.cpu().detach().numpy()
    I = target.cpu().detach().numpy()
    N, C, H, W = I_hat.shape
    ssim_out = []
    for i in range(N):
        img = I[i, 0, :, :]
        img_noisy = I_hat[i, 0, :, :]
        ssim_out.append(ssim(img, img_noisy, data_range=img_noisy.max() - img_noisy.min()))
    return np.mean(ssim_out)


def train(args):
    """
    train model
    """

    ####################################### Initializing Model #######################################

    step = 0.01
    experiment_dir = args['--experiment_dir']
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print_every = int(args['--print_every'])
    num_epochs = int(args['--num_epochs'])
    save_every = int(args['--save_every'])
    save_path = str(args['--model_save_path'])
    batch_size = int(args['--batch_size'])
    train_data_path = args['--data_path']
    n = int(args['--n'])
    noise_type = args['--noise_type']
    train_split, val_split = args['--train_split'], args['--val_split']

    img_directory = args['--train_img_directory']

    model = MW_Unet(num_conv=3, in_ch=1)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=step)

    ######################################### Loading Data ##########################################
    dataset_total = patchesDataset(patches_path=train_data_path, noise_type=noise_type, n=n)

    train_max_idx = int(train_split * len(dataset_total))
    #     print(train_max_idx)
    dataset_train = torch.utils.data.Subset(dataset_total, range(0, train_max_idx))
    val_max_idx = train_max_idx + int(val_split * len(dataset_total))
    dataset_val = torch.utils.data.Subset(dataset_total, range(train_max_idx, val_max_idx))
    #     print(test_max_idx)

    #     dataset_test = torch.utils.data.Subset(dataset_total,range(val_max_idx,test_max_idx))

    dataloader_train = DataLoader(dataset_train, batch_size=batch_size)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size)

    print("length of train set: ", len(dataset_train))
    print("length of val set: ", len(dataset_val))
    #     print("length of test set: ",len(dataset_test))

    train_PSNRs = []
    train_losses = []
    val_PSNRs = []
    val_losses = []
    init_epoch = 0

    best_val_PSNR = 0.0
    try:
        for epoch in range(1, num_epochs + 1):
            # INITIATE dataloader_train
            print("epoch: ", epoch)
            with tqdm(total=len(dataloader_train)) as pbar:
                for index, sample in enumerate(dataloader_train):

                    model.train()

                    target, model_input = sample['target'], sample['input']
                    target = target.to(device)
                    model_input = model_input.to(device)

                    output = model.forward(model_input)

                    train_loss = backprop(optimizer, output, target)

                    train_PSNR = get_PSNR(output, target)

                    avg_val_PSNR = []
                    avg_val_loss = []
                    model.eval()
                    with torch.no_grad():
                        for val_index, val_sample in enumerate(dataloader_val):
                            target, model_input = val_sample['target'], val_sample['input']

                            target = target.to(device)
                            model_input = model_input.to(device)

                            output = model.forward(model_input)
                            loss_fn = nn.MSELoss()
                            loss_val = loss_fn(output, target)
                            PSNR = get_PSNR(output, target)
                            avg_val_PSNR.append(PSNR)
                            avg_val_loss.append(loss_val.cpu().detach().numpy())
                    avg_val_PSNR = np.mean(avg_val_PSNR)
                    avg_val_loss = np.mean(avg_val_loss)
                    val_PSNRs.append(avg_val_PSNR)
                    val_losses.append(avg_val_loss)

                    train_losses.append(train_loss.cpu().detach().numpy())
                    train_PSNRs.append(train_PSNR)

                    if index == len(dataloader_train) - 1:
                        img_grid = output.data
                        img_grid = torchvision.utils.make_grid(img_grid)
                        real_grid = target.data
                        real_grid = torchvision.utils.make_grid(real_grid)
                        directory = img_directory
                        input_grid = model_input.data
                        input_grid = torchvision.utils.make_grid(input_grid)
                        save_image(input_grid, '{}train_input_img.png'.format(directory))
                        save_image(img_grid, '{}train_img_{}.png'.format(directory, epoch))
                        save_image(real_grid, '{}train_real_img_{}.png'.format(directory, epoch))
                        print('train images')
                        imshow(input_grid)
                        imshow(img_grid)
                        imshow(real_grid)

                    pbar.update(1)
                if epoch % print_every == 0:
                    print("Epoch: {}, Loss: {}, Training PSNR: {}".format(epoch, train_loss, train_PSNR))
                    print("Epoch: {}, Avg Val Loss: {},Avg Val PSNR: {}".format(epoch, avg_val_loss, avg_val_PSNR))
                if epoch % save_every == 0 and best_val_PSNR < avg_val_PSNR:
                    best_val_PSNR = avg_val_PSNR
                    print("new best Avg Val PSNR: {}".format(best_val_PSNR))
                    print("Saving model to {}".format(save_path))
                    torch.save({'epoch': epoch,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'loss': train_loss},
                               save_path)
                    print("Saved successfully to {}".format(save_path))


    except KeyboardInterrupt:
        print("Training interupted...")
        print("Saving model to {}".format(save_path))
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': train_loss},
                   save_path)
        print("Saved successfully to {}".format(save_path))

    print("Training completed.")

    return (train_losses, train_PSNRs, val_losses, val_PSNRs, best_val_PSNR)


def Test(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    experiment_dir = args['--experiment_dir']
    print_every = int(args['--print_every'])
    num_epochs = int(args['--num_epochs'])
    save_every = int(args['--save_every'])
    batch_size = int(args['--batch_size'])
    n = int(args['--n'])
    train_split, val_split, test_split = args['--train_split'], args['--val_split'], args['--test_split']
    data_path = args['--data_path']
    model_path = args['--model_save_path']
    img_directory = args['--test_img_directory']
    noise_type = args['--noise_type']
    ################################ Load Data ###################################################
    dataset_total = patchesDataset(patches_path=data_path, noise_type=noise_type, n=n)
    train_max_idx = int(train_split * len(dataset_total))
    val_max_idx = train_max_idx + int(val_split * len(dataset_total))
    test_max_idx = val_max_idx + int(test_split * len(dataset_total))
    dataset_test = torch.utils.data.Subset(dataset_total, range(val_max_idx, test_max_idx))
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size)

    #     print(len(dataset_test))
    load_path = model_path

    model = MW_Unet(num_conv=3, in_ch=1)
    model.to(device)

    if (load_path != None):
        if torch.cuda.is_available():
            print("Loading model from {}".format(load_path))
            checkpoint = torch.load(load_path, )
            model.load_state_dict(checkpoint['model_state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # epoch = checkpoint['epoch']
            print("Model successfully loaded from {}".format(load_path))
        else:
            print("Loading model from {}".format(load_path))
            checkpoint = torch.load(load_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # epoch = checkpoint['epoch']
            print("Model successfully loaded from {}".format(load_path))

    model.eval()

    print("Testing...")

    test_loss = []
    test_PSNR = []
    test_SSIM = []

    with tqdm(total=len(dataloader_test)) as pbar:
        with torch.no_grad():
            for index, sample in enumerate(dataloader_test):
                target, model_input = sample['target'].type(torch.FloatTensor), sample['input'].type(torch.FloatTensor)
                target = target.to(device)
                model_input = model_input.to(device)

                output = model.forward(model_input)

                loss_fn = nn.MSELoss()

                loss = loss_fn(output, target)
                PSNR = get_PSNR(output, target)
                SSIM = get_SSIM(output, target)

                test_loss.append(loss.cpu().numpy())
                test_PSNR.append(PSNR)
                test_SSIM.append(SSIM)

                if index == len(dataloader_test) - 1:
                    img_grid = output.data
                    img_grid = torchvision.utils.make_grid(img_grid)
                    directory = img_directory
                    save_image(img_grid, '{}test_img.png'.format(directory))
                    input_grid = model_input.data
                    input_grid = torchvision.utils.make_grid(input_grid)
                    save_image(input_grid, '{}test_input_img.png'.format(directory))
                    real_grid = target.data
                    real_grid = torchvision.utils.make_grid(real_grid)
                    save_image(real_grid, '{}test_real_img.png'.format(directory))
                    print('test images')
                    print("Input")
                    imshow(input_grid)
                    print("Output")
                    imshow(img_grid)
                    print("Real")
                    imshow(real_grid)
                    print("Images saved to {}".format(directory))
                pbar.update(1)

    test_loss, test_PSNR, test_SSIM = np.mean(np.array(test_loss)), np.mean(np.array(test_PSNR)), np.mean(
        np.array(test_SSIM))

    str_to_save = "Test_loss: " + str(test_loss) + " , Test PSNR: " + str(test_PSNR) + ", Test SSIM: " + str(test_SSIM)

    with open(experiment_dir + "/test_results.txt", 'a') as test_writer:
        test_writer.write(str_to_save + "\n")

    return (test_loss, test_PSNR, test_SSIM)


# load_imgs("./data/Train/")
# load_patches("./data/patches/")
if __name__ == "__main__":
    # patchesDataset(patches_path="./data/patches/",n=24000,noise_type='s&p')
    make_plots('./experiments/baseline')
    make_plots('./experiments/mixture')
    make_plots('./experiments/s&p')

# patchesDataset(patches_path=None,n=-1)
# patchesDataset(patches_path=None,n=-1)
