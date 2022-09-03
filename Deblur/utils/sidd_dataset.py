import os
import cv2
import numpy as np
from skimage import img_as_float32 as img_as_float
import random
from scipy.io import loadmat
import torch.utils.data as u_data
import torch
def data_augmentation(image, mode):
    """
    Performs data augmentation of the input image
    Input:
        image: a cv2 (OpenCV) image
        mode: int. Choice of transformation to apply to the image
                0 - no transformation
                1 - flip up and down
                2 - rotate counterwise 90 degree
                3 - rotate 90 degree and flip up and down
                4 - rotate 180 degree
                5 - rotate 180 degree and flip
                6 - rotate 270 degree
                7 - rotate 270 degree and flip
    """
    if mode == 0:
        # original
        out = image
    elif mode == 1:
        # flip up and down
        out = np.flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(image)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(image)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(image, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(image, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(image, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(image, k=3)
        out = np.flipud(out)
    else:
        raise Exception('Invalid choice of image transformation')

    return out


def random_augmentation(*args):
    out = []
    if random.randint(0, 1) == 1:
        flag_aug = random.randint(1, 7)
        for data in args:
            out.append(data_augmentation(data, flag_aug).copy())
    else:
        for data in args:
            out.append(data)
    return out


def load_img(filepath):
    img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
    return img

def is_png_file(filename):
    return any(filename.endswith(extension) for extension in ['.PNG', '.png'])

class SIDDData(u_data.Dataset):
    def __init__(self, path, patch_size):
        """
        Args:
            h5_path (str): path of the hdf5 file
            length (int): length of Datasets
        """
        super(SIDDData, self).__init__()
        # 需要进行额外的排序才能将noisy和gt图像实现对齐
        clean_files = sorted(os.listdir(os.path.join(path, 'patches_train', 'target_crops')))
        noisy_files = sorted(os.listdir(os.path.join(path, 'patches_train', 'input_crops')))
        # training data
        self.clean_filenames = [os.path.join(path, 'patches_train', 'target_crops', x) for x in clean_files if is_png_file(x)]
        self.noisy_filenames = [os.path.join(path, 'patches_train', 'input_crops', x) for x in noisy_files if is_png_file(x)]
        self.num_images = len(self.clean_filenames)
        self.pch_size = patch_size

    def __len__(self):
        return self.num_images

    def crop_patch(self, n_img, gt_img):
        H, W, C = n_img.shape
        # minus the bayer patter channel
        ind_H = random.randint(0, H - self.pch_size)
        ind_W = random.randint(0, W - self.pch_size)
        #im_noisy = n_img[self.pch_size:ind_H, 0:self.pch_size, :]
        #im_gt = gt_img[self.pch_size:ind_H, 0:self.pch_size, :]
        im_noisy = n_img[ind_H:ind_H + self.pch_size, ind_W:ind_W + self.pch_size, :]
        im_gt = gt_img[ind_H:ind_H + self.pch_size, ind_W:ind_W + self.pch_size, :]
        return im_noisy, im_gt

    def __getitem__(self, index):
        index = index % len(self.clean_filenames)
        noisy_img = load_img(self.noisy_filenames[index])
        gt_img = load_img(self.clean_filenames[index])
        if noisy_img.shape[0] > self.pch_size:
            noisy_img, gt_img = self.crop_patch(noisy_img, gt_img)  # 随机裁剪
        gt_img = img_as_float(gt_img)
        noisy_img = img_as_float(noisy_img)
        noisy_img, gt_img = random_augmentation(noisy_img, gt_img)
        gt_img = gt_img.transpose((2, 0, 1))
        noisy_img = noisy_img.transpose((2, 0, 1))
        return noisy_img, gt_img


class SIDDValData(u_data.Dataset):
    def __init__(self, path):
        super(SIDDValData, self).__init__()
        # 需要进行额外的排序才能将noisy和gt图像实现对齐
        clean_files = sorted(os.listdir(os.path.join(path, 'patches_test', 'target_crops')))
        noisy_files = sorted(os.listdir(os.path.join(path, 'patches_test', 'input_crops')))
        # training data
        self.clean_filenames = [os.path.join(path, 'patches_test', 'target_crops', x) for x in clean_files if is_png_file(x)]
        self.noisy_filenames = [os.path.join(path, 'patches_test', 'input_crops', x) for x in noisy_files if is_png_file(x)]
        self.num_images = len(self.clean_filenames)
        self.pch_size = 128

    def __len__(self):
        return self.num_images

    def crop_patch(self, n_img, gt_img):
        H, W, C = n_img.shape
        random.seed(0)
        # minus the bayer patter channel
        ind_H = random.randint(0, H - self.pch_size)
        ind_W = random.randint(0, W - self.pch_size)
        #im_noisy = n_img[self.pch_size:ind_H, 0:self.pch_size, :]
        #im_gt = gt_img[self.pch_size:ind_H, 0:self.pch_size, :]
        im_noisy = n_img[ind_H:ind_H + self.pch_size, ind_W:ind_W + self.pch_size, :]
        im_gt = gt_img[ind_H:ind_H + self.pch_size, ind_W:ind_W + self.pch_size, :]
        return im_noisy, im_gt

    def __getitem__(self, index):
        index = index % len(self.clean_filenames)
        noisy_img = load_img(self.noisy_filenames[index])
        gt_img = load_img(self.clean_filenames[index])
        if noisy_img.shape[0] > self.pch_size:
            noisy_img, gt_img = self.crop_patch(noisy_img, gt_img)  # 随机裁剪
        gt_img = img_as_float(gt_img)
        noisy_img = img_as_float(noisy_img)
        gt_img = gt_img.transpose((2, 0, 1))
        noisy_img = noisy_img.transpose((2, 0, 1))
        return noisy_img, gt_img
