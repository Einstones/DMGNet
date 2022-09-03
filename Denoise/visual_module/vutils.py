import numpy as np
import math
from torchvision.utils import make_grid
from skimage import img_as_float32 as img_as_float
import cv2
import torch
# from pytorch_wavelets import DWTForward, DWTInverse


def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    """
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    """
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)


def load_img(filepath):
    img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
    return img


def save_img(img, img_path, mode='RGB'):
    img = tensor2img(img)
    cv2.imwrite(img_path, img)


# if __name__ == '__main__':
#     img = load_img('44_56.PNG')
#     img = torch.from_numpy(img_as_float(img))
#     img = torch.transpose(img, 2, 1)
#     img = torch.transpose(img, 1, 0).unsqueeze(0)
#     # img_r = img[:, 0, :, :]
#     # img_g = img[:, 1, :, :]
#     # img_b = img[:, 2, :, :]
#
#     xfm = DWTForward(J=1, wave='haar', mode='zero')
#     xin = DWTInverse(wave='haar', mode='zero')
#     img_l, img_h = xfm(img)
#     img_inverse = xin((img_l, img_h))
#     img_l = tensor2img(img_l.squeeze(0))
#     img_h0 = tensor2img(img_h[0].squeeze(0))
#     # img_h1 = tensor2img(img_h[1].squeeze(0))
#     # img_h2 = tensor2img(img_h[2].squeeze(0))
#     print(img_l)
#     save_img(img_l, 'haar_low.PNG')
#     save_img(img_h0, 'haar_h0.PNG')
#     # save_img(img_h1, 'haar_h1.PNG')
#     # save_img(img_h2, 'haar_h2.PNG')
#     print(img)





