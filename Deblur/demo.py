import os
import cv2
import numpy as np

def load_img(filepath):
    img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB) 
    return img

def is_png_file(filename):
    return any(filename.endswith(extension) for extension in ['.PNG', '.png'])

path = "/disk/hyd/Restormer/Denoising/Datasets/Downloads/test/Set12"
clean_files = sorted(os.listdir(path))
clean_filenames = [os.path.join(path, x) for x in clean_files if is_png_file(x)]
len_img = len(clean_filenames)

for index in range(len_img):
    img = load_img(clean_filenames[index])
    print(img.shape)
    #img = cv2.cvtColor(cv2.imread(clean_filenames[index]), cv2.COLOR_BGR2RGB)
    #print(type(img))
    #if type(img) == np.array:
    #    print(img.size)
        #continue
    #else:
    #    continue
        #print(img.size, '--------', clean_filenames[index])
