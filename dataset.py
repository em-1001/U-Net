import os
import numpy as np

import torch
import torch.nn as nn

# reference : https://sd118687.tistory.com/11?category=549654
# tensor shape (1, x, y)
def mirroring_Extrapolate(img):
    # mirroring 92 pixel
    img = torch.from_numpy(img.transpose((2, 0, 1)))

    x = img.shape[1]
    y = img.shape[2]

    np_img = np.array(img)

    np_img = np_img[0]

    if x < 388:
        pad_x_left = (572 - x) / 2
        pad_x_right = (572 - x) / 2
    else:
        pad_x_left = 92
        pad_x_right = 388 - (x % 388) + 92

    if y < 388:
        pad_y_up = (572 - y) / 2
        pad_y_down = (572 - y) / 2
    else:
        pad_y_up = 92
        pad_y_down = 388 - (y % 388) + 92

    np_img = np.pad(np_img, ((pad_x_left, pad_x_right), (pad_y_up, pad_y_down)), 'reflect')

    np_img = np_img[:, :, np.newaxis]

    return np_img


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        lst_data = os.listdir(self.data_dir)

        lst_label = [f for f in lst_data if f.startswith('label')]
        lst_input = [f for f in lst_data if f.startswith('input')]

        lst_label.sort()
        lst_input.sort()

        self.lst_label = lst_label
        self.lst_input = lst_input

    def __len__(self):
        return len(self.lst_label)

    def __getitem__(self, index):
        label = np.load(os.path.join(self.data_dir, self.lst_label[index]))
        input = np.load(os.path.join(self.data_dir, self.lst_input[index]))
        label = label/255.0
        input = input/255.0

        #채널이 없는 경우 늘리기
        if label.ndim == 2:
            label = label[:, :, np.newaxis]
        if input.ndim == 2:
            input = input[:, :, np.newaxis]

        label = mirroring_Extrapolate(label)
        input = mirroring_Extrapolate(input)


        data = {'input' : input, 'label' : label}

        if self.transform:
            data = self.transform(data)

        return data
