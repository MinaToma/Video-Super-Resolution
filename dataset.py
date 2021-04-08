from os import listdir
import random
import torch
import numpy as np
import torch.utils.data as data
from torchvision.transforms import Compose
import cv2

def get_training_set(opt):
    return REDSTrainDataset(opt)

class DataAug(object):
    def __call__(self, sample):
        hflip = random.random() < 0.5
        vflip = random.random() < 0.5
        rot90 = random.random() < 0.5

        lr, hr = sample['lr'], sample['hr']
        num, r, c, ch = lr.shape

        if hflip:
            cv2.flip(hr, 1, hr)
            for idx in range(num):
                cv2.flip(lr[idx, :, :, :], 1, lr[idx, :, :, :])
        if vflip:
            hr = hr[::-1, :, :]
            cv2.flip(hr, 0, hr)
            for idx in range(num):
                cv2.flip(lr[idx, :, :, :], 0, lr[idx, :, :, :])
        if rot90:
            hr = hr.transpose(1, 0, 2)
            lr = lr.transpose(0, 2, 1, 3)

        return {'lr': lr, 'hr': hr}

class ToTensor(object):
    def __call__(self, sample):
        lr, hr = sample['lr'], sample['hr']
        lr = torch.from_numpy(lr.copy()).float()
        hr = torch.from_numpy(hr.copy()).float()
        return {'lr': lr.permute(0, 3, 1, 2), 'hr': hr.permute(2, 0, 1)}

class REDSTrainDataset(data.Dataset):
    def __init__(self, opt):
        self.dir_HR = opt.gt_dir
        self.dir_LR = opt.lr_dir
        self.dir_lis = sorted(listdir(self.dir_HR))
        self.img_list = sorted(listdir('{}/{}/'.format(self.dir_HR, self.dir_lis[0])))
        self.frame_num = opt.frame
        self.half_frame_num = int(self.frame_num / 2)
        self.transform = Compose([ToTensor()])
        self.scale = 4
        self.len = len(self.dir_lis) * len(self.img_list)

    def __len__(self):
        return self.len

    def __getitem__(self, _):
        folder_index = random.randint(0, len(self.dir_lis) - 1)
        folder_name = self.dir_lis[folder_index]
        hr_folder_path = '{}/{}'.format(self.dir_HR, folder_name)

        center_index = random.randint(self.half_frame_num, len(self.img_list) - (self.half_frame_num + 1))

        frames_hr_name = '{}/{}'.format(hr_folder_path, self.img_list[center_index])
        frames_hr = cv2.imread(frames_hr_name)
        h, w, ch = frames_hr.shape

        frames_lr = np.zeros((self.frame_num, int(h / self.scale), int(w / self.scale), ch))
        for j in range(center_index - self.half_frame_num, center_index + self.half_frame_num + 1):
            i = j - center_index + self.half_frame_num
            frames_lr_name = '{}/{}/{}'.format(self.dir_LR, folder_name, self.img_list[j])
            img = cv2.imread(frames_lr_name)
            frames_lr[i, :, :, :] = img  # t h w c

        sample = {'lr': frames_lr, 'hr': frames_hr}
        sample = self.transform(sample)

        return sample['lr'], sample['hr']
