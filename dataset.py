from os import listdir
import random
import torch
import numpy as np
import torch.utils.data as data
from torchvision.transforms import Compose
import cv2

def get_training_set(opt):
    if opt.dataset_name == 'vemo90k':
        return Vemo90KTrainDataset(opt)
    return REDSTrainDataset(opt)

def get_test_set(opt):
    if opt.dataset_name == 'REDS':
        return REDSTestDataset(opt)
    return Vid4TestDataset(opt)

class DataAug(object):
    def __call__(self, sample):
        hflip = random.random() < 0.5
        vflip = random.random() < 0.5

        lr, hr = sample['lr'], sample['hr']
        num, r, c, ch = lr.shape

        if hflip:
            hr = hr[:, ::-1, :]
            for idx in range(num):
                lr[idx, :, :, :] = lr[idx, :, ::-1, :]
        if vflip:
            hr = hr[::-1, :, :]
            for idx in range(num):
                lr[idx, :, :, :] = lr[idx, ::-1, :, :]
        
        return {'lr': lr, 'hr': hr}

class ToTensor(object):
    def __call__(self, sample):
        lr, hr = sample['lr'] / 255, sample['hr'] / 255
        lr = torch.from_numpy(lr).float()
        hr = torch.from_numpy(hr).float()
        return {'lr': lr.permute(0, 3, 1, 2), 'hr': hr.permute(2, 0, 1)}

class REDSTrainDataset(data.Dataset):
    def __init__(self, opt):
        self.dir_HR = opt.gt_dir
        self.dir_LR = opt.lr_dir
        self.dir_lis = sorted(listdir(self.dir_HR))
        self.img_list = sorted(listdir('{}/{}/'.format(self.dir_HR, self.dir_lis[0])))
        self.frame_num = opt.frame
        self.half_frame_num = int(self.frame_num / 2)
        if opt.data_augmentation:
            self.transform = Compose([DataAug(), ToTensor()])
        else:
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

class REDSTestDataset(data.Dataset):
    def __init__(self, opt):
        self.dir_HR = opt.gt_dir + '/' + opt.clip_name
        self.dir_LR = opt.lr_dir + '/' + opt.clip_name
        self.img_list = sorted(listdir(self.dir_HR))
        self.frame_num = opt.frame
        self.half_frame_num = int(self.frame_num / 2)
        self.transform = Compose([ToTensor()])
        self.scale = 4
        self.len = len(self.img_list)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        frames_hr_name = '{}/{}'.format(self.dir_HR, self.img_list[idx])
        frames_hr = cv2.imread(frames_hr_name)
        h, w, ch = frames_hr.shape

        center_index = idx

        frames_lr = np.zeros((self.frame_num, int(h / self.scale), int(w / self.scale), ch))
        for j in range(center_index - self.half_frame_num, center_index + self.half_frame_num + 1):
            i = j - center_index + self.half_frame_num
            if j < 0: 
                j = 0
            if j >= len(self.img_list):
                j = len(self.img_list) - 1
            frames_lr_name = '{}/{}'.format(self.dir_LR, self.img_list[j])
            img = cv2.imread(frames_lr_name)
            frames_lr[i, :, :, :] = img  # t h w c

        sample = {'lr': frames_lr, 'hr': frames_hr}
        sample = self.transform(sample)

        return sample['lr'], sample['hr']

class Vemo90KTrainDataset(data.Dataset):
    def __init__(self, opt):
        self.dir_HR = opt.gt_dir
        self.dir_LR = opt.lr_dir
        alist = [line.rstrip() for line in open(opt.file_list)]
        self.folder_list = [x for x in alist]
        self.frame_num = opt.frame
        self.half_frame_num = int(self.frame_num / 2)
        if opt.data_augmentation:
            self.transform = Compose([DataAug(), ToTensor()])
        else:
            self.transform = Compose([ToTensor()])
        self.scale = 4
        self.len = len(self.folder_list)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        folder_name = self.folder_list[idx]
        hr_folder_path = '{}/{}'.format(self.dir_HR, folder_name)

        center_index = random.randint(self.half_frame_num, 7 - (self.half_frame_num + 1))

        frames_hr_name = '{}/{}'.format(hr_folder_path, 'im' + str(center_index) + '.png')
        frames_hr = cv2.imread(frames_hr_name)
        h, w, ch = frames_hr.shape

        frames_lr = np.zeros((self.frame_num, int(h / self.scale), int(w / self.scale), ch))
        for j in range(center_index - self.half_frame_num, center_index + self.half_frame_num + 1):
            i = j - center_index + self.half_frame_num
            frames_lr_name = '{}/{}/{}'.format(self.dir_LR, folder_name, 'im' + str(j) + '.png')
            img = cv2.imread(frames_lr_name)
            frames_lr[i, :, :, :] = img  # t h w c

        sample = {'lr': frames_lr, 'hr': frames_hr}
        sample = self.transform(sample)

        return sample['lr'], sample['hr']

class Vid4TestDataset(data.Dataset):
    def __init__(self, opt):
        self.dir_HR = opt.gt_dir + '/' + opt.clip_name
        self.dir_LR = opt.lr_dir + '/' + opt.clip_name
        self.img_list = sorted(listdir(self.dir_HR))
        self.frame_num = opt.frame
        self.half_frame_num = int(self.frame_num / 2)
        self.transform = Compose([ToTensor()])
        self.scale = 4
        self.len = len(self.img_list)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        frames_hr_name = '{}/{}'.format(self.dir_HR, self.img_list[idx])
        frames_hr = cv2.imread(frames_hr_name)
        h, w, ch = frames_hr.shape

        print(frames_hr.shape)
        center_index = idx

        frames_lr = np.zeros((self.frame_num, int(h / self.scale), int(w / self.scale), ch))
        for j in range(center_index - self.half_frame_num, center_index + self.half_frame_num + 1):
            i = j - center_index + self.half_frame_num
            if j < 0: 
                j = 0
            if j >= self.len:
                j = self.len - 1
            frames_lr_name = '{}/{}'.format(self.dir_LR, self.img_list[j])
            img = cv2.imread(frames_lr_name)
            frames_lr[i, :, :, :] = img  # t h w c
            print(img.shape)

        sample = {'lr': frames_lr, 'hr': frames_hr}
        sample = self.transform(sample)

        return sample['lr'], sample['hr']

