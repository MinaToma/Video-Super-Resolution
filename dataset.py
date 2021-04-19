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

class Patcher(object):
    def __call__(self, sample):
      img_lqs, img_gts, gt_patch_size = sample['lr'], sample['hr'], sample['patch_size']

      scale = 4
      num, h_lq, w_lq, ch = img_lqs.shape
      h_gt, w_gt, _ = img_gts.shape
      lq_patch_size = gt_patch_size // scale

      if h_gt != h_lq * scale or w_gt != w_lq * scale:
          raise ValueError(
              f'Scale mismatches. GT ({h_gt}, {w_gt}) is not {scale}x ',
              f'multiplication of LQ ({h_lq}, {w_lq}).')
      if h_lq < lq_patch_size or w_lq < lq_patch_size:
          raise ValueError(f'LQ ({h_lq}, {w_lq}) is smaller than patch size '
                          f'({lq_patch_size}, {lq_patch_size}). '
                          f'Please remove {gt_path}.')

      # randomly choose top and left coordinates for lq patch
      top = random.randint(0, h_lq - lq_patch_size)
      left = random.randint(0, w_lq - lq_patch_size)

      # crop lq patch
      frames_lr = np.zeros((num, lq_patch_size, lq_patch_size, ch))
      for idx in range(num):
          frames_lr[idx, :, :, :] = img_lqs[idx, top:top + lq_patch_size, left:left + lq_patch_size, ...]

      # crop corresponding gt patch
      top_gt, left_gt = int(top * scale), int(left * scale)
      img_gts = img_gts[top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size, ...]
       
      return {'lr': frames_lr, 'hr': img_gts}

class DataAug(object):
    def __call__(self, sample):
        hflip = random.random() < 0.5
        vflip = random.random() < 0.5
        rot90 = random.random() < 0.5

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
        
        if rot90:
            hr = hr.transpose(1, 0, 2)
            lr = lr.transpose(0, 2, 1, 3)
        
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
            self.transform = Compose([Patcher(), DataAug(), ToTensor()])
        else:
            self.transform = Compose([ToTensor()])
        self.patch_size = opt.patch_size
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

        sample = {'lr': frames_lr, 'hr': frames_hr, 'patch_size': self.patch_size}
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
            self.transform = Compose([Patcher(), DataAug(), ToTensor()])
        else:
            self.transform = Compose([ToTensor()])
        self.patch_size = opt.patch_size
        self.scale = 4
        self.len = len(self.folder_list)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        folder_name = self.folder_list[idx]
        hr_folder_path = '{}/{}'.format(self.dir_HR, folder_name)

        center_index = 4

        frames_hr_name = '{}/{}'.format(hr_folder_path, 'im' + str(center_index) + '.png')
        frames_hr = cv2.imread(frames_hr_name)
        h, w, ch = frames_hr.shape

        frames_lr = np.zeros((self.frame_num, int(h / self.scale), int(w / self.scale), ch))
        for j in range(center_index - self.half_frame_num, center_index + self.half_frame_num + 1):
            i = j - center_index + self.half_frame_num
            frames_lr_name = '{}/{}/{}'.format(self.dir_LR, folder_name, 'im' + str(j) + '.png')
            img = cv2.imread(frames_lr_name)
            frames_lr[i, :, :, :] = img  # t h w c

        sample = {'lr': frames_lr, 'hr': frames_hr, 'patch_size': self.patch_size}
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

        sample = {'lr': frames_lr, 'hr': frames_hr}
        sample = self.transform(sample)

        return sample['lr'], sample['hr']

class Vid4ValidationDataset():
    def __init__(self):
        self.dir_HR = '/content/drive/MyDrive/datasets/test/Vid4/GT'
        self.dir_LR = '/content/drive/MyDrive/datasets/test/Vid4/BIx4'
        alist = [line.rstrip() for line in open('/content/drive/MyDrive/datasets/test/Vid4/folder_list.txt')]
        self.folder_list = [x for x in alist]
        self.frame_num = 5
        self.half_frame_num = int(self.frame_num / 2)
        self.transform = Compose([ToTensor()])
        self.patch_size = 256
        self.scale = 4
        self.len = len(self.folder_list)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        folder_name = self.folder_list[idx]
        hr_folder_path = '{}/{}'.format(self.dir_HR, folder_name)

        center_index = idx

        frames_hr_name = '{}/{}'.format(hr_folder_path, str("%08d" % (center_index) + '.png'))
        print('frames_hr_name,=', frames_hr_name)
        frames_hr = cv2.imread(frames_hr_name)
        h, w, ch = frames_hr.shape

        frames_lr = np.zeros((self.frame_num, int(h / self.scale), int(w / self.scale), ch))
        for j in range(center_index - self.half_frame_num, center_index + self.half_frame_num + 1):
            i = j - center_index + self.half_frame_num
            frames_lr_name = '{}/{}/{}'.format(self.dir_LR, folder_name, 'im' + str(j) + '.png')
            img = cv2.imread(frames_lr_name)
            frames_lr[i, :, :, :] = img  # t h w c

        sample = {'lr': frames_lr, 'hr': frames_hr, 'patch_size': self.patch_size}
        sample = self.transform(sample)

        return sample['lr'], sample['hr']



class REDSValidationDataset():
    def __init__(self):
        self.dir_HR = '/content/drive/MyDrive/datasets/test/REDS4/GT'
        self.dir_LR = '/content/drive/MyDrive/datasets/test/REDS4/sharp_bicubic'
        alist = [line.rstrip() for line in open('/content/drive/MyDrive/datasets/test/REDS4/folder_list.txt')]
        self.folder_list = [x for x in alist]
        self.frame_num = 5
        self.half_frame_num = int(self.frame_num / 2)
        self.transform = Compose([ToTensor()])
        self.patch_size = 256
        self.scale = 4
        self.len = len(self.folder_list)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        folder_name = self.folder_list[idx]
        hr_folder_path = '{}/{}'.format(self.dir_HR, folder_name)

        center_index = idx

        frames_hr_name = '{}/{}'.format(hr_folder_path, str("%08d" % (center_index) + '.png'))
        frames_hr = cv2.imread(frames_hr_name)
        h, w, ch = frames_hr.shape

        frames_lr = np.zeros((self.frame_num, int(h / self.scale), int(w / self.scale), ch))
        for j in range(center_index - self.half_frame_num, center_index + self.half_frame_num + 1):
            i = j - center_index + self.half_frame_num
            frames_lr_name = '{}/{}/{}'.format(self.dir_LR, folder_name, 'im' + str(j) + '.png')
            img = cv2.imread(frames_lr_name)
            frames_lr[i, :, :, :] = img  # t h w c

        sample = {'lr': frames_lr, 'hr': frames_hr, 'patch_size': self.patch_size}
        sample = self.transform(sample)

        return sample['lr'], sample['hr']

