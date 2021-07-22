import torch
import numpy as np
from metrics import *
import os
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataset import get_test_set

def make_data_loader(dataset_loader):
    return DataLoader(dataset=dataset_loader, num_workers=1, batch_size=1, shuffle=False)

def save_img(img, count, save_dir, epoch):
    save_img = img.cpu().data.squeeze().clamp(0, 1).numpy().transpose(1, 2, 0)
    save_fn = save_dir + '/' + str(epoch).zfill(3) + '_' + str(count).zfill(8) + '.png'
    cv2.imwrite(save_fn, save_img * 255, [cv2.IMWRITE_PNG_COMPRESSION, 0])

class OPT_MOCK:
    def __init__(self, gt_dir, lr_dir, clip_name, frame, dataset_name):
        self.gt_dir = gt_dir
        self.lr_dir = lr_dir
        self.clip_name = clip_name
        self.frame = frame
        self.dataset_name = dataset_name

class Validator:
    def __init__(self, opt, device):
        self.device = device
        self.save_dir = opt.save_dir + '/validation'
        if not os.path.exists(self.save_dir):
          os.makedirs(self.save_dir)

        self.vid4_clips = ['calendar', 'city', 'foliage' , 'walk']
        self.reds_clips = ['000', '011', '015', '020']
        
        self.validateVid4Loaders = []
        for clip in self.vid4_clips:
            opt_mock = OPT_MOCK('/content/drive/MyDrive/datasets/test/Vid4/GT', '/content/drive/MyDrive/datasets/test/Vid4/BIx4', clip, opt.frame, 'vid4')
            validate_set = get_test_set(opt_mock)
            self.validateVid4Loaders.append(make_data_loader(validate_set))
        
        self.validateREDSLoaders = []
        for clip in self.reds_clips:
            opt_mock = OPT_MOCK('/content/drive/MyDrive/datasets/test/REDS4/GT', '/content/drive/MyDrive/datasets/test/REDS4/sharp_bicubic', clip, opt.frame, 'REDS')
            validate_set = get_test_set(opt_mock)
            self.validateREDSLoaders.append(make_data_loader(validate_set))
    
    def validate(self, model, runningResults, epoch):
        print('===========> Started Validation')
        model.eval()

        avg_vid4_psnr = 0.0
        avg_vid4_ssim = 0.0

        avg_reds_psnr = 0.0
        avg_reds_ssim = 0.0

        print('======> Started Vid4 Validation')
        for i, datasetLoader in enumerate(self.validateVid4Loaders):
            psnr, ssim = self.validate_model(model, datasetLoader, epoch, i == 0)
            runningResults[self.vid4_clips[i] + '_PSNR'] = psnr
            runningResults[self.vid4_clips[i] + '_SSIM'] = ssim

            print(self.vid4_clips[i] + '_PSNR: ' + str(psnr))
            print(self.vid4_clips[i] + '_SSIM: ' + str(ssim))

            avg_vid4_psnr += psnr
            avg_vid4_ssim += ssim

        runningResults['Vid4_PSNR'] = avg_vid4_psnr / 4
        runningResults['Vid4_SSIM'] = avg_vid4_ssim / 4
        print('** Vid4_PSNR: %.20f' % (runningResults['Vid4_PSNR']))
        print('** Vid4_SSIM: %.20f' % (runningResults['Vid4_SSIM']))

        print()
        # print('('======> Started REDS Validation')
        # for i, datasetLoader in enumerate(self.validateREDSLoaders):
        #     psnr, ssim = self.validate_model(model, datasetLoader, epoch, i == 0)
        #     runningResults[self.reds_clips[i] + '_PSNR'] = psnr
        #     runningResults[self.reds_clips[i] + '_SSIM'] = ssim

        #     print(self.reds_clips[i] + '_PSNR: ' + str(psnr))
        #     print(self.reds_clips[i] + '_SSIM: ' + str(ssim))

        #     avg_reds_psnr += psnr
        #     avg_reds_ssim += ssim
        
        # runningResults['REDS_PSNR'] = avg_reds_psnr / 4
        # runningResults['REDS_SSIM'] = avg_reds_ssim / 4

        # print('REDS_PSNR: %.20f' % (runningResults['REDS_PSNR']))
        # print('REDS_SSIM: %.20f' % (runningResults['REDS_SSIM']))
        # print()

    def validate_model(self, model, dataset_loader, epoch, is_save_img = False):
        avg_psnr_predicted = 0.0
        avg_ssim_predicted = 0.0

        count = 0
        for batch in dataset_loader:
            input, target = batch[0], batch[1]
            with torch.no_grad():
                input = Variable(input)
                target = Variable(target)
                input = input.to(self.device)
                target = target.to(self.device)
                prediction = model(input)

            if count == 10 and is_save_img:
              save_img(prediction, count, self.save_dir, epoch)

            prediction = prediction.cpu()
            prediction = prediction.data[0].numpy().astype(np.float32)
            prediction = prediction * 255.

            target = target.cpu().squeeze().numpy().astype(np.float32)
            target = target * 255.
            psnr_predicted = PSNR(prediction, target)
            ssim_predicted = SSIM(prediction, target)

            avg_psnr_predicted += psnr_predicted
            avg_ssim_predicted += ssim_predicted          
            count += 1

        avg_psnr_predicted /= count
        avg_ssim_predicted /= count
        return avg_psnr_predicted, avg_ssim_predicted
