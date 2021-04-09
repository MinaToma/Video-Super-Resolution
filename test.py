from __future__ import print_function
import argparse
import glob
from copy import deepcopy

import os
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
import utils
import time
import cv2
import math
from model import EDVR
from torchvision.transforms import Compose, ToTensor
from dataset import get_test_set
from torch.nn.parallel import DataParallel, DistributedDataParallel


# Training settings
parser = argparse.ArgumentParser(description='EDVR GAN Test')
parser.add_argument('-m', '--model', default="/content/drive/MyDrive/VSR/EDVR/Copy of EDVR_L_x4_SR_Vimeo90K_official-162b54e4.pth", help="Model")
parser.add_argument('-o', '--output', default='/content/VSR/results/', help="Location to save test results")
parser.add_argument('-c', '--gpu_mode',default=True, action='store_true', required=False, help="Use a CUDA compatible GPU if available")
parser.add_argument('--testBatchSize', type=int, default=1, help="Testing Batch Size")
parser.add_argument('--threads', type=int, default=1, help="Dataloader Threads")
parser.add_argument('--gpus', default=1, type=int, help="How many GPU's to use")
parser.add_argument('--dataset_name', default='Vid4', help='Location to ground truth frames')
parser.add_argument('--gt_dir', help='Location to ground truth frames')
parser.add_argument('--lr_dir', help='Location to low resolution frames')
parser.add_argument('--frame', type=int, default=7, help="")
parser.add_argument('-u', '--upscale_only', type=bool, default=False, help="Upscale mode - without downscaling.")

opt = parser.parse_args()

cuda = opt.gpu_mode
if cuda:
    print("Using GPU mode")
    if not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --gpu_mode")

print('==> Loading datasets')
test_set = get_test_set(opt)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)

print('==> Building model ')
model = EDVR(num_frame=opt.frame)

device = torch.device("cuda:0" if cuda and torch.cuda.is_available() else "cpu")
model = model.to(device)

def eval():
    # print EDVR GAN architecture
    # utils.printNetworkArch(netG=model, netD=None)

    # load model
    modelPath = os.path.join(opt.model)
    utils.loadPreTrainedModel(gpuMode=opt.gpu_mode, model=model, modelPath=modelPath)
    model.eval()
    count = 0
    
    if not opt.upscale_only:
        avg_psnr_predicted = 0.0
        avg_ssim_predicted = 0.0

    for batch in testing_data_loader:
        input, target = batch[0], batch[1]

        with torch.no_grad():
            input = input.to(device)
            target = target.to(device)

        t0 = time.time()
        with torch.no_grad():
            prediction = model(input)

        t1 = time.time()
        print("==> Processing: %s || Timer: %.4f sec." % (str(count), (t1 - t0)))
        save_img(prediction.cpu().data, str(count), True)
        save_img(target.cpu().data, 'target' + str(count), True)

        prediction = prediction.cpu()
        prediction = prediction.data[0].numpy().astype(np.float32)
        prediction = prediction * 255.

        target = target.cpu().squeeze().numpy().astype(np.float32)
        target = target * 255.
        if not opt.upscale_only:
            psnr_predicted = PSNR(prediction, target)
            ssim_predicted = SSIM(prediction, target)
            print("PSNR Predicted = ", psnr_predicted)
            print("SSIM Predicted = ", ssim_predicted)
            avg_psnr_predicted += psnr_predicted
            avg_ssim_predicted += ssim_predicted
        count += 1
    
    if not opt.upscale_only:
        print("Avg PSNR Predicted = ", avg_psnr_predicted / count)
        print("Avg SSIM Predicted = ", avg_ssim_predicted / count)


def save_img(img, img_name, pred_flag):
    save_img = img.squeeze().clamp(0, 1).numpy().transpose(1, 2, 0)

    # save img
    save_dir = os.path.join(opt.output, opt.dataset_name, '4x')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if pred_flag:
        save_fn = save_dir + '/' + img_name + '_' + 'EDVR_GAN' + 'F' + str(opt.frame) + '.png'
        cv2.imwrite(save_fn, save_img * 255, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    else:
        save_fn = save_dir + '/' + img_name + '.png'

def PSNR(pred, gt, shave_border=4):
    height, width = pred.shape[1:3]
    pred = pred[:, 1 + shave_border:height - shave_border, 1 + shave_border:width - shave_border]
    gt = gt[:, 1 + shave_border:height - shave_border, 1 + shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)

def _ssim(img1, img2):
    """Calculate SSIM (structural similarity) for one channel images.

    It is called by func:`calculate_ssim`.

    Args:
        img1 (ndarray): Images with range [0, 255] with order 'HWC'.
        img2 (ndarray): Images with range [0, 255] with order 'HWC'.

    Returns:
        float: ssim result.
    """

    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def SSIM(img1, img2):
    """Calculate SSIM (structural similarity).

    Ref:
    Image quality assessment: From error visibility to structural similarity

    The results are the same as that of the official released MATLAB code in
    https://ece.uwaterloo.ca/~z70wang/research/ssim/.

    For three-channel images, SSIM is calculated for each channel and then
    averaged.

    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the SSIM calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: ssim result.
    """

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    ssims = []
    for i in range(img1.shape[0]):
        ssims.append(_ssim(img1[i, ...], img2[i, ...]))
    return np.array(ssims).mean()

if __name__ == "__main__":
    eval()
