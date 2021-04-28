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
import shutil
import time
import cv2
import math
from model import EDVR
from torchvision.transforms import Compose, ToTensor
from dataset import get_test_set
from metrics import *
from torch.nn.parallel import DataParallel, DistributedDataParallel

# Training settings
parser = argparse.ArgumentParser(description='EDVR GAN Test')
parser.add_argument('--model_path', help="Model folder path")
parser.add_argument('--output', help="Location to save test results")
parser.add_argument('--gpu_mode',default=True, action='store_true', required=False, help="Use a CUDA compatible GPU if available")
parser.add_argument('--testBatchSize', type=int, default=1, help="Testing Batch Size")
parser.add_argument('--threads', type=int, default=1, help="Dataloader Threads")
parser.add_argument('--gpus', default=1, type=int, help="How many GPU's to use")
parser.add_argument('--dataset_name', default='Vid4', help='Dataset name to use')
parser.add_argument('--gt_dir', help='Location to ground truth frames')
parser.add_argument('--lr_dir', help='Location to low resolution frames')
parser.add_argument('--clip_name', help='clip name in dataset')
parser.add_argument('--frame', type=int, default=7, help="number of frames")
parser.add_argument('--generate_video', action='store_true', help="whether to generate video or not")
parser.add_argument('--save_image', action='store_true', help="whether to save output image or not")
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

model_folder, model_name = os.path.split(opt.model_path)
_, model_folder_name = os.path.split(model_folder)

save_dir = os.path.join(opt.output, opt.dataset_name, opt.clip_name, str(opt.frame), model_folder_name, model_name)

def eval():
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir) # delete existing folder
    os.makedirs(save_dir) # create empty directory

    # save results to file
    f = open(save_dir + "/results.txt", "a")
    
    # print EDVR GAN architecture
    # utils.printNetworkArch(netG=model, netD=None)

    # load model
    modelPath = os.path.join(opt.model_path)
    utils.loadPreTrainedModel(gpuMode=opt.gpu_mode, model=model, modelPath=modelPath)
    model.eval()
    
    if not opt.upscale_only:
        avg_psnr_predicted = 0.0
        avg_ssim_predicted = 0.0

    count = 0
    for batch in testing_data_loader:
        input, target = batch[0], batch[1]

        with torch.no_grad():
            input = Variable(input)
            target = Variable(target)
            input = input.to(device)
            target = target.to(device)

        t0 = time.time()
        with torch.no_grad():
            prediction = model(input)

        t1 = time.time()
        print("==> Processing: %s || Timer: %.4f sec." % (str(count), (t1 - t0)))
        # write to file
        f.write("==> Processing: %s || Timer: %.4f sec." % (str(count), (t1 - t0))  + "\n")
        if opt.save_image:
            save_img(prediction, count)

        prediction = prediction.cpu()
        prediction = prediction.data[0].numpy().astype(np.float32)
        prediction = prediction * 255.

        target = target.cpu().squeeze().numpy().astype(np.float32)
        target = target * 255.
        if not opt.upscale_only:
            psnr_predicted = PSNR(prediction, target)
            ssim_predicted = SSIM(prediction, target)
            avg_psnr_predicted += psnr_predicted
            avg_ssim_predicted += ssim_predicted
            print("PSNR Predicted = ", psnr_predicted)
            print("SSIM Predicted = ", ssim_predicted)
            # write to file
            f.write("PSNR Predicted = " + str(psnr_predicted) + "\n")
            f.write("SSIM Predicted = " + str(ssim_predicted) + "\n")
            
        count += 1
    
    if not opt.upscale_only:
        print("Avg PSNR Predicted = ", avg_psnr_predicted / count)
        print("Avg SSIM Predicted = ", avg_ssim_predicted / count)
        # write to file
        f.write("Avg PSNR Predicted = " + str(avg_psnr_predicted / count) +  "\n")
        f.write("Avg SSIM Predicted = " + str(avg_ssim_predicted / count) + "\n")
    f.close()
        

def save_img(img, count):
    save_img = img.cpu().data.squeeze().clamp(0, 1).numpy().transpose(1, 2, 0)

    save_fn = save_dir + '/' +  str(count).zfill(8) + '.png'
    cv2.imwrite(save_fn, save_img * 255, [cv2.IMWRITE_PNG_COMPRESSION, 0])

if __name__ == "__main__":
    eval()
    if opt.generate_video:
        generateVideo()
