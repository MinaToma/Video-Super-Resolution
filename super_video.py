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
import math
from model import EDVR
from torchvision.transforms import Compose, ToTensor
from dataset import get_output_set
from metrics import *
from torch.nn.parallel import DataParallel, DistributedDataParallel
import moviepy.editor as mp



"""if __name__ == "__main__":
    if os.path.exists("results"):
        shutil.rmtree("results")

    os.makedirs(opt.input)
    os.makedirs(opt.output)
    generateVideoFrames()
    test_set = get_output_set(opt)
    testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)

    eval()
    generateVideo()"""

class SuperVideo:

    def __init__(self):
        self.model_path = "models/netG_EDVR_4x.pth"
        self.input = "results/input/"
        self.output = "results/output/"
        self.sound = "results/sound.mp3"
        self.final_video = "results/final_video.mp4"
        self.video_path = "1.mp4"
        self.input = "results/input/"
        self.input = "results/input/"
        self.input = "results/input/"
        self.input = "results/input/"
        self.input = "results/input/"
        self.testBatchSize = 1
        self.threads = 1
        self.gpus = 1
        self.frame = 5
        self.upscale_factor = 4
        self.testBatchSize = 1
        self.future_frame = True
        self.cuda = True
        self.gpu_mode = True
        if self.cuda:
            print("Using GPU mode")
            if not torch.cuda.is_available():
                raise Exception("No GPU found, please run without --gpu_mode")

        self.model = EDVR(num_frame=self.frame)

        self.device = torch.device("cuda:0" if self.cuda and torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        self.model_folder, self.model_name = os.path.split(self.model_path)
        _, self.model_folder_name = os.path.split(self.model_folder)

        self.fps = 30
        self.test_set = None
        self.testing_data_loader = None

    def generateVideo(self):
        clip = mp.ImageSequenceClip(self.output, fps=self.fps)
        clip.write_videofile(self.final_video)

    def writeVideo(self):
        clip = mp.ImageSequenceClip(self.output, fps=self.fps)
        clip.write_videofile(self.final_video)

    def eval(self):

        # load model
        modelPath = os.path.join(self.model_path)
        utils.loadPreTrainedModel(gpuMode=self.gpu_mode, model=self.model, modelPath=modelPath)
        self.model.eval()

        count = 0
        for batch in self.testing_data_loader:
            input = batch

            with torch.no_grad():
                input = Variable(input)
                input = input.to(self.device)

            t0 = time.time()
            with torch.no_grad():
                prediction = self.model(input)

            t1 = time.time()
            print("==> Processing: %s || Timer: %.4f sec." % (str(count), (t1 - t0)))
            self.save_img(prediction, count)
            count += 1

    def save_img(self,img, count):
        save_img = img.cpu().data.squeeze().clamp(0, 1).numpy().transpose(1, 2, 0)

        save_fn = self.output + '/' + str(count).zfill(8) + '.png'
        cv2.imwrite(save_fn, save_img * 255, [cv2.IMWRITE_PNG_COMPRESSION, 0])


    def generateVideoFrames(self):
        vidcap = cv2.VideoCapture(self.video_path)
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        success, image = vidcap.read()
        count = 0
        while success and count < 600:
            cv2.imwrite(self.input + "{0:0=6d}".format(count) + ".jpg", image)
            success, image = vidcap.read()
            count += 1

        video = mp.VideoFileClip(self.video_path)
        video.audio.write_audiofile(self.sound)

    def superVideo(self):
        if os.path.exists("results"):
            shutil.rmtree("results")

        os.makedirs(self.input)
        os.makedirs(self.output)
        self.generateVideoFrames()
        self.test_set = get_output_set(self.input,self.frame)
        self.testing_data_loader = DataLoader(dataset=self.test_set, num_workers=self.threads, batch_size=self.testBatchSize,
                                         shuffle=False)
        self.eval()
        self.generateVideo()