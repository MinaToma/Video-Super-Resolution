import os
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import *
import torch.nn.functional as F
from torchvision.models import vgg16

class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        vgg = vgg16(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()
        self.tv_loss = TVLoss()
        self.charbonnier_loss = CharbonnierLoss()

    def forward(self, hr_est, hr_img, runningResults, batchSize):
        # Perception Loss
        perception_loss = self.mse_loss(self.loss_network(hr_est), self.loss_network(hr_img))
        # Image Loss
        image_loss = self.mse_loss(hr_est, hr_img)
        # TV Loss
        tv_loss = self.tv_loss(hr_est)
        # charbonnier_loss
        charbonnier_loss = self.charbonnier_loss(hr_est, hr_img)

        runningResults["perception_loss"] += perception_loss.item() * batchSize
        runningResults["mse_loss"] += image_loss.item() * batchSize
        runningResults["tv_loss"] += tv_loss.item() * batchSize
        runningResults["charbonnier_loss"] += charbonnier_loss.item() * batchSize

        return perception_loss


class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class CharbonnierLoss(torch.nn.Module):
    def __init__(self):
        super(CharbonnierLoss, self).__init__()
        self.eps = 1e-3

    def forward(self, x, y):
        diff = torch.add(x, -y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss
