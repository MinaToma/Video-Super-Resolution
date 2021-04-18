import os
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import *
import torch.nn.functional as F
from torchvision.models import vgg16

def get_loss_function(opt):
    return GeneratorLoss(opt)

class GeneratorLoss(nn.Module):
    def __init__(self, opt):
        super(GeneratorLoss, self).__init__()
        vgg = vgg16(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.adversarial_loss = nn.BCEWithLogitsLoss()
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()
        self.tv_loss = TVLoss()
        self.charbonnier_loss = CharbonnierLoss()
        
        self.opt = opt

    def forward(self, out_labels, out_images, target_images, target_is_real, is_disc):
        loss = 0.0
        
        # Adversarial Loss
        if self.opt.adversarial_loss != 0.0:
          target_label = out_labels.new_ones(out_labels.size()) * target_is_real
          if is_disc:
            return self.adversarial_loss(out_labels, target_label)
          else:
            loss += self.opt.adversarial_loss * self.adversarial_loss(out_labels, target_label)
        
        # Perception Loss
        if self.opt.perception_loss != 0.0:
            loss += self.opt.perception_loss * self.mse_loss(self.loss_network(out_images), self.loss_network(target_images))
        
        # Image Loss
        if self.opt.mse_loss != 0.0:
            loss += self.opt.mse_loss * self.mse_loss(out_images, target_images)
        
        # TV Loss
        if self.opt.tv_loss != 0.0:
            loss += self.opt.tv_loss * self.tv_loss(out_images)

        if self.opt.charbonnier_loss != 0.0:
            loss += self.opt.charbonnier_loss * self.charbonnier_loss(out_images, target_images)

        return loss


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
