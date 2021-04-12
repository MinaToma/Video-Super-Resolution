import argparse
import gc
import os
import pandas as pd
import torch.optim as optim
import torch.utils.data
from tqdm import tqdm
from SRGAN.model import Discriminator
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import utils
from model import EDVR
from dataset import get_training_set
from loss import get_loss_function
from torchvision.transforms import Compose, ToTensor


# Handle command line arguments
parser = argparse.ArgumentParser(description='Train EDRV GAN: Super Resolution Models')
parser.add_argument('--batchSize', type=int, default=2, help='training batch size')
parser.add_argument('--start_epoch', type=int, default=1, help='Starting epoch for continuing training')
parser.add_argument('--nEpochs', type=int, default=5, help='number of epochs to train for')
parser.add_argument('--snapshots', type=int, default=1, help='Snapshots')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate. Default=0.01')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=2, help='number of threads for data loader to use')
parser.add_argument('--gpus', default=8, type=int, help='number of gpu')
parser.add_argument('--frame', type=int, default=7)
parser.add_argument('--data_augmentation', type=bool, default=True)
parser.add_argument('--pretrained', action='store_true', help='Use pretrained model')
parser.add_argument('--gen_model_name', default='netG_EDVR_4x', help='Name of generator model')
parser.add_argument('--dis_model_name', default='netD_EDVR_4x', help='Name of discriminator model')
parser.add_argument('--patch_size', type=int, default=256, help='patch of gt, 0 to use original frame size')
parser.add_argument('--file_list', help='sr pretrained base model')
parser.add_argument('--output', default='/content/out', help='Location to save checkpoint models')
parser.add_argument('--dataset_name', default='vemo90k', help='Location to ground truth frames')
parser.add_argument('--gt_dir', help='Location to ground truth frames')
parser.add_argument('--lr_dir', help='Location to low resolution frames')
parser.add_argument('--mse_loss', type=float, default=1.0)
parser.add_argument('--adversarial_loss', type=float, default=0.001)
parser.add_argument('--perception_loss', type=float, default=0.006)
parser.add_argument('--tv_loss', type=float, default=2e-8)
parser.add_argument('--charbonnier_loss', type=float, default=0.0)

opt = parser.parse_args()
loss_name = ''

print('=========================LOSS=========================================')
if opt.perception_loss != 0.0:
  loss_name += 'perception_loss_' + str(opt.perception_loss) + '_'
  print("Uses perception_loss with weight: " + str(opt.perception_loss))

if opt.mse_loss != 0.0:
  loss_name += 'mse_loss_' + str(opt.mse_loss) + '_'
  print("Uses mse_loss with weight: " + str(opt.mse_loss))

if opt.tv_loss != 0.0:
  loss_name += 'tv_loss_' + str(opt.tv_loss) + '_'
  print("Uses tv_loss with weight: " + str(opt.tv_loss))

if opt.charbonnier_loss != 0.0:
  loss_name += 'charbonnier_loss_' + str(opt.charbonnier_loss) + '_'
  print("Uses charbonnier_loss with weight: " + str(opt.charbonnier_loss))

if opt.adversarial_loss != 0.0:
  loss_name += 'adversarial_loss_' + str(opt.adversarial_loss) + '_'
  print("Uses adversarial_loss with weight: " + str(opt.adversarial_loss))

save_dir = os.path.join(opt.output, opt.dataset_name, str(opt.frame), loss_name)
print('save dir: ', save_dir)


def trainModel(epoch, tot_epoch, training_data_loader, netG, netD, optimizerD, optimizerG, generatorCriterion, device, opt):
    trainBar = tqdm(training_data_loader)
    runningResults = {'batchSize': 0, 'DLoss': 0, 'GLoss': 0, 'DScore': 0, 'GScore': 0}

    netG.train()
    netD.train()

    for input, target in trainBar:
        batchSize = input.size(0)
        runningResults['batchSize'] += batchSize

        input = Variable(input)
        target = Variable(target)

        input = input.to(device)
        target = target.to(device)

        ################################################################################################################
        # (1) Update D network: maximize D(x)-1-D(G(z))
        ################################################################################################################

        DLoss = 0
        netD.zero_grad()

        fakeHR = netG(input)
        realOut = netD(target).mean()
        fakeOut = netD(fakeHR).mean()

        DLoss += 1 - realOut + fakeOut
        DLoss.backward(retain_graph=True)
        optimizerD.step()

        ################################################################################################################
        # (2) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
        ################################################################################################################
        netG.zero_grad()

        fakeOut = netD(fakeHR).mean()
        GLoss = generatorCriterion(fakeOut, fakeHR, target)
        GLoss.backward()
        optimizerG.step()

        fakeHR = netG(input)
        fakeOut = netD(fakeHR).mean()
        runningResults['GLoss'] += GLoss.item() * batchSize
        runningResults['DLoss'] += DLoss.item() * batchSize
        runningResults['DScore'] += realOut.item() * batchSize
        runningResults['GScore'] += fakeOut.item() * batchSize

        trainBar.set_description(desc='[Epoch: %d/%d] D Loss: %.20f G Loss: %.20f D(x): %.20f D(G(z)): %.20f' %
                                       (epoch, tot_epoch, runningResults['DLoss'] / runningResults['batchSize'],
                                       runningResults['GLoss'] / runningResults['batchSize'],
                                       runningResults['DScore'] / runningResults['batchSize'],
                                       runningResults['GScore'] / runningResults['batchSize']))
        gc.collect()

    # learning rate is decayed by a factor of 10 every half of total epochs
    if epoch % 10 == 0:
        for param_group in optimizerG.param_groups:
            param_group['lr'] /= 2.0
        print('Learning rate decayed by half every 10 epochs: lr= ', (optimizerG.param_groups[0]['lr']))

    return runningResults

def saveModelParams(epoch, runningResults, netG, netD, opt):
    results = {'DLoss': [], 'GLoss': [], 'DScore': [], 'GScore': [], 'PSNR': [], 'SSIM': []}

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    gen_save_path = save_dir + '/' + opt.gen_model_name + '_' + str(epoch) + '.pth'
    dis_save_path = save_dir + '/' + opt.dis_model_name + '_' + str(epoch) + '.pth'
    # Save model parameters
    torch.save(netG.state_dict(), gen_save_path)
    torch.save(netD.state_dict(), dis_save_path)

    print("Checkpoint saved to {}".format(gen_save_path))
    print("Checkpoint saved to {}".format(dis_save_path))

def main():
    """ Lets begin the training process! """

    # Load dataset
    print('==> Loading datasets')
    train_set = get_training_set(opt)                                 
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)

    # Use generator as EDVR
    netG = EDVR(num_frame=opt.frame)
    print('# of Generator parameters: ', sum(param.numel() for param in netG.parameters()))

    # Use discriminator from SRGAN
    netD = Discriminator()
    print('# of Discriminator parameters: ', sum(param.numel() for param in netD.parameters()))

    # get loss function
    generatorCriterion = get_loss_function(opt)

    # Specify device
    device = torch.device("cuda:0" if torch.cuda.is_available() and opt.gpu_mode else "cpu")

    if opt.gpu_mode and torch.cuda.is_available():
        utils.printCUDAStats()

    netG.to(device)
    netD.to(device)
    generatorCriterion.to(device)

    # divide learning by half every 10 epochs
    lr = opt.lr / (2 ** (opt.start_epoch // 10))

    # Use Adam optimizer
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)
  
    # print EDVR_GAN architecture
    # utils.printNetworkArch(netG, netD)
    
    start_epoch  = opt.start_epoch
    if opt.pretrained:
        modelDisPath = save_dir + '/' + opt.dis_model_name + '_' + str(start_epoch) + '.pth'
        utils.loadPreTrainedModel(gpuMode=opt.gpu_mode, model=netD, modelPath=modelDisPath)
        modelPath = save_dir + '/' + opt.gen_model_name + '_' + str(start_epoch) + '.pth'
        utils.loadPreTrainedModel(gpuMode=opt.gpu_mode, model=netG, modelPath=modelPath)
        start_epoch += 1

    for epoch in range(start_epoch, start_epoch + opt.nEpochs):
        runningResults = trainModel(epoch, start_epoch + opt.nEpochs - 1, training_data_loader, netG, netD, optimizerD, optimizerG, generatorCriterion, device, opt)

        if epoch % (opt.snapshots) == 0:
            saveModelParams(epoch, runningResults, netG, netD, opt)

if __name__ == "__main__":
    main()
