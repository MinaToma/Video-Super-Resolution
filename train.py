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
parser.add_argument('--batchSize', type=int, default=4, help='training batch size')
parser.add_argument('--start_epoch', type=int, default=1, help='Starting epoch for continuing training')
parser.add_argument('--nEpochs', type=int, default=150, help='number of epochs to train for')
parser.add_argument('--snapshots', type=int, default=1, help='Snapshots')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate. Default=0.01')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=2, help='number of threads for data loader to use')
parser.add_argument('--gpus', default=8, type=int, help='number of gpu')
parser.add_argument('--frame', type=int, default=7)
parser.add_argument('--data_augmentation', type=bool, default=True)
parser.add_argument('--pretrained_sr', help='sr pretrained base model')
parser.add_argument('--pretrained_dis', help='sr pretrained base model')
parser.add_argument('--file_list', help='sr pretrained base model')
parser.add_argument('--pretrained', action='store_true', help='Use pretrained model')
parser.add_argument('--save_folder', default='/content/out', help='Location to save checkpoint models')
parser.add_argument('--dataset_name', default='vemo90k', help='Location to ground truth frames')
parser.add_argument('--gt_dir', help='Location to ground truth frames')
parser.add_argument('--lr_dir', help='Location to low resolution frames')
parser.add_argument('--loss', default='', help='Location to low resolution frames')
parser.add_argument('--pretrained_epoch',type=str, default='epoch.txt',  help='number of pretrained epoch')

def trainModel(epoch, training_data_loader, netG, netD, optimizerD, optimizerG, generatorCriterion, device, opt):
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
                                       (epoch, opt.nEpochs, runningResults['DLoss'] / runningResults['batchSize'],
                                       runningResults['GLoss'] / runningResults['batchSize'],
                                       runningResults['DScore'] / runningResults['batchSize'],
                                       runningResults['GScore'] / runningResults['batchSize']))
        gc.collect()

    # learning rate is decayed by a factor of 10 every half of total epochs
    if (epoch + 1) % (opt.nEpochs / 2) == 0:
        for param_group in optimizerG.param_groups:
            param_group['lr'] /= 10.0
        logger.info('Learning rate decay: lr=%s', (optimizerG.param_groups[0]['lr']))

    return runningResults

def saveModelParams(epoch, runningResults, netG, netD, opt):
    results = {'DLoss': [], 'GLoss': [], 'DScore': [], 'GScore': [], 'PSNR': [], 'SSIM': []}

    # Save number of Epoch
    # f = open('/content/VSR/weights/epoch.txt', 'w')
    # f.write(str(epoch))  
    # f.close()

    # Save model parameters
    torch.save(netG.state_dict(), opt.save_folder + '/netG_EDVR_epoch_4x_%d.pth' % (epoch))
    torch.save(netD.state_dict(), opt.save_folder + '/netD_EDVR_epoch_4x_%d.pth' % (epoch))

    print("Checkpoint saved to {}".format('weights/netD_EDVR_epoch_4x_%d.pth' % (epoch)))
    print("Checkpoint saved to {}".format('weights/netG_EDVR_epoch_4x_%d.pth' % (epoch)))

    # Save Loss\Scores\PSNR\SSIM
    results['DLoss'].append(runningResults['DLoss'] / runningResults['batchSize'])
    results['GLoss'].append(runningResults['GLoss'] / runningResults['batchSize'])
    results['DScore'].append(runningResults['DScore'] / runningResults['batchSize'])
    results['GScore'].append(runningResults['GScore'] / runningResults['batchSize'])

    # if epoch % 1 == 0 and epoch != 0:
    #     out_path = '/content/VSR/statistics/'
    #     data_frame = pd.DataFrame(data={'DLoss': results['DLoss'], 'GLoss': results['GLoss'], 'DScore': results['DScore'],
    #                               'GScore': results['GScore']}, index=range(1, epoch + 1))
    #     data_frame.to_csv(out_path + 'EDVR_GAN_4x_Train_Results.csv', index_label='Epoch')

def main():
    """ Lets begin the training process! """

    opt = parser.parse_args()

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

    # Use Adam optimizer
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)
  
    # print EDVR_GAN architecture
    # utils.printNetworkArch(netG, netD)
    
    if opt.pretrained:
        modelDisPath = os.path.join(opt.save_folder + opt.pretrained_dis + '_' + str(opt.start_epoch) + '.pth')
        utils.loadPreTrainedModel(gpuMode=opt.gpu_mode, model=netD, modelPath=modelDisPath)
        modelPath = os.path.join(opt.save_folder + opt.pretrained_sr + '_' + str(opt.start_epoch) + '.pth')
        utils.loadPreTrainedModel(gpuMode=opt.gpu_mode, model=netG, modelPath=modelPath)
        epochs = open(os.path.join(opt.save_folder + opt.pretrained_epoch),'r')
        numberOfEpochs = int(epochs.readline())
        epochs.close()
    else:
        numberOfEpochs = 0

    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        runningResults = trainModel(epoch, training_data_loader, netG, netD, optimizerD, optimizerG, generatorCriterion, device, opt)

        if (epoch + 1) % (opt.snapshots) == 0:
            saveModelParams(epoch + numberOfEpochs, runningResults, netG, netD, opt)

if __name__ == "__main__":
    main()
