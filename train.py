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
from loss import GeneratorLoss
from validate import Validator
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
parser.add_argument('--pretrained_gen_path', default= 'x.pth', help='Use pretrained model')
parser.add_argument('--pretrained_dis_path', default= 'x.pth', help='Use pretrained model')
parser.add_argument('--gen_model_name', default='netG_EDVR_4x', help='Name of generator model')
parser.add_argument('--dis_model_name', default='netD_EDVR_4x', help='Name of discriminator model')
parser.add_argument('--patch_size', type=int, default=256, help='patch of gt, 0 to use original frame size')
parser.add_argument('--file_list', help='sr pretrained base model')
parser.add_argument('--output', default='/content/out', help='Location to save checkpoint models')
parser.add_argument('--dataset_name', default='vemo90k', help='Location to ground truth frames')
parser.add_argument('--gt_dir', help='Location to ground truth frames')
parser.add_argument('--lr_dir', help='Location to low resolution frames')
parser.add_argument('--folder_save_name', help='folder name to save models')

opt = parser.parse_args()

save_dir = os.path.join(opt.output, opt.dataset_name, str(opt.frame), opt.folder_save_name)
print('save dir: ', save_dir)

ganLoss = nn.BCEWithLogitsLoss()
def getGanLoss(input, target_is_real, is_disc):
    target_label = input.new_ones(input.size()) * target_is_real
    if is_disc:
      return ganLoss(input, target_label)
    else:      
      return 0.001 * ganLoss(input, target_label)

def trainModel(epoch, tot_epoch, training_data_loader, netG, netD, optimizerD, optimizerG, generatorCriterion, device, opt):
    trainBar = tqdm(training_data_loader)
    runningResults = {'batchSize': 0,
                      'DLoss': 0, 
                      'GLoss': 0, 
                      'DScore': 0, 
                      'GScore': 0,
                      'REDS_PSNR': 0, 
                      'REDS_SSIM': 0, 
                      'Vid4_PSNR': 0, 
                      'Vid4_SSIM': 0,
                      'adversarial_loss': 0,
                      'perception_loss': 0,
                      'mse_loss': 0,
                      'tv_loss': 0,
                      }

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
        # (1) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
        ################################################################################################################
        for p in netD.parameters():
            p.requires_grad = False

        optimizerG.zero_grad()
        fakeHR = netG(input)
        losses = generatorCriterion(fakeHR, target, runningResults, batchSize)
        real_d_pred = netD(target).detach()
        fake_g_pred = netD(fakeHR)
        l_g_real = getGanLoss(real_d_pred - torch.mean(fake_g_pred),
                              False,
                              False)
        l_g_fake = getGanLoss(fake_g_pred - torch.mean(real_d_pred),
                              True,
                              False)
        l_g_gan = (l_g_real + l_g_fake) / 2
        runningResults["adversarial_loss"] += l_g_gan.item() * batchSize
        GLoss = l_g_gan + losses
        GLoss.backward()
        optimizerG.step()

        ################################################################################################################
        # (2) Update D network: maximize D(x)-1-D(G(z))
        ################################################################################################################
        for p in netD.parameters():
            p.requires_grad = True
        optimizerD.zero_grad()
        # real
        fake_d_pred = netD(fakeHR).detach()
        real_d_pred = netD(target)
        l_d_real = getGanLoss(real_d_pred - torch.mean(fake_d_pred),
                              True,
                              True
                              )
        # fake
        fake_d_pred = netD(fakeHR.detach())
        l_d_fake = getGanLoss(fake_d_pred - torch.mean(real_d_pred.detach()),
                              False,
                              True
                              ) 
        DLoss = (l_d_fake + l_d_real) / 2
        optimizerD.step()

        runningResults['GLoss'] += GLoss.item() * batchSize
        runningResults['DLoss'] += DLoss.item() * batchSize
        runningResults['DScore'] += torch.sigmoid(real_d_pred).mean().item() * batchSize
        runningResults['GScore'] += torch.sigmoid(fake_d_pred).mean().item() * batchSize

        trainBar.set_description(desc='[Epoch: %d/%d] D Loss: %.20f G Loss: %.20f D(x): %.20f D(G(z)): %.20f' %
                                       (epoch, tot_epoch, runningResults['DLoss'] / runningResults['batchSize'],
                                       runningResults['GLoss'] / runningResults['batchSize'],
                                       runningResults['DScore'] / runningResults['batchSize'],
                                       runningResults['GScore'] / runningResults['batchSize']))

    # learning rate is decayed by a factor of 10 every half of total epochs
    if epoch % 10 == 0:
        for param_group in optimizerG.param_groups:
            param_group['lr'] /= 2.0
        for param_group in optimizerD.param_groups:
            param_group['lr'] /= 2.0
        print('Learning rate in gen decayed by half every 10 epochs: lr= ', (optimizerG.param_groups[0]['lr']))
        print('Learning rate in dis decayed by half every 10 epochs: lr= ', (optimizerD.param_groups[0]['lr']))

    return runningResults

def saveModelParams(epoch, results, netG, netD, opt, validator):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    gen_save_path = save_dir + '/' + opt.gen_model_name + '_' + str(epoch) + '.pth'
    dis_save_path = save_dir + '/' + opt.dis_model_name + '_' + str(epoch) + '.pth'
    
    # Validate Generator and fill accuracy
    validator.validate(netG, results)

    # Save model parameters
    if epoch % (opt.snapshots) == 0:
        torch.save(netG.state_dict(), gen_save_path)
        torch.save(netD.state_dict(), dis_save_path)
        print("Checkpoint saved to {}".format(gen_save_path))
        print("Checkpoint saved to {}".format(dis_save_path))

    csv_path = save_dir + '/train_results.csv'
    header = epoch == 1
    if epoch == 1 and os.path.exists(csv_path):
      os.remove(csv_path)

    data_frame = pd.DataFrame(data={'DLoss': results['DLoss'] / results['batchSize'],
                                    'GLoss': results['GLoss'] / results['batchSize'],
                                    'DScore': results['DScore'] / results['batchSize'],
                                    'GScore': results['GScore'] / results['batchSize'],
                                    'adversarial_loss': results['adversarial_loss'] / results['batchSize'],
                                    'perception_loss': results['perception_loss'] / results['batchSize'],
                                    'mse_loss': results['mse_loss'] / results['batchSize'],
                                    'tv_loss': results['tv_loss'] / results['batchSize'],
                                    'REDS_PSNR': results['REDS_PSNR'], 
                                    'REDS_SSIM': results['REDS_SSIM'],
                                    'Vid4_PSNR': results['Vid4_PSNR'], 
                                    'Vid4_SSIM': results['Vid4_SSIM'],
                                    },
                                     index=range(epoch, epoch + 1))

    data_frame.to_csv(csv_path, mode='a', index_label='Epoch', header=header)

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
    generatorCriterion = GeneratorLoss()

    # Specify device
    device = torch.device("cuda:0" if torch.cuda.is_available() and opt.gpu_mode else "cpu")

    validator = Validator(opt, device)

    if opt.gpu_mode and torch.cuda.is_available():
        utils.printCUDAStats()

    netG.to(device)
    netD.to(device)
    generatorCriterion.to(device)

    # divide learning by half every 10 epochs
    lr = opt.lr / (2 ** ((opt.start_epoch - 1) // 10))

    # Use Adam optimizer
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
  
    # print EDVR_GAN architecture
    # utils.printNetworkArch(netG, netD)
    
    start_epoch  = opt.start_epoch
    if opt.pretrained:
        utils.loadPreTrainedModel(gpuMode=opt.gpu_mode, model=netD, modelPath=opt.pretrained_dis_path)
        utils.loadPreTrainedModel(gpuMode=opt.gpu_mode, model=netG, modelPath=opt.pretrained_gen_path)

    for epoch in range(start_epoch, start_epoch + opt.nEpochs):
        runningResults = trainModel(epoch, start_epoch + opt.nEpochs - 1, training_data_loader, netG, netD, optimizerD, optimizerG, generatorCriterion, device, opt)
        saveModelParams(epoch, runningResults, netG, netD, opt, validator)

if __name__ == "__main__":
    main()
