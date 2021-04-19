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

loss_name = 'esrganLoss_1e-3_mse'
save_dir = os.path.join(opt.output, opt.dataset_name, str(opt.frame), loss_name)
print('save dir: ', save_dir)

ganLoss = nn.BCEWithLogitsLoss()
mseLoss = nn.MSELoss()
def getGanLoss(input, target_is_real, is_disc=False):
    target_label = input.new_ones(input.size()) * target_is_real
    if is_disc:
      return ganLoss(input, target_label)
    else:
      return 1 * ganLoss(input, target_label)
        
def trainModel(epoch, tot_epoch, training_data_loader, netG, netD, optimizerD, optimizerG, generatorCriterion, device, opt):
    trainBar = tqdm(training_data_loader)
    runningResults = {'batchSize': 0, 'DLoss': 0, 'TLoss': 0, 'GLoss': 0, 'MSELoss': 0, 'DScore': 0, 'GScore': 0}

    netG.train()
    netD.train()

    for input, target in trainBar:
        batchSize = input.size(0)
        runningResults['batchSize'] += batchSize

        input = Variable(input)
        target = Variable(target)

        input = input.to(device)
        target = target.to(device)

        # disable update for discriminator
        for p in netD.parameters():
            p.requires_grad = False

        netG.zero_grad()

        output = netG(input)    
        real_d_pred = netD(target).detach()
        fake_g_pred = netD(output)
        l_g_real = getGanLoss(
            real_d_pred - torch.mean(fake_g_pred), False, is_disc=False)
        l_g_fake = getGanLoss(
            fake_g_pred - torch.mean(real_d_pred), True, is_disc=False)
        l_g_gan = (l_g_real + l_g_fake) / 2
        l_g_tot = l_g_gan + mseLoss(output, target)
        l_g_tot.backward()
        optimizerG.step()

        # enable update for discriminator
        for p in netD.parameters():
            p.requires_grad = True

       # real
        fake_d_pred = netD(output).detach()
        real_d_pred = netD(target)
        l_d_real = getGanLoss(
            real_d_pred - torch.mean(fake_d_pred), True, is_disc=True) * 0.5
        l_d_real.backward()
        # fake
        fake_d_pred = netD(output.detach())
        l_d_fake = getGanLoss(
            fake_d_pred - torch.mean(real_d_pred.detach()),
            False,
            is_disc=True) * 0.5
        l_d_fake.backward()
        optimizerD.step()

        runningResults['TLoss'] += l_g_tot.item() * batchSize
        runningResults['GLoss'] += l_g_gan.item() * batchSize
        runningResults['MSELoss'] += runningResults['TLoss'] - runningResults['GLoss']
        runningResults['DLoss'] += (l_d_fake.item() + l_d_real.item())  * batchSize
        runningResults['DScore'] += torch.mean(real_d_pred.detach()).mean().item() * batchSize
        runningResults['GScore'] += torch.mean(fake_d_pred.detach()).mean().item() * batchSize

        trainBar.set_description(desc='[Epoch: %d/%d] DLoss: %.20f TLoss: %.20f GLoss: %.20f MSELoss: %.20f D(x): %.20f D(G(z)): %.20f' %
                                       (epoch, tot_epoch, runningResults['DLoss'] / runningResults['batchSize'],
                                       runningResults['TLoss'] / runningResults['batchSize'],
                                       runningResults['GLoss'] / runningResults['batchSize'],
                                       runningResults['MSELoss'] / runningResults['batchSize'],
                                       runningResults['DScore'] / runningResults['batchSize'],
                                       runningResults['GScore'] / runningResults['batchSize']))
        gc.collect()

    # learning rate is decayed by a factor of 10 every half of total epochs
    if epoch % 10 == 0:
        for param_group in optimizerG.param_groups:
            param_group['lr'] /= 2.0
        print('Learning rate decayed by half every 10 epochs: lr= ', (optimizerG.param_groups[0]['lr']))

    return runningResults

def saveModelParams(epoch, results, netG, netD, opt):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    gen_save_path = save_dir + '/' + opt.gen_model_name + '_' + str(epoch) + '.pth'
    dis_save_path = save_dir + '/' + opt.dis_model_name + '_' + str(epoch) + '.pth'
    
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
                                     'GLoss': results['TLoss'] / results['batchSize'],
                                     'GLoss': results['GLoss'] / results['batchSize'],
                                     'GLoss': results['MSELoss'] / results['batchSize'],
                                    'DScore': results['DScore'] / results['batchSize'],
                                    'GScore': results['GScore'] / results['batchSize']},
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
        saveModelParams(epoch, runningResults, netG, netD, opt)

if __name__ == "__main__":
    main()
