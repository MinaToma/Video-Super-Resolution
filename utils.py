import os
import torch
import torch.nn as nn

def loadPreTrainedModel(gpuMode, model, modelPath):
    if os.path.exists(modelPath):
        if gpuMode and torch.cuda.is_available():
            state_dict = torch.load(modelPath)
        else:
            state_dict = torch.load(modelPath, map_location=torch.device('cpu'))

        # Handle the usual (non-DataParallel) case
        try:
            model.load_state_dict(state_dict)

        # Handle DataParallel case
        except:
            # create new OrderedDict that does not contain module.
            from collections import OrderedDict

            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = "module." + k if not k.startswith("module.") else k  # remove module.
                new_state_dict[name] = v

            # load params
            try:
                model.load_state_dict(new_state_dict)
            except:
                model = nn.DataParallel(model)
                model.load_state_dict(new_state_dict)
            
        print('Pre-trained SR model loaded from:', modelPath)
    else:
        print('Couldn\'t find pre-trained SR model at:', modelPath)

def printCUDAStats():
    print("# of CUDA devices detected: %s", torch.cuda.device_count())
    print("Using CUDA device #: %s", torch.cuda.current_device())
    print("CUDA device name: %s", torch.cuda.get_device_name(torch.cuda.current_device()))

def _printNetworkArch(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

def printNetworkArch(netG, netD):
    print('------------- EDVR GAN Network Architecture -------------')
    if netG:
        print('----------------- Generator Architecture ------------------')
        _printNetworkArch(netG)

    if netD:
        print('--------------- Discriminator Architecture ----------------')
        _printNetworkArch(netD)
        print('-----------------------------------------------------------')