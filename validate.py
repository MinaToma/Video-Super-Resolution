import torch
import numpy as np
from metrics import *
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataset import REDSValidationDataset, Vid4ValidationDataset

device = torch.device("cuda:0")
print('==> Loading Validation datasets')
val_REDS_set = REDSValidationDataset()
val_REDS_set_loader = DataLoader(dataset=val_REDS_set, num_workers=1, batch_size=1, shuffle=False)

val_Vid4_set = Vid4ValidationDataset()
val_Vid4_set_loader = DataLoader(dataset=val_Vid4_set, num_workers=1, batch_size=1, shuffle=False)

def validate_model(model):
    model.eval()
    REDS_psnr, REDS_ssim = validate_REDS(model)
    Vid4_psnr, Vid4_ssim = validate_Vid4(model)
    return REDS_psnr, REDS_ssim, Vid4_psnr, Vid4_ssim
    

def validate_Vid4(model):
    avg_psnr_predicted = 0.0
    avg_ssim_predicted = 0.0

    count = 0
    for batch in val_Vid4_set_loader:
        input, target = batch[0], batch[1]
        with torch.no_grad():
            input = Variable(input)
            target = Variable(target)
            input = input.to(device)
            target = target.to(device)
            prediction = model(input)
        prediction = prediction.cpu()
        prediction = prediction.data[0].numpy().astype(np.float32)
        prediction = prediction * 255.

        target = target.cpu().squeeze().numpy().astype(np.float32)
        target = target * 255.
        psnr_predicted = PSNR(prediction, target)
        ssim_predicted = SSIM(prediction, target)

        avg_psnr_predicted += psnr_predicted
        avg_ssim_predicted += ssim_predicted          
        count += 1
    avg_psnr_predicted /= count
    avg_ssim_predicted /= count
    return avg_psnr_predicted, avg_ssim_predicted

def validate_REDS(model):
    avg_psnr_predicted = 0.0
    avg_ssim_predicted = 0.0

    count = 0
    for batch in val_REDS_set_loader:
        input, target = batch[0], batch[1]
        with torch.no_grad():
            input = Variable(input)
            target = Variable(target)
            input = input.to(device)
            target = target.to(device)
            prediction = model(input)
        prediction = prediction.cpu()
        prediction = prediction.data[0].numpy().astype(np.float32)
        prediction = prediction * 255.

        target = target.cpu().squeeze().numpy().astype(np.float32)
        target = target * 255.
        psnr_predicted = PSNR(prediction, target)
        ssim_predicted = SSIM(prediction, target)

        avg_psnr_predicted += psnr_predicted
        avg_ssim_predicted += ssim_predicted          
        count += 1
    avg_psnr_predicted /= count
    avg_ssim_predicted /= count
    return avg_psnr_predicted, avg_ssim_predicted