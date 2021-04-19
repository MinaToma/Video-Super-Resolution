import cv2
import math
import torch
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataset import REDSValidationDataset, Vid4ValidationDataset

device = torch.device("cuda:0")
print('==> Loading Validation datasets')
val_set = Vid4ValidationDataset()
val_set_loader = DataLoader(dataset=val_set, num_workers=1, batch_size=1, shuffle=False)
validateBar = tqdm(val_set_loader)

def validate_model(model):
    model.eval()

    avg_psnr_predicted = 0.0
    avg_ssim_predicted = 0.0

    count = 0
    for batch in validateBar:
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
        #print("*********************************PSNR: %f SSIM: %f*********************************\n" % (psnr_predicted, ssim_predicted))

        avg_psnr_predicted += psnr_predicted
        avg_ssim_predicted += ssim_predicted          
        count += 1
    avg_psnr_predicted /= count
    avg_ssim_predicted /= count
    #print("*********************************AVG PSNR: %f AVG SSIM: %f*********************************\n" % (avg_psnr_predicted, avg_ssim_predicted))
    return avg_psnr_predicted, avg_ssim_predicted

def PSNR(pred, gt, shave_border=4):
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)

def _ssim(img1, img2):
    """Calculate SSIM (structural similarity) for one channel images.

    It is called by func:`calculate_ssim`.

    Args:
        img1 (ndarray): Images with range [0, 255] with order 'HWC'.
        img2 (ndarray): Images with range [0, 255] with order 'HWC'.

    Returns:
        float: ssim result.
    """

    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def SSIM(img1, img2):
    """Calculate SSIM (structural similarity).

    Ref:
    Image quality assessment: From error visibility to structural similarity

    The results are the same as that of the official released MATLAB code in
    https://ece.uwaterloo.ca/~z70wang/research/ssim/.

    For three-channel images, SSIM is calculated for each channel and then
    averaged.

    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the SSIM calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: ssim result.
    """

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    ssims = []
    for i in range(img1.shape[0]):
        ssims.append(_ssim(img1[i, ...], img2[i, ...]))
    return np.array(ssims).mean()
