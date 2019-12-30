import argparse
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
import random
import sys

import dataset


def validate(model, opt):
    # ----------------------------------------
    #       Initialize testing dataset
    # ----------------------------------------

    # Define the dataset
    opt.dataroot = opt.baseroot + 'DIV2K_valid_HR'
    testset = dataset.FullResDenoisingDataset(opt)
    print('The overall number of images equals to %d' % len(testset))

    # Define the dataloader
    dataloader = DataLoader(testset, batch_size=1, pin_memory=True)

    # ----------------------------------------
    #                 Testing
    # ----------------------------------------

    overall_psnr = 0

    for batch_idx, (noisy_img, img) in enumerate(dataloader):
        # To Tensor
        noisy_img = noisy_img.cuda()
        img = img.cuda()

        # Generator output
        recon_img = model(noisy_img)

        # convert to visible image format
        h = img.shape[2]
        w = img.shape[3]
        img = img.cpu().numpy().reshape(3, h, w).transpose(1, 2, 0)
        img = (img + 1) * 128
        img = img.astype(np.uint8)

        noisy_img = noisy_img.cpu().numpy().reshape(3, h, w).transpose(1, 2, 0)
        noisy_img = (noisy_img + 1) * 128
        noisy_img = noisy_img.astype(np.uint8)

        recon_img = recon_img.detach().cpu().numpy().reshape(3, h, w).transpose(1, 2, 0)
        recon_img = (recon_img + 1) * 128
        recon_img = recon_img.astype(np.uint8)

        overall_psnr += cv2.PSNR(recon_img, img)

    return overall_psnr / len(testset)
