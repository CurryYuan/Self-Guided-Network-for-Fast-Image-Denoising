import argparse
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
import random
import sys

import dataset


if __name__ == "__main__":

    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    # Dataset parameters
    parser.add_argument('--baseroot', type=str, default="/data/DIV2K/", help='the testing folder')
    parser.add_argument('--geometry_aug', type=bool, default=False, help='geometry augmentation (scaling)')
    parser.add_argument('--angle_aug', type=bool, default=False, help='geometry augmentation (rotation, flipping)')
    parser.add_argument('--scale', type=float, default=1.0, help='scaling factor')
    parser.add_argument('--mu', type=float, default=0, help='min scaling factor')
    parser.add_argument('--sigma', type=float, default=30, help='max scaling factor')
    # Other parameters
    parser.add_argument('--batch_size', type=int, default=1, help='test batch size, always 1')
    parser.add_argument('--load_name', type=str, default='../SGN_epoch30_bs8_mu0_sigma30.pth', help='test model name')
    opt = parser.parse_args()
    print(opt)

    # ----------------------------------------
    #       Initialize testing dataset
    # ----------------------------------------

    # Define the dataset
    opt.dataroot = opt.baseroot + 'DIV2K_valid_HR'
    testset = dataset.FullResDenoisingDataset(opt)
    print('The overall number of images equals to %d' % len(testset))

    # Define the dataloader
    dataloader = DataLoader(testset, batch_size=opt.batch_size, pin_memory=True)

    # ----------------------------------------
    #                 Testing
    # ----------------------------------------

    model = torch.load(opt.load_name)

    ave_psnr = 0

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

        ave_psnr += cv2.PSNR(recon_img, img)

        # show
        if batch_idx == random.randint(0, len(testset)-1) and False:
            print('noisy PSNR:', cv2.PSNR(noisy_img, img))
            print('denoisy PSNR:', cv2.PSNR(recon_img, img))

            show_img = np.concatenate((img, recon_img), axis=1)
            r, g, b = cv2.split(show_img)
            show_img = cv2.merge([b, g, r])
            # cv2.namedWindow('display', cv2.WINDOW_NORMAL)
            cv2.imshow('display', show_img)
            # cv2.imwrite('result_%d.jpg' % batch_idx, show_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    print('average PSNR:', ave_psnr / len(testset))
