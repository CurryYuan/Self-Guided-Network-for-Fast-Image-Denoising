import time
import datetime
import itertools
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.autograd as autograd
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

import dataset
import utils
import validation
import unet

from torch.utils.tensorboard import SummaryWriter

from spring.dirichlet import dirichlet, set_logger_path, set_tb_logger
from spring.dirichlet.convert import pytorch_to_caffe
import runpy
import yaml
import os


def Trainer(opt):
    # ----------------------------------------
    #       Network training parameters
    # ----------------------------------------

    # cudnn benchmark
    cudnn.benchmark = opt.cudnn_benchmark

    # Loss functions
    # criterion_L1 = torch.nn.L1Loss().cuda()
    # criterion = MS_SSIM_Loss(data_range=1.0, size_average=True, channel=3).cuda()
    criterion = nn.MSELoss().cuda()

    # Initialize model
    if opt.model == 'SGN':
        model = utils.create_generator(opt)
    elif opt.model == 'UNet':
        if opt.load:
            model = torch.load(opt.load).module
            print(f'Model loaded from {opt.load}')
        else:
            model = unet.UNet(opt.in_channels, opt.out_channels, opt.start_channels)
    else:
        raise NotImplementedError(opt.model + 'is not implemented')

    dir_checkpoint = 'checkpoints/'
    try:
        os.mkdir(dir_checkpoint)
        print('Created checkpoint directory')
    except OSError:
        pass

    writer = SummaryWriter(comment=f'_{opt.model}_LR_{opt.lr}_BS_{opt.batch_size}')

    # To device
    if opt.multi_gpu:
        model = nn.DataParallel(model)
        model = model.cuda()
    else:
        model = model.cuda()

    # Optimizers
    optimizer_G = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2),
                                   weight_decay=opt.weight_decay)

    # Learning rate decrease
    def adjust_learning_rate(opt, iteration, optimizer):
        # Set the learning rate to the specific value
        if iteration >= opt.iter_decreased:
            for param_group in optimizer.param_groups:
                param_group['lr'] = opt.lr_decreased

    # Save the model if pre_train == True
    def save_model(opt, epoch, iteration, len_dataset, network):
        """Save the model at "checkpoint_interval" and its multiple"""
        if (epoch % opt.save_interval == 0) and (iteration % len_dataset == 0):
            if opt.quantize:
                torch.save(network.module.state_dict(), dir_checkpoint +
                           'quant_8_%s_epoch%d_bs%d_mu%d_sigma%d.pth' % (
                           opt.model, epoch, opt.batch_size, opt.mu, opt.sigma))
            else:
                torch.save(network, dir_checkpoint +
                           '%s_epoch%d_bs%d_mu%d_sigma%d.pth' % (
                               opt.model, epoch, opt.batch_size, opt.mu, opt.sigma))
            print('The trained model is successfully saved at epoch %d' % (epoch))

        if (epoch % opt.validate_interval == 0) and (iteration % len_dataset == 0):
            psnr = validation.validate(network, opt)
            print('validate PSNR:', psnr)
            writer.add_scalar('PSNR/validate', psnr, iteration)

    # ----------------------------------------
    #             Network dataset
    # ----------------------------------------

    # Define the dataset
    opt.dataroot = opt.baseroot + 'DIV2K_train_HR'
    trainset = dataset.DenoisingDataset(opt)
    print('The overall number of images:', len(trainset))

    # Define the dataloader
    dataloader = DataLoader(trainset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers,
                            pin_memory=True)

    # ----------------------------------------
    #                 Training
    # ----------------------------------------

    # Count start time
    prev_time = time.time()

    # For loop training
    for epoch in range(opt.epochs):
        for i, (noisy_img, img) in enumerate(dataloader):
            # To device
            noisy_img = noisy_img.cuda()
            img = img.cuda()

            # Train model
            optimizer_G.zero_grad()

            # Forword propagation
            recon_img = model(noisy_img)
            loss = criterion(recon_img, img)

            # Overall Loss and optimize
            loss.backward()
            optimizer_G.step()

            # Determine approximate time left
            iters_done = epoch * len(dataloader) + i
            iters_left = opt.epochs * len(dataloader) - iters_done
            time_left = datetime.timedelta(seconds=iters_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            print("\r[Epoch %d/%d] [Batch %d/%d] [Recon Loss: %.4f] Time_left: %s" %
                  ((epoch + 1), opt.epochs, i, len(dataloader), loss.item(), time_left))

            writer.add_scalar('Loss/train', loss.item(), iters_done)

            # Save model at certain epochs or iterations
            save_model(opt, (epoch + 1), (iters_done + 1), len(dataloader), model)

            # Learning rate decrease at certain epochs
            adjust_learning_rate(opt, (iters_done + 1), optimizer_G)
