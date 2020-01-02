import argparse
import os
import torch

import trainer

if __name__ == "__main__":
    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    # Pre-train, saving, and loading parameters According to paper, the recommend setting is to train the model for
    # 1M iterations, so it is better to save at each 100K iterations The epoch is large enough that the model can be
    # trained more than 1M iterations, users could stop it if it is well trained The learning rate is set to 1e-4
    # during first 500K iterations, while it is 1e-5 during last 500K iterations For DIV2K dataset: epoch 10000 +
    # batch_size 8 = iteration 1000000; I recommend to save 10 models for the whole training stage
    parser.add_argument('--model', type=str, default='SGN', help='use model SGN or UNet')
    parser.add_argument('--save_interval', type=int, default=1000, help='save interval (by epochs)')
    parser.add_argument('--validate_interval', type=int, default=10, help='validate interval (by epochs)')
    parser.add_argument('--load', type=str, default='', help='load the pre-trained model with certain epoch')
    # GPU parameters
    parser.add_argument('--multi_gpu', type=bool, default=True, help='True for more than 1 GPU')
    parser.add_argument('--cudnn_benchmark', type=bool, default=True, help='True for unchanged input data type')
    # Training parameters
    parser.add_argument('--epochs', type=int, default=10000,
                        help='number of epochs of training that ensures 100K training iterations')
    parser.add_argument('--batch_size', type=int, default=8, help='size of the batches, 8 is recommended')
    parser.add_argument('--lr', type=float, default=0.0001, help='Adam: learning rate')
    parser.add_argument('--b1', type=float, default=0.9, help='Adam: decay of first order momentum of gradient')
    parser.add_argument('--b2', type=float, default=0.999, help='Adam: decay of second order momentum of gradient')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay for optimizer')
    parser.add_argument('--iter_decreased', type=int, default=500000, help='the certain iteration that lr decreased')
    parser.add_argument('--lr_decreased', type=float, default=0.00001, help='decreased learning rate at certain epoch')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='number of cpu threads to use during batch generation')
    # Initialization parameters
    parser.add_argument('--pad', type=str, default='reflect', help='pad type of networks')
    parser.add_argument('--norm', type=str, default='none', help='normalization type of networks')
    parser.add_argument('--in_channels', type=int, default=3, help='input channels for generator')
    parser.add_argument('--out_channels', type=int, default=3, help='output channels for generator')
    parser.add_argument('--start_channels', type=int, default=32, help='start channels for generator')
    parser.add_argument('--m_block', type=int, default=2, help='the additional blocks used in mainstream')
    parser.add_argument('--init_type', type=str, default='normal', help='initialization type of generator')
    parser.add_argument('--init_gain', type=float, default=0.02, help='initialization gain of generator')
    # Dataset parameters
    parser.add_argument('--baseroot', type=str, default='/data/DIV2K/',
                        help='images baseroot')
    parser.add_argument('--crop_size', type=int, default=256, help='single patch size')
    parser.add_argument('--geometry_aug', type=bool, default=False, help='geometry augmentation (scaling)')
    parser.add_argument('--angle_aug', type=bool, default=True, help='geometry augmentation (rotation, flipping)')
    parser.add_argument('--scale_min', type=float, default=1, help='min scaling factor')
    parser.add_argument('--scale_max', type=float, default=1, help='max scaling factor')
    parser.add_argument('--mu', type=int, default=0, help='Gaussian noise mean')
    parser.add_argument('--sigma', type=int, default=30, help='Gaussian noise variance: 30 | 50 | 70')

    opt = parser.parse_args()

    # ----------------------------------------
    #        Choose CUDA visible devices
    # ----------------------------------------
    if opt.multi_gpu:
        device_count = torch.cuda.device_count()
        print('Multi-GPU mode, %s GPUs are used' % device_count)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print('Single-GPU mode')

    # ----------------------------------------
    #                 Trainer
    # ----------------------------------------
    trainer.Trainer(opt)
