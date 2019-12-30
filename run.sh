#!/usr/bin/env sh
#export PATH=/mnt/lustre/share/zhangzhaoyang/anaconda3/bin:$PATH
mkdir -p log
now=$(date +"%Y%m%d_%H%M%S")
srun --partition=sensevideo --gres=gpu:2 -n1 --ntasks-per-node=1 --job-name=UNet --kill-on-bad-exit=1 \
python SGN/train.py --baseroot /mnt/lustre/yuanzhihao/projects/data/DIV2K/ --model UNet --num_workers 16 --lr 0.00001 \
--load /mnt/lustre/yuanzhihao/projects/Self-Guided-Network-for-Fast-Image-Denoising/UNet_epoch10000_bs8_mu0_sigma30.pth