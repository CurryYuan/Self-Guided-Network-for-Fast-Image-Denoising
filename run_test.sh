#!/usr/bin/env sh
#export PATH=/mnt/lustre/share/zhangzhaoyang/anaconda3/bin:$PATH
mkdir -p log
now=$(date +"%Y%m%d_%H%M%S")
srun --partition=sensevideo --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=test --kill-on-bad-exit=1 \
python SGN/test.py --baseroot /mnt/lustre/yuanzhihao/projects/data/DIV2K/ \
--load_name /mnt/lustre/yuanzhihao/projects/Self-Guided-Network-for-Fast-Image-Denoising/SGN_epoch10000_bs8_mu0_sigma30.pth