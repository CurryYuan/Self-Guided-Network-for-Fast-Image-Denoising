# Self-Guided-Network-for-Fast-Image-Denoising

The PyTorch implementation of SGN (ICCV 2019), modified from https://github.com/zhaoyuzhi/Self-Guided-Network-for-Fast-Image-Denoising

add U-Net and quantization
## Result

Sigma=30, DIV2K		
		
|	|Paper|	1920*1280|	16bit|	8bit|
|---|---|---|---|---|
|Unet|	31.93|	30.38|	31.49|	31.34|
|SGN|	32.21|	30.95|  |   |		