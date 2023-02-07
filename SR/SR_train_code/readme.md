作为超分的训练代码，以 [bsrgan](https://github.com/cszn/BSRGAN) 为基础构建的代码



### Training

1. Put your training high-quality images into `trainsets/trainH` or set `"dataroot_H": "trainsets/trainH"`
3. Train BSRNet
    1. Modify [train_bsrgan_x4_psnr.json](https://github.com/cszn/KAIR/blob/master/options/train_bsrgan_x4_psnr.json) e.g., `"gpu_ids": [0]`, `"dataloader_batch_size": 4`
    2. Training with `DataParallel`
    ```bash
    python main_train_psnr.py --opt options/train_bsrgan_x4_psnr.json
    ```
    2. Training with `DistributedDataParallel` - 4 GPUs
    ```bash
    python -m torch.distributed.launch --nproc_per_node=4 --master_port=1234 main_train_psnr.py --opt options/train_bsrgan_x4_psnr.json  --dist True
    ```
4. Train BSRGAN
    1. Put BSRNet model (e.g., '400000_G.pth') into `superresolution/bsrgan_x4_gan/models`
    2. Modify [train_bsrgan_x4_gan.json](https://github.com/cszn/KAIR/blob/master/options/train_bsrgan_x4_gan.json) e.g., `"gpu_ids": [0]`, `"dataloader_batch_size": 4`
    3. Training with `DataParallel`
    ```bash
    python main_train_gan.py --opt options/train_bsrgan_x4_gan.json
    ```
    3. Training with `DistributedDataParallel` - 4 GPUs
    ```bash
    python -m torch.distributed.launch --nproc_per_node=4 --master_port=1234 main_train_gan.py --opt options/train_bsrgan_x4_gan.json  --dist True
    ```
5. Test BSRGAN model `'xxxxxx_E.pth'` by modified `main_test_bsrgan.py`
    
    1. `'xxxxxx_E.pth'` is more stable than `'xxxxxx_G.pth'`

### Train with BSRGAN+RealSR  
    1. Put BSRNet model (e.g., '400000_G.pth') into `superresolution/bsrgan_x4_gan/models`
        2. Modify [train_bsrgan_x4_gan.json](https://github.com/cszn/KAIR/blob/master/options/train_bsrgan_x4_gan.json) e.g., `"gpu_ids":[0]`,`"dataloader_batch_size":4`,`"realsr_noise":"./trainsets/realsr/DPEDiphone_noise"`,`"realsr_dataroot_GT":"./trainsets/realsr/HR"`,`"realsr_dataroot_LQ":"./trainsets/realsr/LR"`,`"data_temp": "./trainsets/temp"`,`噪声、HR图、LR图、中间图片保存文件夹`
        3. Training with `DataParallel`
    ```bash
    python main_train_gan_add_realsr.py --opt options/train_bsrgan_x4_gan_realsr.json
    ```
    3. Training with `DistributedDataParallel` - 4 GPUs
    ```bash
    python -m torch.distributed.launch --nproc_per_node=4 --master_port=1234 main_train_gan_add_realsr.py --opt options/train_bsrgan_x4_gan_realsr.json  --dist True
    ```
