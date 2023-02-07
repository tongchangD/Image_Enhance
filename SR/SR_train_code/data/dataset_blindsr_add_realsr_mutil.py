import os

import cv2
import math
import glob
import torch
import random
import numpy as np
from PIL import Image
import torch.utils.data as data
import utils.utils_image as util
import utils.utils_realsr as realsr_util
from utils import utils_blindsr as blindsr
from torchvision import transforms
from torchvision.utils import make_grid


class noiseDataset_mutil(data.Dataset):
    def __init__(self, dataset='x2/', size=32, other=False):
        super(noiseDataset_mutil, self).__init__()
        base = dataset
        import os
        assert os.path.exists(base)
        self.other_noise = other
        self.size = size
        # 遍历 base 文件
        self.noise_imgs={}
        for noise_folder_name in os.listdir(base):
            self.noise_imgs[noise_folder_name] = sorted(glob.glob(os.path.join(base,noise_folder_name, '*.png')))
        self.pre_process = transforms.Compose([transforms.RandomCrop(size),
                                               transforms.ToTensor()])

    def __getitem__(self, noise_folfer_name):
        index = np.random.randint(0, len(self.noise_imgs[noise_folfer_name]))  # 此处加的噪声
        noise = self.pre_process(Image.open(self.noise_imgs[noise_folfer_name][index]))
        norm_noise = (noise - torch.mean(noise, dim=[1, 2], keepdim=True))
        return norm_noise

    def __len__(self):
        return len(self.noise_imgs)


class DatasetBlindSR_add_RealSR_mutil(data.Dataset):
    '''
    # -----------------------------------------
    # dataset for BSRGAN  and realsr
    # -----------------------------------------
    '''

    def __init__(self, opt):
        super(DatasetBlindSR_add_RealSR_mutil, self).__init__()
        self.opt = opt
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 3
        self.sf = opt['scale'] if opt['scale'] else 4
        self.shuffle_prob = opt['shuffle_prob'] if opt['shuffle_prob'] else 0.1
        self.use_sharp = opt['use_sharp'] if opt['use_sharp'] else False
        self.degradation_type = opt['degradation_type'] if opt['degradation_type'] else 'bsrgan'
        self.lq_patchsize = self.opt['lq_patchsize'] if self.opt['lq_patchsize'] else 64
        self.patch_size = self.opt['H_size'] if self.opt['H_size'] else self.lq_patchsize * self.sf

        ############ bsrgan
        self.paths_H = util.get_image_paths(opt['dataroot_H'])

        print("opt\n", opt)
        ############ realsr
        if self.opt["USE_realsr"]:
            self.data_type = self.opt['data_type']
            self.paths_GT = []
            self.paths_LQ = []
            for dirpath, _, fnames in sorted(os.walk(self.opt['realsr_dataroot_GT'])):
                if fnames != []:
                    self.temp, _ = realsr_util.get_image_paths(self.data_type, dirpath)  # realsr HR
                    self.paths_GT += self.temp
            self.sizes_GT = len(self.paths_GT)
            for dirpath, _, fnames in sorted(os.walk(self.opt['realsr_dataroot_LQ'])):
                if fnames != []:
                    self.temp, _ = realsr_util.get_image_paths(self.data_type, dirpath)  # realsr HR
                    self.paths_LQ += self.temp
            self.sizes_LQ = len(self.paths_LQ)
            # realsr LR 保证 HR图和LR图相一致

            self.noises = noiseDataset_mutil(opt['realsr_noise'], opt['H_size'] / opt['scale'], other=True)
            util.mkdir(self.opt['data_temp'])

            # 考虑 1:1
            # random_sample=random.sample([i for i in range(len(self.paths_H))],self.sizes_GT)
            # self.paths_HR = [self.paths_H[i] for i in random_sample]+self.paths_GT
            # self.paths_LR = ["" for i in range(self.sizes_GT)] + self.paths_LQ
            # self.paths_HR = [self.paths_H[i] for i in random_sample]+self.paths_GT
            # self.paths_LR = ["" for i in range(self.sizes_GT)] + self.paths_LQ

            # 依照 制作的数据集 数量
            self.paths_HR = self.paths_H + self.paths_GT
            self.paths_LR = ["" for i in range(len(self.paths_H))] + self.paths_LQ

            random_ = list(zip(self.paths_HR, self.paths_LR))
            random.shuffle(random_)
            self.paths_HR[:], self.paths_LR[:] = zip(*random_)
        else:
            self.paths_HR = self.paths_H
            self.paths_LR = ["" for i in range(len(self.paths_H))]

        assert self.paths_HR, 'Error: H path is empty.'

    def __getitem__(self, index):
        # bsrgan
        if self.paths_LR[index] == "":
            L_path = None
            # ------------------------------------
            # get H image
            # ------------------------------------
            H_path = self.paths_HR[index]
            img_H = util.imread_uint(H_path, self.n_channels)
            img_name, ext = os.path.splitext(os.path.basename(H_path))
            H, W, C = img_H.shape
            if H < self.patch_size or W < self.patch_size:  # 如果 图尺寸 比 self.patch_size 小 提供纯色图
                img_H = np.tile(np.random.randint(0, 256, size=[1, 1, self.n_channels], dtype=np.uint8),
                                (self.patch_size, self.patch_size, 1))

            # ------------------------------------
            # if train, get L/H patch pair
            # ------------------------------------
            if self.opt['phase'] == 'train':
                H, W, C = img_H.shape
                rnd_h_H = random.randint(0, max(0, H - self.patch_size))
                rnd_w_H = random.randint(0, max(0, W - self.patch_size))
                img_H = img_H[rnd_h_H:rnd_h_H + self.patch_size, rnd_w_H:rnd_w_H + self.patch_size, :]

                if 'face' in img_name:
                    mode = random.choice([0, 4])
                    img_H = util.augment_img(img_H, mode=mode)
                else:
                    mode = random.randint(0, 7)
                    img_H = util.augment_img(img_H, mode=mode)
                img_H = util.uint2single(img_H)
                if self.degradation_type == 'bsrgan':
                    img_L, img_H = blindsr.degradation_bsrgan(img_H, self.sf, lq_patchsize=self.lq_patchsize,
                                                              isp_model=None)
                elif self.degradation_type == 'bsrgan_plus':
                    img_L, img_H = blindsr.degradation_bsrgan_plus(img_H, self.sf, shuffle_prob=self.shuffle_prob,
                                                                   use_sharp=self.use_sharp,
                                                                   lq_patchsize=self.lq_patchsize)

            else:
                img_H = util.uint2single(img_H)
                if self.degradation_type == 'bsrgan':
                    img_L, img_H = blindsr.degradation_bsrgan(img_H, self.sf, lq_patchsize=self.lq_patchsize,
                                                              isp_model=None)
                elif self.degradation_type == 'bsrgan_plus':
                    img_L, img_H = blindsr.degradation_bsrgan_plus(img_H, self.sf, shuffle_prob=self.shuffle_prob,
                                                                   use_sharp=self.use_sharp,
                                                                   lq_patchsize=self.lq_patchsize)
            if self.opt['phase'] == 'train':
                util.imsave(util.single2uint(img_L), os.path.join(self.opt['data_temp'], img_name + "_BSRGAN_LR.png"))
                util.imsave(util.single2uint(img_H), os.path.join(self.opt['data_temp'], img_name + "_BSRGAN_HR.png"))
            # ------------------------------------
            # L/H pairs, HWC to CHW, numpy to tensor
            # ------------------------------------
            img_H, img_L = util.single2tensor3(img_H), util.single2tensor3(img_L)

            if L_path is None:
                L_path = H_path

        # realsr
        else:
            # GT_path, LQ_path = None, None
            scale = self.opt['scale']
            GT_size = self.opt['H_size']
            # get GT image
            H_path = self.paths_HR[index]
            img_H = realsr_util.read_img(None, H_path, None)
            # modcrop in the validation / test phase
            if self.opt['phase'] != 'train':
                img_H = realsr_util.modcrop(img_H, scale)
            # change color space if necessary
            if self.opt['color']:
                img_H = realsr_util.channel_convert(img_H.shape[2], self.opt['color'], [img_H])[0]
            # get LQ image
            if self.paths_LR:
                L_path = self.paths_LR[index]
                resolution = None
                ############### 获取 LR img
                # ''' # source
                img_L = realsr_util.read_img(None, L_path, None)
                # '''
                ###############
            else:  # down-sampling on-the-fly  在线 降采样
                # randomly scale during training
                if self.opt['phase'] == 'train':
                    random_scale = random.choice([1])
                    H_s, W_s, _ = img_H.shape

                    def _mod(n, random_scale, scale, thres):
                        rlt = int(n * random_scale)
                        rlt = (rlt // scale) * scale
                        return thres if rlt < thres else rlt

                    H_s = _mod(H_s, random_scale, scale, GT_size)
                    W_s = _mod(W_s, random_scale, scale, GT_size)
                    img_H = cv2.resize(np.copy(img_H), (W_s, H_s), interpolation=cv2.INTER_LINEAR)
                    # force to 3 channels
                    if img_H.ndim == 2:
                        img_H = cv2.cvtColor(img_H, cv2.COLOR_GRAY2BGR)
                """
                #######################################################
                # 定义参数，参数初始化
                ################################################
                # 随机 kernel
                kernel_size = random.choice([2 * v + 1 for v in range(3, 11)])
                if np.random.uniform() < 0.1:
                    # this sinc filter setting is for kernels ranging from [7, 21]
                    if kernel_size < 13:
                        omega_c = np.random.uniform(np.pi / 3, np.pi)
                    else:
                        omega_c = np.random.uniform(np.pi / 5, np.pi)
                    kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
                else:
                    kernel_list = ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso',
                                   'plateau_aniso']
                    kernel_prob = [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]
                    blur_sigma = [0.2, 3]
                    betag_range = [0.5, 4]
                    betap_range = [1, 2]
                    kernel = random_mixed_kernels(kernel_list, kernel_prob, kernel_size, blur_sigma, blur_sigma,
                                                  [-math.pi, math.pi], betag_range, betap_range,
                                                  noise_range=None)
                # pad kernel
                pad_size = (21 - kernel_size) // 2
                kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))
                if np.random.uniform() < 0.8:
                    kernel_size = random.choice([2 * v + 1 for v in range(3, 11)])
                    omega_c = np.random.uniform(np.pi / 3, np.pi)
                    sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
                    sinc_kernel = torch.FloatTensor(sinc_kernel)
                else:
                    sinc_kernel = torch.zeros(21, 21).float()  # convolving with pulse tensor brings no blurry effect
                    sinc_kernel[10, 10] = 1
                ###############################################
                img_L = self.usm_sharpener(img_H)
                # 模糊
                if np.random.uniform() < 0.8:
                    img_L = filter2D(img_L, kernel)
                # 下采样
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                img_L = F.interpolate(img_L, size=(
                img_H.size()[0] // self.opt['scale'], img_H.size()[1] // self.opt['scale']), mode=mode)
                # 振铃
                if np.random.uniform() < 0.5:
                    img_L = filter2D(img_L, sinc_kernel)
                #######################################################
                # """

                # """ source 在线下采样
                ##############################################
                H, W, _ = img_H.shape
                # using matlab imresize
                img_L = realsr_util.imresize_np(img_H, 1 / scale, True)
                ##############################################
                # """
                if img_L.ndim == 2:
                    img_L = np.expand_dims(img_L, axis=2)
            # print(img_H.shape,img_L.shape)
            if self.opt['phase'] == 'train':
                # if the image size is too small
                H, W, _ = img_H.shape
                if H < GT_size or W < GT_size:
                    img_H = cv2.resize(np.copy(img_H), (GT_size, GT_size),
                                       interpolation=cv2.INTER_LINEAR)
                    # using matlab imresize
                    img_L = realsr_util.imresize_np(img_H, 1 / scale, True)
                    if img_L.ndim == 2:
                        img_L = np.expand_dims(img_L, axis=2)
                H, W, C = img_L.shape
                LQ_size = GT_size // scale

                # randomly crop
                rnd_h = random.randint(0, max(0, H - LQ_size))
                rnd_w = random.randint(0, max(0, W - LQ_size))
                img_L = img_L[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :]
                rnd_h_GT, rnd_w_GT = int(rnd_h * scale), int(rnd_w * scale)
                img_H = img_H[rnd_h_GT:rnd_h_GT + GT_size, rnd_w_GT:rnd_w_GT + GT_size, :]

                # augmentation - flip, rotate
                img_L, img_H = realsr_util.augment([img_L, img_H], self.opt['use_flip'],
                                                   self.opt['use_rot'])

            # change color space if necessary
            if self.opt['color']:
                img_L = realsr_util.channel_convert(C, self.opt['color'],
                                                    [img_L])[0]  # TODO during val no definition

            # BGR to RGB, HWC to CHW, numpy to tensor
            if img_H.shape[2] == 3:
                img_H = img_H[:, :, [2, 1, 0]]
                img_L = img_L[:, :, [2, 1, 0]]
            img_H = torch.from_numpy(np.ascontiguousarray(np.transpose(img_H, (2, 0, 1)))).float()
            img_L = torch.from_numpy(np.ascontiguousarray(np.transpose(img_L, (2, 0, 1)))).float()

            # """
            #########保存中间LR图 begin
            if self.opt['phase'] == 'train':
                img_temp = self.tensor2img(img_L)
                self.save_img(img_temp,
                              L_path.replace(self.opt["realsr_dataroot_LQ"], self.opt["data_temp"])[:-4] + "_LR.jpg")
            #########保存中间LR图 end

            ############################噪声添加###################################
            if self.opt['phase'] == 'train':
                # realsr 源代码 ########
                # add noise to LR during train
                if self.opt['aug'] and 'noise' in self.opt['aug']:
                    noise=self.noises[self.paths_HR[index].split("/")[-2]]
                    # noise = noises_[np.random.randint(0, len(self.noises))]  # 此处加的噪声
                    # noise = self.noises[np.random.randint(0, len(self.noises))]  # 此处 加的噪声
                    # print("img_L",img_L.shape,"noise",noise.shape)
                    img_L = torch.clamp(img_L + noise, 0, 1)
            # """
            # if L_path is None:
            #     L_path = L_path
            #########保存中间图
            if self.opt['phase'] == 'train':
                img_temp = self.tensor2img(img_L)
                self.save_img(img_temp, L_path.replace(self.opt["realsr_dataroot_LQ"], self.opt["data_temp"])[
                                        :-4] + "_LR_add_noise.jpg")
                img_temp = self.tensor2img(img_H)
                self.save_img(img_temp,
                              H_path.replace(self.opt["realsr_dataroot_GT"], self.opt["data_temp"])[:-4] + "_HR.jpg")
            #########保存中间图

        #     from data.dataset_realsr import

        return {'L': img_L, 'H': img_H, 'L_path': L_path, 'H_path': H_path}

    def __len__(self):
        return len(self.paths_HR)

    def tensor2img(self, tensor, out_type=np.uint8, min_max=(0, 1)):
        '''
        Converts a torch Tensor into an image Numpy array
        Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
        Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
        '''
        tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
        tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
        n_dim = tensor.dim()
        if n_dim == 4:
            n_img = len(tensor)
            img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
            img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
        elif n_dim == 3:
            img_np = tensor.numpy()
            img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
        elif n_dim == 2:
            img_np = tensor.numpy()
        else:
            raise TypeError(
                'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
        if out_type == np.uint8:
            img_np = (img_np * 255.0).round()
            # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
        return img_np.astype(out_type)

    def save_img(self, img, img_path, mode='RGB'):
        cv2.imwrite(img_path, img)
