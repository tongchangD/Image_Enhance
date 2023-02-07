# -*- encoding: utf-8 -*-
'''
@File    :   make_test_data   
@brief   :   
@License :   
@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/6/8 下午4:20   tongchangD      1.0         None
'''
from utils import utils_blindsr as blindsr
import utils.utils_image as util
import cv2
import numpy as np
import random
import os
def makedata(PATH,output):
    print("begin")
    if not os.path.exists(output + "_HR"):
        os.makedirs(output + "_HR")
    if not os.path.exists(output + "_LR"):
        os.makedirs(output + "_LR")

    for filename in os.listdir(PATH):
        file=os.path.join(PATH,filename)
        sf = 4
        n_channels = 3
        img_H = util.imread_uint(file, n_channels)
        H, W, C = img_H.shape
        patch_size = int(min(H , W)/4)
        rnd_h_H = random.randint(0, max(0, H - patch_size*4))
        rnd_w_H = random.randint(0, max(0, W - patch_size*4))
        img_H = img_H[rnd_h_H:rnd_h_H+patch_size*4, rnd_w_H:rnd_w_H+patch_size*4, :]
        img_H = util.uint2single(img_H)
        img_L, img_H = blindsr.degradation_bsrgan(img_H, sf, lq_patchsize=patch_size,isp_model=None)
        img = cv2.imread(file)
        util.imsave(util.single2uint(img_H), os.path.join(output + "_HR",filename))
        util.imsave(util.single2uint(img_L), os.path.join(output + "_LR", filename))

        # cv2.imwrite(os.path.join(output + "bsrgan_HR",filename), np.clip(img_H*255, 0, 255))
        # cv2.imwrite(os.path.join(output + "bsrgan_LR", filename), np.clip(img_H*255, 0, 255))

PATH = "/home/tongchangdong/DATA/super_resolution_test_data/现代彩图修复/bsrgan"
makedata(PATH,PATH)

print("done")
