import os.path
import logging
import torch
from glob import glob
import cv2
from utils import utils_logger
from utils import utils_image as util
# from utils import utils_model
from models.network_rrdbnet import RRDBNet as net


"""
Spyder (Python 3.6-3.7)
PyTorch 1.4.0-1.8.1
Windows 10 or Linux
Kai Zhang (cskaizhang@gmail.com)
github: https://github.com/cszn/BSRGAN
        https://github.com/cszn/KAIR
If you have any question, please feel free to contact with me.
Kai Zhang (e-mail: cskaizhang@gmail.com)
by Kai Zhang ( March/2020 --> March/2021 --> )
This work was previously submitted to CVPR2021.

# --------------------------------------------
@inproceedings{zhang2021designing,
  title={Designing a Practical Degradation Model for Deep Blind Image Super-Resolution},
  author={Zhang, Kai and Liang, Jingyun and Van Gool, Luc and Timofte, Radu},
  booktitle={arxiv},
  year={2021}
}
# --------------------------------------------

"""


def main_old():

    utils_logger.logger_info('blind_sr_log', log_path='blind_sr_log.log')
    logger = logging.getLogger('blind_sr_log')

#    print(torch.__version__)               # pytorch version
#    print(torch.version.cuda)              # cuda version
#    print(torch.backends.cudnn.version())  # cudnn version
    "/media/tcd/data/github/CV/code/CV_project/浙江档案的网址/SR_data_1541"
    testsets = '/media/tcd/data/work/Shanghai_Archives_Bureau/实验结果/第四次实验结果/档案局小原图'       # fixed, set path of testsets
    testset_Ls = ['no_zoom_img']  # ['RealSRSet','DPED']

    model_names = ['RRDB','ESRGAN','FSSR_DPED','FSSR_JPEG','RealSR_DPED','RealSR_JPEG']
    model_names = ['20000_G']    # 'BSRGANx2' for scale factor 2

    save_results = True
    sf = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for model_name in model_names:
        if model_name in ['BSRGANx2']:
            sf = 2
        model_path = os.path.join('model_zoo', model_name+'.pth')          # set model path
        # model_path = "/home/tcd/KAIR_superresolution/bsrgan_x4_gan/models/210000_G.pth"
        model_path = "/media/tcd/data/work/Shanghai_Archives_Bureau/bsrgan/model_zoo/1599_训练结果/BSRNet/480000_G.pth"
        logger.info('{:>16s} : {:s}'.format('Model Name', model_name))

        # torch.cuda.set_device(0)      # set GPU ID
        logger.info('{:>16s} : {:<d}'.format('GPU ID', torch.cuda.current_device()))
        torch.cuda.empty_cache()

        # --------------------------------
        # define network and load model
        # --------------------------------
        model = net(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=sf)  # define network

#            model_old = torch.load(model_path)
#            state_dict = model.state_dict()
#            for ((key, param),(key2, param2)) in zip(model_old.items(), state_dict.items()):
#                state_dict[key2] = param
#            model.load_state_dict(state_dict, strict=True)

        model.load_state_dict(torch.load(model_path), strict=True)
        model.eval()
        for k, v in model.named_parameters():
            v.requires_grad = False
        model = model.to(device)
        torch.cuda.empty_cache()

        for testset_L in testset_Ls:

            L_path = os.path.join(testsets, testset_L)
            #E_path = os.path.join(testsets, testset_L+'_'+model_name)
            E_path = os.path.join(testsets, testset_L+'_results_x'+str(sf))
            util.mkdir(E_path)

            logger.info('{:>16s} : {:s}'.format('Input Path', L_path))
            logger.info('{:>16s} : {:s}'.format('Output Path', E_path))
            idx = 0

            for img in util.get_image_paths(L_path):
                try:
                    # --------------------------------
                    # (1) img_L
                    # --------------------------------
                    idx += 1
                    img_name, ext = os.path.splitext(os.path.basename(img))
                    logger.info('{:->4d} --> {:<s} --> x{:<d}--> {:<s}'.format(idx, model_name, sf, img_name+ext))

                    img_L = util.imread_uint(img, n_channels=3)
                    img_L = util.uint2tensor4(img_L)
                    img_L = img_L.to(device)

                    # --------------------------------
                    # (2) inference
                    # --------------------------------
                    img_E = model(img_L)

                    # --------------------------------
                    # (3) img_E
                    # --------------------------------
                    img_E = util.tensor2uint(img_E)
                    if save_results:
                        util.imsave(img_E, os.path.join(E_path, img_name+'_'+model_name+'.png'))
                except Exception as e:
                    print(e)

def main_with_folder():

    # info = utils_logger.logger_info('blind_sr_log', log_path='blind_sr_log.log')
    logger = logging.getLogger('blind_sr_log')
#    print(torch.__version__)               # pytorch version
#    print(torch.version.cuda)              # cuda version
#    print(torch.backends.cudnn.version())  # cudnn version

    # model_names = ['RRDB','ESRGAN','FSSR_DPED','FSSR_JPEG','RealSR_DPED','RealSR_JPEG']
    model_path = "/home/tcd/40000_E.pth" # 20211116 最好的 485000_E.pth
    # model_path = "/media/tcd/data/work/Shanghai_Archives_Bureau/bsrgan/model_zoo/BSRGAN.pth" # 原始模型
    model_path = "/media/tcd/data/DATA/super_resolution/BSRGAN/bsrgan_x4_gan_1094_scale12_44/models/230000_E.pth"
    # testsets = "/media/tcd/data/work/Shanghai_Archives_Bureau/测试数据集/上海_picture"
    testsets = '/home/tcd/Desktop/cat'  # fixed, set path of testsets
    # testsets = "/media/tcd/data/DATA/浙江档案/测试集/zyy"
    test_outputs = testsets + "_bsrgan_add_realsr1294_1"
    # model_path = "/media/tcd/data/work/Shanghai_Archives_Bureau/bsrgan/model_zoo/1599_训练结果/bsrgan/760000_E.pth"
    # testsets = "/media/tcd/data/work/Shanghai_Archives_Bureau/测试数据集/上海_picture"
    # testsets = '/media/tcd/data/work/Shanghai_Archives_Bureau/实验结果/source_imgs_132'       # fixed, set path of testsets
    # test_outputs = testsets + "_bsrgan"
    util.mkdir(test_outputs)

    save_results = True
    sf = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if 'x2' in model_path :sf = 2
    # torch.cuda.set_device(0)      # set GPU ID
    logger.info('{:>16s} : {:<d}'.format('GPU ID', torch.cuda.current_device()))
    torch.cuda.empty_cache()
    # --------------------------------
    # define network and load model
    # --------------------------------
    model = net(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=sf)  # define network
    #            model_old = torch.load(model_path)
    #            state_dict = model.state_dict()
    #            for ((key, param),(key2, param2)) in zip(model_old.items(), state_dict.items()):
    #                state_dict[key2] = param
    #            model.load_state_dict(state_dict, strict=True)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)
    torch.cuda.empty_cache()

    # L_path = os.path.join(testsets, testset_L)
    # # E_path = os.path.join(testsets, testset_L+'_'+model_name)
    # E_path = os.path.join(testsets, testset_L + '_results_x' + str(sf))

    logger.info('{:>16s} : {:s}'.format('Input Path', testsets))
    logger.info('{:>16s} : {:s}'.format('Output Path', test_outputs))
    idx = 0

    for img in util.get_image_paths(testsets):
        try:
            print(img)
            # --------------------------------
            # (1) img_L
            # --------------------------------
            idx += 1
            img_name, ext = os.path.splitext(os.path.basename(img))
            logger.info('{:->4d} --> --> x{:<d}--> {:<s}'.format(idx, sf, img_name + ext))
            img_L = util.imread_uint(img, n_channels=3)
            img_L = util.uint2tensor4(img_L)

            img_L = img_L.to(device)
            # --------------------------------
            # (2) inference
            # --------------------------------
            img_E = model(img_L)
            # --------------------------------
            # (3) img_E
            # --------------------------------
            img_E = util.tensor2uint(img_E)
            if save_results:
                # util.imsave(img_E, os.path.join(test_outputs, img_name +'.png'))
                util.imsave(img_E, os.path.join(test_outputs, img_name + '_' + os.path.split(model_path)[-1][:-4] + '.png'))
        except RuntimeError as e:
            print(e)
            img_name, ext = os.path.splitext(os.path.basename(img))
            logger.info('{:->4d} --> --> x{:<d}--> {:<s}'.format(idx, sf, img_name + ext))
            img_L = util.imread_uint(img, n_channels=3)
            img_L=img_L[400:800,400:800,:]
            img_L = util.uint2tensor4(img_L)

            img_L = img_L.to(device)
            # --------------------------------
            # (2) inference
            # --------------------------------
            img_E = model(img_L)
            # --------------------------------
            # (3) img_E
            # --------------------------------
            img_E = util.tensor2uint(img_E)
            if save_results:
                # util.imsave(img_E, os.path.join(test_outputs, img_name +'.png'))
                util.imsave(img_E, os.path.join(test_outputs,
                                                img_name + '_' + os.path.split(model_path)[-1][:-4] + '.png'))
            print(os.path.join(test_outputs,img_name + '_' + os.path.split(model_path)[-1][:-4] + '.png'))


def main_with_file_and_show(model_path, img_path):
    # utils_logger.logger_info('blind_sr_log', log_path='blind_sr_log.log')
    # logger = logging.getLogger('blind_sr_log')
    #    print(torch.__version__)               # pytorch version
    #    print(torch.version.cuda)              # cuda version
    #    print(torch.backends.cudnn.version())  # cudnn version

    sf = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if 'x2' in model_path: sf = 2
    # torch.cuda.set_device(0)      # set GPU ID
    torch.cuda.empty_cache()
    # --------------------------------
    # define network and load model
    # --------------------------------
    model = net(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=sf)  # define network
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)
    torch.cuda.empty_cache()
    print(model_path)
    print(img_path)
    try:
        # --------------------------------
        # (1) img_L
        # --------------------------------
        img_L = util.imread_uint(img_path, n_channels=3)
        img_L = util.uint2tensor4(img_L)
        img_L = img_L.to(device)
        # --------------------------------
        # (2) inference
        # --------------------------------
        img_E = model(img_L)
        # --------------------------------
        # (3) img_E
        # --------------------------------
        img_L = util.tensor2uint(img_L)
        img_E = util.tensor2uint(img_E)
        util.cv2show(img_L, "source")
        util.cv2show(img_E, "super_resolution")
        cv2.waitKey(0)
        # util.imsave(img_E, "./temp.png")
    except Exception as e:
        print(e)

def main_all_model_with_same_images():
    model_path1 = "/media/tcd/data/DATA/super_resolution/BSRGAN/bsrgan_add_realsr_544add544/models/*_G.pth"
    model_path2 = "/media/tcd/data/DATA/super_resolution/BSRGAN/bsrgan_add_realsr_544add544/models/*_E.pth"
    images_path="/media/tcd/data/DATA/浙江档案/测试集/1"
    images=glob(images_path+"/*")
    out_pth=images_path+"_all"
    model_paths = glob(model_path1)
    model_paths += glob(model_path2)
    if not os.path.exists(out_pth):
        os.makedirs(out_pth)

    sf = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # torch.cuda.set_device(0)      # set GPU ID
    torch.cuda.empty_cache()
    # --------------------------------
    # define network and load model
    # --------------------------------
    model = net(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=sf)  # define network
    for model_path in model_paths:
        print("model_path",model_path)
        model.load_state_dict(torch.load(model_path), strict=True)
        model.eval()
        for k, v in model.named_parameters():
            v.requires_grad = False
        model = model.to(device)
        torch.cuda.empty_cache()
        for image in images:
            try:
                # --------------------------------
                # (1) img_L
                # --------------------------------
                img_name, ext = os.path.splitext(os.path.basename(image))
                img_L = util.imread_uint(image, n_channels=3)
                img_L = util.uint2tensor4(img_L)
                img_L = img_L.to(device)
                # --------------------------------
                # (2) inference
                # --------------------------------
                img_E = model(img_L)
                # --------------------------------
                # (3) img_E
                # --------------------------------
                img_E = util.tensor2uint(img_E)
                util.imsave(img_E, os.path.join(out_pth,
                                                img_name + '_' + os.path.split(model_path)[-1][:-4] + '.png'))
            except Exception as e:
                print(e)
if __name__ == '__main__':

    # main_old()
    main_with_folder()
    # model_path="/media/tcd/data/DATA/super_resolution/BSRGAN/bsrgan_add_realsr_544add544/models/170000_E.pth"
    # filename="/media/tcd/data/Other/头像/tcd.jpg"
    # main_with_file_and_show(model_path,filename)

    # main_all_model_with_same_images()

    print("done")