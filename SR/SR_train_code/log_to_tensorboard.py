import os
from tensorboardX import SummaryWriter

import re

def read_log(path):
    lines=open(path,"r").readlines()
    logKey={}
    g_loss_min=200
    F_loss_min=200
    D_loss_min=0
    D_real_min=200
    D_fake_min=200
    g_loss_index=0
    F_loss_index=0
    D_loss_index=0
    D_real_index=0
    D_fake_index=0
    for i,line in enumerate(lines):
        if "epoch" in line:
            print(line)
            epoch = int(re.findall('iter:(.*?), lr', line, re.S)[0].replace(",",""))
            lr = float(re.findall('lr:(.*?)> G_loss', line, re.S)[0].replace(",", ""))
            g_loss = float(re.findall('G_loss:(.*?) F_loss', line, re.S)[0].replace(",", ""))
            F_loss = float(re.findall('F_loss:(.*?) D_loss', line, re.S)[0].replace(",", ""))
            D_loss = float(re.findall('D_loss:(.*?) D_real', line, re.S)[0].replace(",", ""))
            D_real = float(re.findall('D_real:(.*?) D_fake', line, re.S)[0].replace(",", ""))
            D_fake = float(re.findall('D_fake: (.*?) ', line, re.S)[0].replace(",", ""))
            if epoch not in logKey.keys():
                logKey[epoch]={}
                logKey[epoch]["lr"]=lr
                logKey[epoch]["g_loss"]=g_loss
                logKey[epoch]["F_loss"]=F_loss
                logKey[epoch]["D_loss"]=D_loss
                logKey[epoch]["D_real"]=D_real
                logKey[epoch]["D_fake"]=D_fake
        if "Saving the model." in line:
            line=lines[i-1]
            epoch = int(re.findall('iter:(.*?), lr', line, re.S)[0].replace(",",""))
            g_loss = float(re.findall('G_loss:(.*?) F_loss', line, re.S)[0].replace(",", ""))
            F_loss = float(re.findall('F_loss:(.*?) D_loss', line, re.S)[0].replace(",", ""))
            D_loss = float(re.findall('D_loss:(.*?) D_real', line, re.S)[0].replace(",", ""))
            D_real = float(re.findall('D_real:(.*?) D_fake', line, re.S)[0].replace(",", ""))
            D_fake = float(re.findall('D_fake: (.*?) ', line, re.S)[0].replace(",", ""))
            if g_loss<g_loss_min:
                g_loss_min=g_loss
                g_loss_index=epoch
            if F_loss < F_loss_min:
                F_loss_min = F_loss
                F_loss_index=epoch
            if D_loss > D_loss_min:
                D_loss_min = D_loss
                D_loss_index=epoch
            if D_real < D_real_min and 200000!=epoch:
                D_real_min = D_real
                D_real_index=epoch
            if D_fake < D_fake_min:
                D_fake_min = D_fake
                D_fake_index=epoch

    print(logKey)
    print(    g_loss_min, F_loss_min,
    D_loss_min,
    D_real_min,
    D_fake_min)
    print(    g_loss_index,
    F_loss_index,
    D_loss_index,
    D_real_index,
    D_fake_index)
    return logKey

def write_info(logKey,respath="./"):
    train_info = SummaryWriter(log_dir=(os.path.join(respath + '/train_info')))
    for keys,values in logKey.items():
        for key,value in values.items():
            train_info.add_scalar(key, value, keys)
            # train_info.add_scalar('val_pix_err_f', val_pix_err_f, keys)
            # train_info.add_scalar('val_pix_err_nf', val_pix_err_nf, keys)
            # train_info.add_scalar('val_mean_color_err', val_mean_color_err, keys)
    train_info.close()

if __name__ == '__main__':
    path="/media/tcd/data/work/Shanghai_Archives_Bureau/bsrgan/model_zoo/bsrgan_add_realsr/train.log"
    logKey=read_log(path)
    # respath="/media/tcd/data/work/Shanghai_Archives_Bureau/bsrgan/model_zoo/bsrgan_add_realsr"
    # write_info(logKey,respath)



    print("done")