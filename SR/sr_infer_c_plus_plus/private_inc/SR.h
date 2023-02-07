#ifndef DETECT_H
#define DETECT_H
#include "torch/script.h"
#include <torch/torch.h>
#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <memory>


class SuperResolution{
public:
    SuperResolution();
    ~SuperResolution();

    /**
     * 初始化
     * :param model_file: 模型文件路径
     * :return:
    */
    int init(const std::string model_file);

    /**
     * 判断 文件 是否是图像
     * :param filename: 传入文件路径
     * :return:
    */
    std::string is_image_file(std::string filepath);

    /**
     *  若传入路径存在文件并是图像文件,便做推理,并将结果保存在传入文件同路径下,输出文件标记以 '_SR' 作为标记
     * :param img_path: 传入路径
     * :return:
     *
    */
    void infer(std::string filename);

    /**
     * 内存释放
     * @return
     */
    int destory();


private:
    torch::DeviceType device_type = torch::kCUDA;
    torch::jit::script::Module  model;
    std::vector<std::string> IMG_EXTENSIONS={".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG", ".ppm", ".PPM", ".bmp", ".BMP", ".tif"};
};

#endif // DETECT_H


