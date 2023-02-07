#ifndef UTILS_H
#define UTILS_H
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <string.h>
#include <torch/torch.h>

using namespace cv;
using namespace std;

/**
 * 根据路径,读取图像
 * :param path: 图像路径
 * :param n_channels: 图像通道
 * :return:
*/
cv::Mat imread_uint(std::string path, int n_channels);



/**
 * 将传入的图像保存到传入的图像路径下
 * :param img: 传入图像
 * :param img_path: 传入图像路径
 * :return:
*/
cv::Mat imsave(cv::Mat img, std::string img_path);

/**
 * 将 uint 转换为 4 维 torch tensor
 * :param img: 传入数据
 * :return:
*/
torch::Tensor uint2tensor4(std::string image_path);

/*
 * 将  4 维 torch tensor 转为 uint 的数据
 * :param img:
 * :return:
*/
void tensor2uint(torch::Tensor,std::string filename);




#endif // UTILS_H
