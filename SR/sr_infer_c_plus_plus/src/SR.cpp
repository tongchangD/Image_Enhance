#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <exception>
#include <filesystem>

#include "SR.h"
#include "utils.h"

using namespace std;

SuperResolution::SuperResolution(){}

SuperResolution::~SuperResolution() {}

/*
 * 初始化
 * :param model_file: 模型文件路径
 * :return:
*/
int SuperResolution::init(const std::string model_file){
    try
    {
        // 加载网络模型
        model = torch::jit::load(model_file);
        model.eval();
    }
    catch (std::exception& e)
    {
        std::cout << "\033[31m" << "Standard exception: " << __FILE__ << ": "<<__FUNCTION__<<":"<<__LINE__<<e.what() << "\033[0m" << std::endl;
        return 1;
    }
    return 0;
}

inline bool exists_test (const std::string& name) {
    ifstream f(name.c_str());
    return f.good();
}

/*
 * 判断 文件 是否是图像
 * :param filename: 传入文件路径
 * :return: 文件后缀名
*/
std::string SuperResolution::is_image_file(std::string filepath){
    std::string img_extension="";
    for (size_t i =0; i < IMG_EXTENSIONS.size(); i ++) {
        const string d = IMG_EXTENSIONS[i];
        int index = filepath.find(IMG_EXTENSIONS[i]);
        if (index < filepath.length() && index>0 && exists_test(filepath)){
            return IMG_EXTENSIONS[i];
        }
    }
    return img_extension;

#if 0
    int ret=1;
    for (size_t i =0; i < IMG_EXTENSIONS.size(); i ++) {
        const string d = IMG_EXTENSIONS[i];
        int index = filepath.find(IMG_EXTENSIONS[i]);
        if (index < filepath.length() && index>0){
            return 0;
        }
    }
    return ret;
#endif
}



/*
 *  若传入路径存在文件并是图像文件,便做推理,并将结果保存在传入文件同路径下,输出文件标记以 '_colorize' 作为标记
 * :param img_path: 传入路径
 * :return:
 *
*/
void SuperResolution::infer(std::string filename){
    try{
        std::string img_extension =is_image_file(filename);
        if (img_extension.length()>0){
            std::cout<<"filename: "<<filename<<std::endl;
            torch::Tensor tensor_image = uint2tensor4(filename);
            model.to(at::kCUDA);
            torch::Tensor out_tensor = model.forward({tensor_image}).toTensor();
            tensor2uint(out_tensor,filename.replace(filename.find(img_extension),img_extension.length(),"_SR"+img_extension));
#if 0
        const int channel = 3;
        const int img_h = out_tensor.sizes()[2];
        const int img_w = out_tensor.sizes()[3];
        out_tensor = out_tensor.squeeze().detach().permute({ 1, 2, 0 }).clamp(0,1);
        out_tensor = out_tensor.mul(255).to(torch::kU8); //s3:*255，转uint8
        out_tensor = out_tensor.to(torch::kCPU);
        cv::Mat outimg(img_h, img_w, CV_8UC3); // 定义输出图像空间
        std::vector<cv::Mat> out_stack;
        // 输出图像为rgb,3为通道数
        for (int c = 0; c < 3; c++)
        {
            out_stack.emplace_back(cv::Mat(img_h, img_w, CV_8UC1, out_tensor.data_ptr() + c * img_h*img_w));
        }
        // 将加载到的数组上模型输出值合并到输出图像
        cv::merge(out_stack, outimg);
//        cv::Mat tmp_img(img_h, img_w, CV_8UC1);
//        std::memcpy((void *) tmp_img.data, out_tensor.data_ptr(), out_tensor.numel());
//        cv::Mat outimg(img_h, img_w, CV_8UC3, tmp_img.data + 1 * img_h*img_w);

//        cv::Mat imgbin(img_h, img_w, CV_8UC1, out_tensor.data_ptr());
//        cv::cvtColor(imgbin, imgbin, CV_GRAY2BGR);
        cv::imwrite("outimg.jpg", outimg);
#endif

#if 0
        const int img_h2 = img_h/3;
        const int img_w2 = img_w/3;
//        const int img_length2 = img_h2*img_w2;
        std::vector<cv::Mat> mat_vec;
        cv::Mat split_img(img_h, img_w + 2, CV_8UC3);
        for(int c = 0;c < channel; c++)
        {
            for(int y = 0; y < img_h2; ++y)
            {
                for(int x = 0; x < img_w2; ++x)
                {
                    split_img.at<cv::Vec3b>(y + c*img_h2, x + c*img_w)[i] = tmp_img.at<cv::Vec3b>(y, x);
                }
            }
//            mat_vec.push_back(split_img);
        }
//        cv::merge(mat_vec, tmp_dst);
        cv::imwrite("split_img.jpg", split_img);
#endif

#if 0

        cv::Mat outimg(img_h, img_w, CV_8UC3);
        std::vector<cv::Mat> out_stack;
        // 输出图像为rgb,3为通道数
        for (int c = 0; c < 3; c++)
        {
            out_stack.emplace_back(cv::Mat(img_h, img_w, CV_8UC3, tmp_img.data + c * img_lenth));
        }
        // 将加载到的数组上模型输出值合并到输出图像
        cv::merge(out_stack, outimg);
        cv::imwrite("outimg.jpg", outimg);
#endif
        }
        else{
            printf("please check out image path\n");
        }
    }
    catch(exception RuntimeError) {
            std::cout<<"out of memory"<<std::endl;
        }
}

int SuperResolution::destory(){
    try
    { }
    catch (...)
    {
        return 1;
    }
    return 0;
}
