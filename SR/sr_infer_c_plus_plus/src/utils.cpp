#include <torch/torch.h>
#include <utils.h>

/*
 * 根据路径,读取图像
 * :param path: 图像路径
 * :param n_channels: 图像通道
 * :return:
*/
cv::Mat imread_uint(std::string path, int n_channels=3){
    cv::Mat img_new;
    cv::Mat img;
    if (n_channels == 1){
         img=cv::imread(path, 0);
//        img = np.expand_dims(img, axis=2);
    }
    else if(n_channels == 3)
    {
        cv::Mat img = imread(path, cv::IMREAD_UNCHANGED);  // cv2.IMREAD_UNCHANGED
        if (img.rows == 2)
        {
            cv::cvtColor(img, img_new, cv::COLOR_BGR2GRAY);  // cv2.COLOR_GRAY2RGB
        }
        else
        {
            cv::cvtColor(img, img_new,  cv::COLOR_BGR2RGB);  // cv2.COLOR_BGR2RGB
        }
    }
   return img_new;
}


/*
 * 将 uint 转换为 4 维 torch tensor
 * :param img: 传入数据
 * :return:
*/
torch::Tensor uint2tensor4(std::string image_path){
    //输入图像
    cv::Mat image_transfomed = cv::imread(image_path,cv::ImreadModes::IMREAD_COLOR);
    if (image_transfomed.empty())
    {
        printf("please check out image path\n");
    }
//    cv::cvtColor(image_transfomed, image_transfomed, cv::COLOR_BGR2RGB);
//    image_transfomed.convertTo(image_transfomed, CV_32FC3);
    // 图像转换为Tensor
    torch::Tensor tensor_image = torch::from_blob(image_transfomed.data, {image_transfomed.rows, image_transfomed.cols,3},torch::kByte);
    tensor_image = tensor_image.permute({2,0,1});
    tensor_image = tensor_image.toType(torch::kFloat);
    tensor_image = tensor_image.div(255);
    tensor_image = tensor_image.unsqueeze(0);
    tensor_image = tensor_image.to(at::kCUDA);
    return tensor_image;
}


/*
 * 将  4 维 torch tensor 转为 uint 的数据
 * :param img:
 * :return:
*/
void tensor2uint(torch::Tensor out_tensor,std::string filename){
#if 0
    std::cout<<"out_tensor: "<<out_tensor.sizes()<<std::endl;

    int h = out_tensor.sizes()[2];
    int w = out_tensor.sizes()[3];
    std::cout<<"h: "<<h<<" w: "<<w<<std::endl;

    out_tensor = out_tensor.to(torch::kCPU).permute({0, 3, 1, 2}).squeeze().detach().permute({0,1,2});    //ten_wrp为tensor数据，squeeze()只用于batchsize为1的场景，permute 是将存储格式从pytorch形式转成opencv格式
    out_tensor = out_tensor.mul(255).clamp(0, 255).to(torch::kU8); //在处理前对cvmat中的值做了归一化，所以现在要*255恢复，同时对于不在0-255范围内的数据，需要做限制；cvmat的数据格式是8UC3，所以torch tensor要提前转换成kU8
    std::cout<<"out_tensor: "<<out_tensor.sizes()<<std::endl;
    cv::Mat resultImg(h, w, CV_8UC3); // 定义一个Mat数据接收数据

    std::cout<<"out_tensor.data_ptr(): "<<out_tensor.data_ptr()<<" sizeof(torch::kU8): "<<sizeof(torch::kU8)<<" out_tensor.numel(): "<<out_tensor.numel()<<std::endl;

    std::memcpy((void *) resultImg.data, out_tensor.data_ptr(), sizeof(torch::kU8) * out_tensor.numel());    //将ten_wrp数据存为resultImg格式 ??? 是不是此处拷贝的问题
    cv::cvtColor(resultImg, resultImg, CV_RGB2BGR); // RGB 转 BGR
    std::cout<<"resultImg: "<<resultImg.size<<std::endl;
//    cv::imshow("test",resultImg);
//    cv::waitKey(0);// 按任意键在0秒后退出窗口，不写这句话是不会显示出窗口的
    cv::imwrite(filename,resultImg);    //保存图片


#endif
#if 1
    const int channel = 3;
    const int img_h = out_tensor.sizes()[2];
    const int img_w = out_tensor.sizes()[3];
    out_tensor = out_tensor.squeeze().detach().permute({ 1, 2, 0 }).clamp(0,1);
    out_tensor = out_tensor.mul(255).to(torch::kU8); //s3:*255，转uint8
    out_tensor = out_tensor.to(torch::kCPU);

    cv::Mat resultImg(img_h, img_w, CV_8UC3);
    std::vector<cv::Mat> out_stack;
    // 输出图像为rgb,3为通道数
    for (int c = 0; c < 3; c++)
    {
        out_stack.emplace_back(cv::Mat(img_h, img_w, CV_8UC1, out_tensor.data_ptr() + c * img_h*img_w));
    }
    // 将加载到的数组上模型输出值合并到输出图像
    cv::merge(out_stack, resultImg);
    cv::imwrite(filename,resultImg);    //保存图片
#endif
}


