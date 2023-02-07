#include <algorithm>
#include <ctime>
#include "iostream"
#include "fstream"
#include <sys/time.h>
#include <stdio.h>
#include <string.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <SR.h>
using namespace cv;
using namespace std;



double getCurrentTime()
{
   struct timeval tv;
   gettimeofday(&tv,NULL);
   return tv.tv_sec*1000 + tv.tv_usec/1000.0;
}


int main(int argc, char** argv)
{
    double total_correct_time = 0.0;
    double corret_start_time = getCurrentTime();
    std::string model_path = "/media/tcd/data/C++/SR/model/model.pt";
    SuperResolution superresolution;
    int ret = superresolution.init(model_path);
    if(ret != 0)
    {
        std::cout << "model initial failure!" << std::endl;
        return 0;
    }
    std::cout << "model initial success!" << std::endl;
    string test="/home/tcd/Desktop/IMG_0051_d65b5a3a.png";

    superresolution.infer(test);

    // TODO:
    // email:tongchangdong@beyondbit.com,
    // issue: 空壳,需要填充内容,尚未找到合适的参考代码
    // ODOT
    superresolution.destory();

    double handle_correct_time = getCurrentTime() - corret_start_time;
    total_correct_time +=handle_correct_time;
    std::cout << "total_correct_time: " << total_correct_time << std::endl;
    std::cout<<"done"<<std::endl;
    return 0;
}
