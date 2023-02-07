cmake_minimum_required(VERSION 2.8)
PROJECT(SR_infer)

# 添加 选项 并设置默认值
# option(USE_CAFFE "default is USE_CAFFE,when set ON detection mode change to USE_CAFFE" ON)


set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)


# 平台识别，以适应不同的库环境，获取不同的唯一识别码
IF(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "armv7l")
    MESSAGE(STATUS "Now is TK1 UNIX-like OS's.")
    set(CMAKE_PREFIX_PATH /usr/local/lib/python3.6/dist-packages/torch)
    add_definitions(-DTK1_DEVICE)
ELSEIF(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "aarch64")
    MESSAGE(STATUS "Now is jetson TX2 or Nano UNIX-like OS's.")
    #Set pytorch path
    set(CMAKE_PREFIX_PATH /usr/local/lib/python3.6/dist-packages/torch)
    add_definitions(-DARM64_DEVICE)
ELSE()
    MESSAGE(STATUS "Now is Desktop UNIX-like OS's.")
#    set(CMAKE_PREFIX_PATH /media/tcd/software/anaconda3/lib/python3.7/site-packages/torch)
#    set(CMAKE_PREFIX_PATH /media/tcd/software/anaconda3/envs/py36_torch11/lib/python3.6/site-packages/torch)
     set(CMAKE_PREFIX_PATH /media/tcd/software/anaconda3/envs/cv_torch17/lib/python3.7/site-packages/torch)
ENDIF()
find_package(Torch REQUIRED NO_CMAKE_FIND_ROOT_PATH)



# 设置Opencv的CMake路径
set(OpenCV_DIR /media/tcd/software/opencv-3.4.14/build)
find_package (OpenCV REQUIRED NO_CMAKE_FIND_ROOT_PATH)  # 查找 指定的库文件
include_directories(${OpenCV_INCLUDE_DIRS})
find_library(OpenCV_LIBRARY HINTS ${OpenCV_ROOT} ${OpenCV_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib lib64 lib/x64)
link_libraries(opencv_calib3d;opencv_core;
        opencv_dnn;opencv_features2d;
        opencv_flann;opencv_highgui;opencv_imgcodecs;opencv_imgproc;opencv_ml;opencv_objdetect;
        opencv_photo;opencv_shape;opencv_stitching;opencv_superres;opencv_video;opencv_videoio;opencv_videostab)


if(OpenCV_FOUND)
    INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS}) # 设置包含的目录
    message(STATUS "OpenCV library status:")
    message(STATUS "    version: ${OpenCV_VERSION}")
    message(STATUS "    libraries: ${OpenCV_LIBS}")
    message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
endif()








#set(CMAKE_PREFIX_PATH /media/tcd/software/Qt5.14.1/5.14.1/gcc_64/lib/cmake/Qt5)
#find_package(Qt5 REQUIRED Core)
#include_directories(${Qt5_INCLUDE_DIRS})
#set(CMAKE_PREFIX_PATH /home/wqs/work/sdk/boost_1_71_0/install_pic/lib/cmake/Boost-1.71.0)
#find_package(Boost REQUIRED NO_CMAKE_FIND_ROOT_PATH)
#include_directories(${Boost_INCLUDE_DIRS})

#aux_source_directory(. DIR_SRCS)
#aux_source_directory(./private_inc DIR_SRCS)
#aux_source_directory(./src DIR_SRCS)
#message("DIR_SRCS = ${DIR_SRCS}")


#add_subdirectory(common) # 该命令告诉CMake去子目录中查看可用的CMakeLists.txt文件

file(GLOB_RECURSE CURRENT_HEADERS export_inc/*.h  export_inc/*.hpp private_inc/*.h  private_inc/*.hpp)
file(GLOB CURRENT_SOURCES src/*.cpp src/*.c)
include_directories(
    export_inc/
    private_inc/
    )

INCLUDE_DIRECTORIES(${TORCH_INCLUDE_DIRS})


set(CMAKE_BUILD_TYPE DEBUG)
#设置编译器版本
###############################################
SET(CMAKE_C_COMPILER g++ pkg-config  --cflags --libs opencv)
#if(CMAKE_COMPILER_IS_GNUCXX)
#    add_compile_options(-std=c++11 -fno-stack-protector) # very important key in TK1,otherwise will raise an error call stack smashing detected
#    message(STATUS "optional:-std=c++11")
#endif(CMAKE_COMPILER_IS_GNUCXX)
###############################################
#CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
#CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
#if(COMPILER_SUPPORTS_CXX11)
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
#elseif(COMPILER_SUPPORTS_CXX0X)
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
#else()
#    message(WARNING "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support.")
#    message(STATUS "Please use a different C++ compiler.")
#endif()
###############################################



# 优化组测试用
# add_definitions(-D dialogue_debug)


SET(PROC_ALL_FILES ${CURRENT_HEADERS} ${CURRENT_SOURCES} ${COMMON_SOURCES})
set(TEST_APP Test_app)

#source directory 下面两句是对应的
#AUX_SOURCE_DIRECTORY(test TEST_SRCS)
#add_executable(${TEST_APP} ${TEST_SRCS})

#add executable file，添加要编译的可执行文件 告诉工程生成一个可执行文件
add_library(${PROJECT_NAME} SHARED ${PROC_ALL_FILES}) #编译成动态库
add_executable(${TEST_APP} test/main.cpp)
#add_executable(${PROJECT_NAME} app/src/a.cpp)

#添加可执行文件所需要的库，比如我们用到了libm.so（命名规则：lib+name+.so），就添加该库的名称
# TARGET_LINK_LIBRARIES(${PROJECT_NAME} Qt5::Core)
TARGET_LINK_LIBRARIES(${PROJECT_NAME} -ldl ${TORCH_LIBRARIES} )

# TARGET_LINK_LIBRARIES(${TEST_APP} Qt5::Core)
# TARGET_LINK_LIBRARIES(${TEST_APP} pthread)
#TARGET_LINK_LIBRARIES(${TEST_APP} -ldl ${TORCH_LIBRARIES})
TARGET_LINK_LIBRARIES(${TEST_APP} ${PROJECT_NAME} )

#set_property(TARGET ${TEST_APP} PROPERTY CXX_STANDARD 11)
