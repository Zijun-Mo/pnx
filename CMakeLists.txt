# 设置CMake最低版本要求
cmake_minimum_required(VERSION 3.13)
# 定义项目名称
project(AUTO_AIM)
# 指定C++版本
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# 启用 -O2 优化选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")

# 测试程序生成的路径
set(EXEC_PATH ${CMAKE_CURRENT_SOURCE_DIR}/bin)
set(CMAKE_BUILD_TYPE "Debug")

#加载OpenCV库
find_package(OpenCV REQUIRED)
if(NOT ${OpenCV_FOUND})
    message(FATAL_ERROR "OpenCV not found")
endif()

#库名称
set(LIBS_OpenCV ${OpenCV_LIBS})

#添加宏定义
add_definitions(-DROOT=\"/home/mozijun/Mycode_c/pnx\")

#可执行文件
set(EXEC_AIM auto_aim)
set(EXECUTABLE_OUTPUT_PATH ${EXEC_PATH})
add_executable(${EXEC_AIM} main.cpp)

#添加子目录
add_subdirectory(armor_detector)
add_subdirectory(armor_tracker)

target_link_libraries(${EXEC_AIM} ${LIBS_OpenCV})