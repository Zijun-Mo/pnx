#ifndef ARMOR_HPP
#define ARMOR_HPP

#include <opencv2/opencv.hpp>
#include <string>
#include <cmath>
#include "pnp_solver.hpp"

struct Armor {
    bool is_small; // 是否为小装甲板
    std::string classification; //装甲板类型
    cv::Mat ex_mat; // 装甲板位置(4*4矩阵表示，包括位置和旋转矩阵)
    double probability; // 置信度
    int64 frame_id; // 帧编号
    std::vector<cv::Point2f> mergedRect; 

    // 将函数声明为成员函数
    cv::Point3f calculatePointBehindArmor(double r) const; // 计算装甲板背后的点
    double calculateYawAngle() const; // 计算装甲板绕 y 轴的旋转角
    void calculatemergedRect(); // 计算矩形
};

#endif // ARMOR_HPP