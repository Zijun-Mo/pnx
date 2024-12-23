// Copyright 2022 Chen Jun

#ifndef ARMOR_DETECTOR__NUMBER_CLASSIFIER_HPP_
#define ARMOR_DETECTOR__NUMBER_CLASSIFIER_HPP_

// OpenCV
#include <opencv2/opencv.hpp>

// STL
#include <cstddef>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "armor_detector/armor.hpp"

namespace rm_auto_aim
{
  // NumberClassifier 类用于分类装甲板上的数字
  class NumberClassifier
  {
  public:
    // 构造函数，初始化模型路径、标签路径、阈值和忽略的类别
    NumberClassifier(
        const std::string &model_path, const std::string &label_path, const double threshold,
        const std::vector<std::string> &ignore_classes = {});

    // 提取图像中的数字
    void extractNumbers(const cv::Mat &src, std::vector<Armor> &armors);

    // 对提取的数字进行分类
    void classify(std::vector<Armor> &armors);

    // 分类阈值
    double threshold;

  private:
    // OpenCV DNN 网络
    cv::dnn::Net net_;
    // 类别名称
    std::vector<std::string> class_names_;
    // 忽略的类别
    std::vector<std::string> ignore_classes_;
  };
} // namespace rm_auto_aim

#endif // ARMOR_DETECTOR__NUMBER_CLASSIFIER_HPP_