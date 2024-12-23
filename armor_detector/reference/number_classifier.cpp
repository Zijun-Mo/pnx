// Copyright 2022 Chen Jun
// Licensed under the MIT License.

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

// STL
#include <algorithm>
#include <cstddef>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "armor_detector/armor.hpp"
#include "armor_detector/number_classifier.hpp"

namespace rm_auto_aim
{
// NumberClassifier 构造函数
NumberClassifier::NumberClassifier(
  const std::string & model_path, const std::string & label_path, const double thre,
  const std::vector<std::string> & ignore_classes)
: threshold(thre), ignore_classes_(ignore_classes)  // 初始化成员变量
{
  // 从 ONNX 文件加载模型
  net_ = cv::dnn::readNetFromONNX(model_path);

  // 打开标签文件
  std::ifstream label_file(label_path);
  std::string line;
  // 逐行读取标签文件，并将每一行添加到 class_names_ 向量中
  while (std::getline(label_file, line)) {
    class_names_.push_back(line);
  }
}
}  // namespace rm_auto_aim

void NumberClassifier::extractNumbers(const cv::Mat & src, std::vector<Armor> & armors)
{
  // Light length in image
  const int light_length = 12;  // 灯条在图像中的长度
  // Image size after warp
  const int warp_height = 28;  // 透视变换后的图像高度
  const int small_armor_width = 32;  // 小装甲板的宽度
  const int large_armor_width = 54;  // 大装甲板的宽度
  // Number ROI size
  const cv::Size roi_size(20, 28);  // 数字区域的大小

  for (auto & armor : armors) {
    // Warp perspective transform
    cv::Point2f lights_vertices[4] = {
      armor.left_light.bottom, armor.left_light.top, armor.right_light.top,
      armor.right_light.bottom};  // 获得原图像的四个角点

    const int top_light_y = (warp_height - light_length) / 2 - 1;  // 计算顶部灯条的 y 坐标
    const int bottom_light_y = top_light_y + light_length;  // 计算底部灯条的 y 坐标
    const int warp_width = armor.type == ArmorType::SMALL ? small_armor_width : large_armor_width;  // 根据装甲板类型确定透视变换后的宽度
    cv::Point2f target_vertices[4] = {
      cv::Point(0, bottom_light_y),
      cv::Point(0, top_light_y),
      cv::Point(warp_width - 1, top_light_y),
      cv::Point(warp_width - 1, bottom_light_y),
    };  // 获得目标变换图像的四个角点

    cv::Mat number_image;  // 存储透视变换后的图像
    auto rotation_matrix = cv::getPerspectiveTransform(lights_vertices, target_vertices);  // 计算透视变换矩阵
    cv::warpPerspective(src, number_image, rotation_matrix, cv::Size(warp_width, warp_height));  // 实行透视变换

    // Get ROI
    number_image = number_image(cv::Rect(cv::Point((warp_width - roi_size.width) / 2, 0), roi_size));  // 获取数字区域的 ROI

    // Binarize
    cv::cvtColor(number_image, number_image, cv::COLOR_RGB2GRAY);  // 转换为灰度图像
    cv::threshold(number_image, number_image, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);  // 二值化处理

    armor.number_img = number_image;  // 将处理后的图像存储到装甲板对象中
  }
}

void NumberClassifier::classify(std::vector<Armor> & armors)
// 对装甲板进行数字分类，更新 armor 对象的 confidence 和 number
{
  for (auto & armor : armors) {
    // 克隆装甲板的数字图像
    cv::Mat image = armor.number_img.clone();

    // 归一化图像，将像素值缩放到 [0, 1] 范围
    image = image / 255.0;

    // 从归一化图像创建一个 Blob，这是深度学习中常用的预处理步骤，以便图像可以被神经网络正确处理
    cv::Mat blob;
    cv::dnn::blobFromImage(image, blob);

    // 将 Blob 设置为神经网络的输入
    net_.setInput(blob);

    // 前向传播，获取模型的输出
    cv::Mat outputs = net_.forward();

    // 计算 softmax 概率
    float max_prob = *std::max_element(outputs.begin<float>(), outputs.end<float>());
    cv::Mat softmax_prob;
    cv::exp(outputs - max_prob, softmax_prob);
    float sum = static_cast<float>(cv::sum(softmax_prob)[0]);
    softmax_prob /= sum;

    // 获取最大概率和对应的类别 ID
    double confidence;
    cv::Point class_id_point;
    minMaxLoc(softmax_prob.reshape(1, 1), nullptr, &confidence, nullptr, &class_id_point);
    int label_id = class_id_point.x;

    // 更新装甲板的置信度和数字
    armor.confidence = confidence;
    armor.number = class_names_[label_id];

    // 格式化分类结果字符串
    std::stringstream result_ss;
    result_ss << armor.number << ": " << std::fixed << std::setprecision(1)
              << armor.confidence * 100.0 << "%";
    armor.classfication_result = result_ss.str();
  }

  // 移除低置信度和忽略类别的装甲板
  armors.erase(
    std::remove_if(
      armors.begin(), armors.end(),
      [this](const Armor & armor) {
        // 如果置信度低于阈值，则移除
        if (armor.confidence < threshold) {
          return true;
        }

        // 如果装甲板的数字在忽略类别列表中，则移除
        for (const auto & ignore_class : ignore_classes_) {
          if (armor.number == ignore_class) {
            return true;
          }
        }

        // 根据装甲板类型和数字进行匹配，移除不匹配的装甲板
        bool mismatch_armor_type = false;
        if (armor.type == ArmorType::LARGE) {
          mismatch_armor_type =
            armor.number == "outpost" || armor.number == "2" || armor.number == "guard";
        } else if (armor.type == ArmorType::SMALL) {
          mismatch_armor_type = armor.number == "1" || armor.number == "base";
        }
        return mismatch_armor_type;
      }),
    armors.end());
}

}  // namespace rm_auto_aim
