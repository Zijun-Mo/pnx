#ifndef NUMBER_CLASSIFIER_HPP_
#define NUMBER_CLASSIFIER_HPP_

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <string>
#include <vector>

class NumberClassifier {
public:
    // 构造函数，初始化模型路径、标签路径和阈值
    NumberClassifier(const std::string &model_path, const std::string &label_path, double threshold);

    // 从图像中分类数字
    std::pair<std::string, double> classifyNumber(const cv::Mat &image, bool isSmall); 

private:
    // 加载模型和标签
    void loadModel(const std::string &model_path);
    void loadLabels(const std::string &label_path);

    // 预处理图像
    cv::Mat preprocess(const cv::Mat &image);

    // 模型和标签
    cv::dnn::Net net_;
    std::vector<std::string> class_names_;
    double threshold_;
};

#endif  // NUMBER_CLASSIFIER_HPP_