#include "number_classifier.hpp"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <opencv2/opencv.hpp>

// 构造函数，初始化模型路径、标签路径和阈值
NumberClassifier::NumberClassifier(const std::string &model_path, const std::string &label_path, double threshold)
    : threshold_(threshold) {
    loadModel(model_path);
    loadLabels(label_path);
}

// 加载模型
void NumberClassifier::loadModel(const std::string &model_path) {
    net_ = cv::dnn::readNetFromONNX(std::string(ROOT) + "/armor_detector/model/" + model_path);
    if (net_.empty()) {
        throw std::runtime_error("Failed to load ONNX model from " + model_path);
    }
}

// 加载标签
void NumberClassifier::loadLabels(const std::string &label_path) {
    std::ifstream label_file(std::string(ROOT) + "/armor_detector/model/" + label_path);
    if (!label_file.is_open()) {
        throw std::runtime_error("Failed to open label file: " + label_path);
    }
    std::string line;
    while (std::getline(label_file, line)) {
        class_names_.push_back(line);
    }
}

// 预处理图像
cv::Mat NumberClassifier::preprocess(const cv::Mat &image) {
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_RGB2GRAY);  // 将图像转换为灰度图像

    // 使用大津法进行二值化处理
    cv::Mat binary;
    cv::threshold(gray, binary, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    // 将图像转换为浮点类型，并归一化到 [0, 1] 范围
    cv::Mat float_image;
    binary.convertTo(float_image, CV_32F, 1.0 / 255.0);
    // imshow("binary", float_image); 
    // cv::waitKey(0);

    // 创建一个 Blob，用于深度学习模型的输入
    cv::Mat blob = cv::dnn::blobFromImage(float_image);
    return blob;
}

// 分类数字
std::pair<std::string, double> NumberClassifier::classifyNumber(const cv::Mat &image, bool isSmall) {
    cv::Mat blob = preprocess(image);
    net_.setInput(blob);
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
    // std::cout << "label_id:" << label_id <<" " << "confidence: " << confidence << std::endl; 

    // 如果置信度达不到阈值，则返回 "negative" 和置信度
    if (confidence < threshold_) {
        return std::make_pair("negative", confidence);
    }
    if(isSmall){
        if(label_id == 1 ||label_id == 5 ||label_id == 6) return std::make_pair("negative", -confidence);
    }
    else{
        if(label_id == 0 ||label_id == 7) return std::make_pair("negative", -confidence); 
    }

    // 返回标签字符串和置信度
    return std::make_pair(class_names_[label_id], confidence);
}