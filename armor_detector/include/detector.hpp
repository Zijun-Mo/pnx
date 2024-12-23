#ifndef DETECTOR_HPP_
#define DETECTOR_HPP_

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

class Detector {
public:
    Detector(); 
    cv::Mat convertToAdaptiveBinary(const cv::Mat& img, const cv::Ptr<cv::CLAHE> clahe, const int& clipLimit); // 灰度二值化
    std::vector<cv::RotatedRect> processContours(); // 处理轮廓
    bool isSimilarRotatedRect(const cv::RotatedRect& rect1, const cv::RotatedRect& rect2, bool &issmall); // 判断两个旋转矩形是否相似
    std::vector<cv::Point2f> mergeSimilarRects(const cv::RotatedRect& rect1, const cv::RotatedRect& rect2); // 合并相似的旋转矩形
    cv::Mat warpToRectangle(const cv::Mat& img, const std::vector<cv::Point2f>& quad, int width, int height); // 透视变换
    
 private:
    bool isRedDominant(const cv::RotatedRect& minRect);  // 判断矩形区域内的像素颜色 
    bool isLight(cv::RotatedRect& rect, const std::vector<cv::Point>& contour); // 判断矩形是否为红色
    cv::Mat performPCA(const cv::Mat& roiImage); // 主成分分析
    std::pair<cv::Point2f, cv::Point2f> findExtremePoints(const cv::Mat& roiImage, 
                                                          const cv::Point2f& symmetryAxis, 
                                                          const cv::Size2f& rectSize, 
                                                          const cv::Point2f& rectCenter, 
                                                          double mean_val); // 找到灯条角点
    cv::Mat grayImg, equalizedImg, binaryImg, originalImg; 
};
#endif