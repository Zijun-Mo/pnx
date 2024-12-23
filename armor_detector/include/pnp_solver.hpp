#ifndef PNP_SOLVER_HPP_
#define PNP_SOLVER_HPP_

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

class PnPSolver {
public:
    PnPSolver();
    friend std::vector<cv::Point2f> worldToImage(const std::vector<cv::Point3f>& objectPoints, const cv::Mat& ex_mat, const PnPSolver& pnp);// 世界坐标系转换到图像坐标系
    cv::Mat solvePnPWithIPPE(const std::vector<cv::Point2f>& imagePoints, const std::string& filename, const bool issmall); // PnP解算器函数
    bool readCameraParameters(const std::string& filename); // 读取相机参数

private:
    std::vector<cv::Point3f> l_points, s_points; 
    std::vector<cv::Point3f> objectPoints;  
    cv::Mat rvec, tvec, rotationMatrix, transformMatrix; 
    cv::Mat cameraMatrix, distCoeffs; 
    bool success; 
};

#endif  // PNP_SOLVER_HPP_