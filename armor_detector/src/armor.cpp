#include "armor.hpp"
#include "pnp_solver.hpp"

// 计算装甲板背后的点
cv::Point3f Armor::calculatePointBehindArmor(double r) const {
    // 提取旋转矩阵和平移向量
    cv::Mat rotationMatrix = ex_mat(cv::Rect(0, 0, 3, 3));
    cv::Mat tvec = ex_mat(cv::Rect(3, 0, 1, 3));

    // 将旋转矩阵转换为旋转向量
    cv::Mat rvec;
    cv::Rodrigues(rotationMatrix, rvec);

    // 计算装甲板中心背后 r 处的点的位置
    cv::Mat pointBehind = (cv::Mat_<double>(3, 1) << 0, 0, r);
    cv::Mat rotatedPointBehind;
    cv::Mat rotatedPointBehindHomogeneous = cv::Mat::ones(4, 1, CV_64F);
    rotatedPointBehindHomogeneous(cv::Rect(0, 0, 1, 3)) = rotationMatrix * pointBehind + tvec;

    // 转换为 cv::Point3f
    cv::Point3f result(rotatedPointBehindHomogeneous.at<double>(2), rotatedPointBehindHomogeneous.at<double>(0), -rotatedPointBehindHomogeneous.at<double>(1));
    return result;
}

// 计算装甲板绕 y 轴的旋转角
double Armor::calculateYawAngle() const {
    // 提取旋转矩阵
    cv::Mat rotationMatrix = ex_mat(cv::Rect(0, 0, 3, 3));

    // 提取绕 y 轴的旋转角（yaw）
    double yaw = atan2(rotationMatrix.at<double>(2, 0), rotationMatrix.at<double>(0, 0));

    return yaw;
}
void Armor::calculatemergedRect() {
    PnPSolver pnp_solver;
    pnp_solver.readCameraParameters(std::string(ROOT) + "/input/2BDFA1701242.yaml"); 
    std::vector<cv::Point3f> l_points = {
        cv::Point3f(-0.135, -0.0635, 0.0), 
        cv::Point3f(0.135, -0.0635, 0.0), 
        cv::Point3f(0.135, 0.0635, 0.0), 
        cv::Point3f(-0.135, 0.0635, 0.0) 
    },  
    s_points = {
        cv::Point3f(-0.0635, -0.0625, 0.0), 
        cv::Point3f(0.0635, -0.0625, 0.0), 
        cv::Point3f(0.0635, 0.0625, 0.0), 
        cv::Point3f(-0.0635, 0.0625, 0.0) 
    }, objectPoints; // 世界坐标系中的四个点
    
    if (is_small) {
        objectPoints = s_points; 
    } else {
        objectPoints = l_points;
    }

    mergedRect = worldToImage(objectPoints, ex_mat, pnp_solver);
}