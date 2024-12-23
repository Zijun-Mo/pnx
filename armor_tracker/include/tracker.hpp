#ifndef TRACKER_HPP_
#define TRACKER_HPP_

#include <opencv2/opencv.hpp>
#include <string>
#include <chrono>
#include "armor.hpp"
#include "pnp_solver.hpp"

class Tracker {
public:
    Tracker(const Armor& armor, const double& dt); // 构造函数
    Tracker(); // 默认构造函数
    cv::Mat predict();// 预测下一帧的位置
    void update1(const Armor& armor); // 更新状态
    void update2(const Armor& armor1, const Armor& armor2); // 更新状态
    cv::Point3f getPosition() const; // 获取位置
    cv::Point3f getVelocity() const; // 获取速度
    bool isLost() const; // 判断是否丢失
    void markLost(const int64& frame_id); // 标记丢失
    bool isExpired(const int64& frame_id, const int64& gap_time) const; // 判断是否过期
    void initializeMeasurementMatrix1_1(double theta1, double r);  // 初始化测量矩阵
    void initializeMeasurementMatrix1_2(double theta1, double r);  // 初始化测量矩阵
    void initializeMeasurementMatrix2(double theta1, double theta2, double r1, double r2); // 初始化测量矩阵
    friend bool isSameArmor(const Tracker& tracker, const Armor& armor); // 判断两个装甲板是否是同一个目标
    friend std::vector<Armor> calculateArmorPositions(const Tracker& tracker); // 计算装甲板位置
    std::pair<double, double> getR() const; // 获取半径

private:
    cv::KalmanFilter kf_;
    cv::Mat state_;
    cv::Mat meas1_, meas2_;
    bool issmall, lost_; 
    std::string classification; 
    int64 last_update_time; 
};
double abs_yaw(double x); 
double yawinrange(double x); 


#endif // TRACKER_HPP_