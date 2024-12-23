#include "tracker.hpp"
#include "armor.hpp"
#include <vector>

Tracker::Tracker(const Armor& armor, const double& dt)
    : kf_(12, 10, 0, CV_64F), state_(12, 1, CV_64F), meas1_(4, 1, CV_64F), meas2_(10, 1, CV_64F) {
    // kf_是卡尔曼滤波器对象, state_是状态向量, meas_是测量向量
    // 初始化状态向量 x
    cv::Point3f chassis_position = armor.calculatePointBehindArmor(0.2);

    state_.at<double>(0) = chassis_position.x; // x
    state_.at<double>(1) = chassis_position.y; // y
    state_.at<double>(2) = chassis_position.z; // z1
    state_.at<double>(3) = chassis_position.z; // z2
    state_.at<double>(4) = 0; // v_x
    state_.at<double>(5) = 0; // v_y
    state_.at<double>(6) = 0; // v_z
    state_.at<double>(7) = 0; // w
    state_.at<double>(8) = 0.2; // r1
    state_.at<double>(9) = 0.2; // r2
    state_.at<double>(10) = armor.calculateYawAngle(); // yaw1
    state_.at<double>(11) = state_.at<double>(10) + CV_PI / 2 > CV_PI ? state_.at<double>(10) - CV_PI * 1.5 : state_.at<double>(10) + CV_PI / 2; // yaw2

    // 初始化过程噪声协方差矩阵 Q 
    // 初始化加速度转移矩阵 
    cv::Mat acceleration = (cv::Mat_<double>(12, 4) <<
     0.5 * dt * dt, 0, 0, 0, 
     0, 0.5 * dt * dt, 0, 0, 
     0, 0, 0.5 * dt * dt, 0, 
     0, 0, 0.5 * dt * dt, 0, 
     dt, 0, 0, 0, 
     0, dt, 0, 0, 
     0 ,0, dt, 0, 
     0, 0, 0, dt, 
     0, 0, 0, dt, 
     0, 0, 0, dt, 
     0, 0, 0, 0.5 * dt * dt,
     0, 0, 0, 0.5 * dt * dt);

    kf_.processNoiseCov = acceleration * cv::Mat::eye(4, 4, CV_64F) * acceleration.t() * 1e1;

    // 初始化测量噪声协方差矩阵 R
    kf_.measurementNoiseCov = cv::Mat::eye(10, 10, CV_64F) * 1e-5;

    // 初始化后验错误估计协方差矩阵 P
    kf_.errorCovPost = cv::Mat::eye(12, 12, CV_64F);

    // 初始化状态
    kf_.statePost = state_;

    // 初始化状态转移矩阵
    kf_.transitionMatrix = cv::Mat::eye(12, 12, CV_64F); 
    kf_.transitionMatrix.at<double>(0, 4) = dt;
    kf_.transitionMatrix.at<double>(1, 5) = dt;
    kf_.transitionMatrix.at<double>(2, 6) = dt;
    kf_.transitionMatrix.at<double>(10, 7) = dt;
    kf_.transitionMatrix.at<double>(11, 7) = dt;

    // 初始化分类和是否是小装甲板
    classification = armor.classification;
    issmall = armor.is_small;

    // 初始化更新时间
    last_update_time = armor.frame_id;
}


Tracker::Tracker(){}
cv::Mat Tracker::predict() {
    cv::Mat prediction = kf_.predict();

    double yaw1 = prediction.at<double>(10);
    double yaw2 = prediction.at<double>(11); 
    double temp; 
    if(yaw2 > yaw1){
        temp = (yaw1 + yaw2) / 2; 
        yaw1 = yawinrange(temp - CV_PI / 4);
        yaw2 = yawinrange(temp + CV_PI / 4);
    }
    else{
        temp = (yaw1 + yaw2) / 2; 
        yaw1 = yawinrange(temp + CV_PI * 0.75); 
        yaw2 = yawinrange(temp - CV_PI * 0.75); 
    }
    kf_.statePost.at<double>(10) = yaw1;
    kf_.statePost.at<double>(11) = yaw2;

    return prediction;
}

void Tracker::update1(const Armor& armor) {
    cv::Point3f pointBehindArmor = armor.calculatePointBehindArmor(state_.at<double>(8));
    double yawAngle = armor.calculateYawAngle();
    int temp_i = 0; 
    while(abs_yaw(yawAngle - state_.at<double>(10)) >= CV_PI / 4) yawAngle = yawinrange(yawAngle + CV_PI / 2), temp_i++; 
    // 创建一个 4x1 的 cv::Mat，包含点的坐标和 yaw 角度
    meas1_ = (cv::Mat_<double>(4, 1) << pointBehindArmor.x, pointBehindArmor.y, pointBehindArmor.z, yawAngle);
    if(temp_i % 2 == 0) initializeMeasurementMatrix1_1(yawAngle, state_.at<double>(8)); 
    else{
        while(abs_yaw(yawAngle - state_.at<double>(11)) >= CV_PI / 4) yawAngle = yawinrange(yawAngle + CV_PI / 2);
        initializeMeasurementMatrix1_2(yawAngle, state_.at<double>(9));
    }
    kf_.correct(meas1_);

    // 更新更新时间
    last_update_time = armor.frame_id; 
    lost_ = false;
}

void Tracker::update2(const Armor& armor1, const Armor& armor2) {
    cv::Point3f pointBehindArmor1 = armor1.calculatePointBehindArmor(state_.at<double>(8));
    cv::Point3f pointBehindArmor2 = armor2.calculatePointBehindArmor(state_.at<double>(9));
    double yawAngle1 = armor1.calculateYawAngle(); 
    double yawAngle2 = armor1.calculateYawAngle(); 
    while(abs_yaw(yawAngle1 - state_.at<double>(10)) >= CV_PI / 4) yawAngle1 = yawinrange(yawAngle1 + CV_PI / 2), yawAngle2 = yawinrange(yawAngle2 + CV_PI / 2); 
    // 创建一个 10x1 的 cv::Mat，包含两个点的坐标和 yaw 角度
    double A = cos(yawAngle1) * sin(yawAngle2) + cos(yawAngle2) * sin(yawAngle1);
    double r1 = -(pointBehindArmor1.x * cos(yawAngle2) - pointBehindArmor2.x * cos(yawAngle2) + pointBehindArmor1.y * sin(yawAngle2) - pointBehindArmor2.y * sin(yawAngle2)) / A; 
    double r2 = -(pointBehindArmor1.x * cos(yawAngle1) - pointBehindArmor2.x * cos(yawAngle1) - pointBehindArmor1.y * sin(yawAngle1) + pointBehindArmor2.y * sin(yawAngle1)) / A;
    meas2_ = (cv::Mat_<double>(10, 1) << pointBehindArmor1.x, pointBehindArmor1.y, pointBehindArmor1.z, yawAngle1, pointBehindArmor2.x, pointBehindArmor2.y, pointBehindArmor2.z, yawAngle2, r1, r2);
    initializeMeasurementMatrix2(yawAngle1, yawAngle2, r1, r2); 
    kf_.correct(meas2_);

    // 更新更新时间
    last_update_time = armor1.frame_id; 
    lost_ = false;
}

cv::Point3f Tracker::getPosition() const {
    return cv::Point3f(state_.at<double>(0), state_.at<double>(1), state_.at<double>(2));
}

cv::Point3f Tracker::getVelocity() const {
    return cv::Point3f(state_.at<double>(4), state_.at<double>(5), state_.at<double>(6));
}

bool Tracker::isLost() const {
    return lost_;
}

void Tracker::markLost(const int64& frame_id) {
    if(last_update_time != frame_id) lost_ = true;
}

bool Tracker::isExpired(const int64& frame_id, const int64& gap_time) const {
    return frame_id - last_update_time > gap_time;
}

std::pair<double, double> Tracker::getR() const {
    return std::make_pair(state_.at<double>(8), state_.at<double>(9));
}

void Tracker::initializeMeasurementMatrix1_1(double theta1, double r) {
    kf_.measurementMatrix = cv::Mat::zeros(4, 12, CV_64F);  // 初始化一个 4x12 的全零矩阵 (CV_64F 表示 double 类型)
    kf_.measurementNoiseCov = cv::Mat::eye(4, 4, CV_64F) * 1e-5;
    kf_.measurementMatrix.at<double>(3, 3) = 0.1;

    // 赋值 1 的块 (左上角的 3x3 单位矩阵)
    kf_.measurementMatrix.at<double>(0, 0) = 1;
    kf_.measurementMatrix.at<double>(1, 1) = 1;
    kf_.measurementMatrix.at<double>(2, 2) = 1;
    kf_.measurementMatrix.at<double>(3, 10) = 1;

    // theta1 的三角函数分量
    kf_.measurementMatrix.at<double>(0, 8) = -cos(theta1);
    kf_.measurementMatrix.at<double>(0, 10) = r * sin(theta1);
    kf_.measurementMatrix.at<double>(1, 8) = -sin(theta1);
    kf_.measurementMatrix.at<double>(1, 10) = -r * cos(theta1);
}
void Tracker::initializeMeasurementMatrix1_2(double theta1, double r) {
    kf_.measurementMatrix = cv::Mat::zeros(4, 12, CV_64F);  // 初始化一个 4x12 的全零矩阵 (CV_64F 表示 double 类型)
    kf_.measurementNoiseCov = cv::Mat::eye(4, 4, CV_64F) * 1e-5;
    kf_.measurementMatrix.at<double>(3, 3) = 0.1;

    // 赋值 1 的块 (左上角的 3x3 单位矩阵)
    kf_.measurementMatrix.at<double>(0, 0) = 1;
    kf_.measurementMatrix.at<double>(1, 1) = 1;
    kf_.measurementMatrix.at<double>(2, 3) = 1;
    kf_.measurementMatrix.at<double>(3, 11) = 1;

    // theta1 的三角函数分量
    kf_.measurementMatrix.at<double>(0, 9) = -cos(theta1); 
    kf_.measurementMatrix.at<double>(0, 11) = r * sin(theta1); 
    kf_.measurementMatrix.at<double>(1, 9) = -sin(theta1); 
    kf_.measurementMatrix.at<double>(1, 11) = -r * cos(theta1); 
}
void Tracker::initializeMeasurementMatrix2(double theta1, double theta2, double r1, double r2) { 
    kf_.measurementMatrix = cv::Mat::zeros(10, 12, CV_64F);  // 初始化一个 10x12 的全零矩阵 (CV_64F 表示 double 类型)
    kf_.measurementNoiseCov = cv::Mat::eye(10, 10, CV_64F) * 1e-5; 
    kf_.measurementMatrix.at<double>(3, 3) = 0.1; 
    kf_.measurementMatrix.at<double>(7, 7) = 0.1; 
    kf_.measurementMatrix.at<double>(8, 8) = 0.01; 
    kf_.measurementMatrix.at<double>(9, 9) = 0.01; 

    // 赋值 1 的块 (左上和右下的单位矩阵部分)
    kf_.measurementMatrix.at<double>(0, 0) = 1; kf_.measurementMatrix.at<double>(1, 1) = 1; kf_.measurementMatrix.at<double>(2, 2) = 1;
    kf_.measurementMatrix.at<double>(3, 0) = 1; kf_.measurementMatrix.at<double>(4, 1) = 1; kf_.measurementMatrix.at<double>(5, 2) = 1;
    kf_.measurementMatrix.at<double>(6, 3) = 1; kf_.measurementMatrix.at<double>(7, 4) = 1; kf_.measurementMatrix.at<double>(8, 5) = 1; 
    kf_.measurementMatrix.at<double>(9, 6) = 1;

    // theta1 的三角函数分量
    kf_.measurementMatrix.at<double>(0, 7) = -cos(theta1);
    kf_.measurementMatrix.at<double>(0, 9) = r1 * sin(theta1);
    kf_.measurementMatrix.at<double>(1, 7) = -sin(theta1);
    kf_.measurementMatrix.at<double>(1, 9) = -r1 * cos(theta1);

    // theta2 的三角函数分量
    kf_.measurementMatrix.at<double>(3, 7) = -cos(theta2);
    kf_.measurementMatrix.at<double>(3, 9) = r2 * sin(theta2);
    kf_.measurementMatrix.at<double>(4, 7) = -sin(theta2);
    kf_.measurementMatrix.at<double>(4, 9) = -r2 * cos(theta2); 

    // 其他单元素的赋值
    kf_.measurementMatrix.at<double>(2, 10) = 1;
    kf_.measurementMatrix.at<double>(5, 10) = 1;
    kf_.measurementMatrix.at<double>(6, 11) = 1;
    kf_.measurementMatrix.at<double>(9, 11) = 1;
}
bool isSameArmor(const Tracker& tracker, const Armor& armor) {
    cv::Point3f trackerPos = tracker.getPosition();
    cv::Point3f armorPos(armor.ex_mat.at<double>(2, 3), armor.ex_mat.at<double>(0, 3), -armor.ex_mat.at<double>(1, 3)); // 从装甲板的外参矩阵中提取位置信息

    // 计算装甲板和底盘中心的距离
    double distance = cv::norm(trackerPos - armorPos); 
    if(distance < 0.15 || distance > 0.4) return false;
    // double yaw1 = tracker.state_.at<double>(10); 
    // double yaw = armor.calculateYawAngle();
    // if(abs_yaw(yaw - yaw1) < CV_PI / 12) return true;
    // yaw1 = yawinrange(yaw1 + CV_PI / 2);
    // if(abs_yaw(yaw - yaw1) < CV_PI / 12) return true;
    // yaw1 = yawinrange(yaw1 + CV_PI / 2);
    // if(abs_yaw(yaw - yaw1) < CV_PI / 12) return true;
    // yaw1 = yawinrange(yaw1 + CV_PI / 2);
    // if(abs_yaw(yaw - yaw1) < CV_PI / 12) return true;  
    return true;
}
double abs_yaw(double x){
    while(x > CV_PI * 2) x -= CV_PI * 2;
    while(x < 0) x += CV_PI * 2;
    if(x > CV_PI) x = CV_PI * 2 - x; 
    return x; 
}
double yawinrange(double x){
    while(x > CV_PI) x -= CV_PI * 2;
    while(x < -CV_PI) x += CV_PI * 2;
    return x; 
}
std::vector<Armor> calculateArmorPositions(const Tracker& tracker) {
    std::vector<Armor> armors;

    // 获取底盘核心位置
    cv::Point3f chassis_center(tracker.state_.at<double>(0), tracker.state_.at<double>(1), tracker.state_.at<double>(2));

    // 获取旋转角
    double yaw1 = tracker.state_.at<double>(10);

    // 获取底盘半径
    double r[2]; 
    r[1] = tracker.state_.at<double>(8);
    r[0] = tracker.state_.at<double>(9);
    for(int i = 1; i <= 4; i++){
        Armor armor;
        armor.is_small = tracker.issmall;
        armor.classification = tracker.classification;
        armor.frame_id = tracker.last_update_time;
        // 现有的旋转矩阵
        cv::Mat rotationMatrixYaw = (cv::Mat_<double>(3, 3) << 
            cos(yaw1), 0, sin(yaw1), 
            0, 1, 0, 
            -sin(yaw1), 0, cos(yaw1));

        // 俯仰角的旋转矩阵
        double pitch = 15 * M_PI / 180; // 15度转换为弧度
        cv::Mat rotationMatrixPitch = (cv::Mat_<double>(3, 3) << 
            1, 0, 0, 
            0, cos(pitch), -sin(pitch), 
            0, sin(pitch), cos(pitch));

        // 最终的旋转矩阵
        cv::Mat finalRotationMatrix = rotationMatrixPitch * rotationMatrixYaw;
        cv::Mat transformMatrix = cv::Mat::eye(4, 4, CV_64F);
        finalRotationMatrix.copyTo(transformMatrix(cv::Rect(0, 0, 3, 3)));
        transformMatrix.at<double>(3, 3) = 1.0;
        armor.ex_mat = transformMatrix;
        cv::Point3f position(r[i % 2] * cos(yaw1) + chassis_center.x, r[i % 2] * sin(yaw1) + chassis_center.y, chassis_center.z); 
        armor.ex_mat.at<double>(0, 3) = position.y;
        armor.ex_mat.at<double>(1, 3) = -position.z;
        armor.ex_mat.at<double>(2, 3) = position.x;
        armor.calculatemergedRect();
        armors.push_back(armor);
        yaw1 = yawinrange(yaw1 + CV_PI / 2); 
    }

    return armors;
}