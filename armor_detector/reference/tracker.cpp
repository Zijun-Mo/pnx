// Copyright 2022 Chen Jun

#include "armor_tracker/tracker.hpp"

#include <angles/angles.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/convert.h>

#include <rclcpp/logger.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

// STD
#include <cfloat>
#include <memory>
#include <string>

namespace rm_auto_aim
{

// 构造函数，初始化跟踪器
Tracker::Tracker(double max_match_distance, double max_match_yaw_diff)
: tracker_state(LOST),
  tracked_id(std::string("")),
  measurement(Eigen::VectorXd::Zero(4)),
  target_state(Eigen::VectorXd::Zero(9)),
  max_match_distance_(max_match_distance),
  max_match_yaw_diff_(max_match_yaw_diff)
{
}

// 初始化跟踪器
void Tracker::init(const Armors::SharedPtr & armors_msg)
{
  if (armors_msg->armors.empty()) {
    return;
  }

  // 简单地选择距离图像中心最近的装甲板
  double min_distance = DBL_MAX;
  tracked_armor = armors_msg->armors[0];
  for (const auto & armor : armors_msg->armors) {
    if (armor.distance_to_image_center < min_distance) {
      min_distance = armor.distance_to_image_center;
      tracked_armor = armor;
    }
  }

  // 初始化扩展卡尔曼滤波器（EKF）
  initEKF(tracked_armor);
  RCLCPP_DEBUG(rclcpp::get_logger("armor_tracker"), "Init EKF!");

  // 设置跟踪的装甲板 ID 和状态
  tracked_id = tracked_armor.number;
  tracker_state = DETECTING;

  // 更新装甲板数量
  updateArmorsNum(tracked_armor);
}

// 更新跟踪器状态
void Tracker::update(const Armors::SharedPtr & armors_msg)
{
  // EKF 预测
  Eigen::VectorXd ekf_prediction = ekf.predict();
  RCLCPP_DEBUG(rclcpp::get_logger("armor_tracker"), "EKF predict");

  bool matched = false;
  // 如果没有找到匹配的装甲板，使用 EKF 预测的目标状态
  target_state = ekf_prediction;

  if (!armors_msg->armors.empty()) {
    // 查找具有相同 ID 的最近的装甲板
    Armor same_id_armor;
    int same_id_armors_count = 0;
    auto predicted_position = getArmorPositionFromState(ekf_prediction);
    double min_position_diff = DBL_MAX;
    double yaw_diff = DBL_MAX;
    for (const auto & armor : armors_msg->armors) {
      // 只考虑具有相同 ID 的装甲板
      if (armor.number == tracked_id) {
        same_id_armor = armor;
        same_id_armors_count++;
        // 计算预测位置和当前装甲板位置之间的差异
        auto p = armor.pose.position;
        Eigen::Vector3d position_vec(p.x, p.y, p.z);
        double position_diff = (predicted_position - position_vec).norm();
        if (position_diff < min_position_diff) {
          // 找到最近的装甲板
          min_position_diff = position_diff;
          yaw_diff = abs(orientationToYaw(armor.pose.orientation) - ekf_prediction(6));
          tracked_armor = armor;
        }
      }
    }

    // 存储跟踪器信息
    info_position_diff = min_position_diff;
    info_yaw_diff = yaw_diff;

    // 检查最近的装甲板的距离和偏航角差异是否在阈值内
    if (min_position_diff < max_match_distance_ && yaw_diff < max_match_yaw_diff_) {
      // 找到匹配的装甲板
      matched = true;
      auto p = tracked_armor.pose.position;
      // 更新 EKF
      double measured_yaw = orientationToYaw(tracked_armor.pose.orientation);
      measurement = Eigen::Vector4d(p.x, p.y, p.z, measured_yaw);
      target_state = ekf.update(measurement);
      RCLCPP_DEBUG(rclcpp::get_logger("armor_tracker"), "EKF update");
    } else if (same_id_armors_count == 1 && yaw_diff > max_match_yaw_diff_) {
      // 没有找到匹配的装甲板，但只有一个具有相同 ID 的装甲板
      // 并且偏航角发生了跳变，认为目标在旋转并且装甲板跳变
      handleArmorJump(same_id_armor);
    } else {
      // 没有找到匹配的装甲板
      RCLCPP_WARN(rclcpp::get_logger("armor_tracker"), "No matched armor found!");
    }
  }

  // 防止半径扩散
  if (target_state(8) < 0.12) {
    target_state(8) = 0.12;
    ekf.setState(target_state);
  } else if (target_state(8) > 0.4) {
    target_state(8) = 0.4;
    ekf.setState(target_state);
  }

  // 跟踪状态机
  if (tracker_state == DETECTING) {
    if (matched) {
      detect_count_++;
      if (detect_count_ > tracking_thres) {
        detect_count_ = 0;
        tracker_state = TRACKING;
      }
    } else {
      detect_count_ = 0;
      tracker_state = LOST;
    }
  } else if (tracker_state == TRACKING) {
    if (!matched) {
      tracker_state = TEMP_LOST;
      lost_count_++;
    }
  } else if (tracker_state == TEMP_LOST) {
    if (!matched) {
      lost_count_++;
      if (lost_count_ > lost_thres) {
        lost_count_ = 0;
        tracker_state = LOST;
      }
    } else {
      tracker_state = TRACKING;
      lost_count_ = 0;
    }
  }
}

// 初始化扩展卡尔曼滤波器（EKF）
void Tracker::initEKF(const Armor & a)
{
  double xa = a.pose.position.x;
  double ya = a.pose.position.y;
  double za = a.pose.position.z;
  last_yaw_ = 0;
  double yaw = orientationToYaw(a.pose.orientation);

  // 设置初始位置在目标后方 0.2 米处
  target_state = Eigen::VectorXd::Zero(9);
  double r = 0.26;
  double xc = xa + r * cos(yaw);
  double yc = ya + r * sin(yaw);
  dz = 0, another_r = r;
  target_state << xc, 0, yc, 0, za, 0, yaw, 0, r;

  ekf.setState(target_state);
}

// 更新装甲板数量
void Tracker::updateArmorsNum(const Armor & armor)
{
  if (armor.type == "large" && (tracked_id == "3" || tracked_id == "4" || tracked_id == "5")) {
    tracked_armors_num = ArmorsNum::BALANCE_2;
  } else if (tracked_id == "outpost") {
    tracked_armors_num = ArmorsNum::OUTPOST_3;
  } else {
    tracked_armors_num = ArmorsNum::NORMAL_4;
  }
}

// 处理装甲板跳变
void Tracker::handleArmorJump(const Armor & current_armor)
{
  double yaw = orientationToYaw(current_armor.pose.orientation);
  target_state(6) = yaw;
  updateArmorsNum(current_armor);
  // 只有 4 个装甲板具有 2 个半径和高度
  if (tracked_armors_num == ArmorsNum::NORMAL_4) {
    dz = target_state(4) - current_armor.pose.position.z;
    target_state(4) = current_armor.pose.position.z;
    std::swap(target_state(8), another_r);
  }
  RCLCPP_WARN(rclcpp::get_logger("armor_tracker"), "Armor jump!");

  // 如果位置差异大于 max_match_distance_，认为 EKF 发散，重置状态
  auto p = current_armor.pose.position;
  Eigen::Vector3d current_p(p.x, p.y, p.z);
  Eigen::Vector3d infer_p = getArmorPositionFromState(target_state);
  if ((current_p - infer_p).norm() > max_match_distance_) {
    double r = target_state(8);
    target_state(0) = p.x + r * cos(yaw);  // xc
    target_state(1) = 0;                   // vxc
    target_state(2) = p.y + r * sin(yaw);  // yc
    target_state(3) = 0;                   // vyc
    target_state(4) = p.z;                 // za
    target_state(5) = 0;                   // vza
    RCLCPP_ERROR(rclcpp::get_logger("armor_tracker"), "Reset State!");
  }

  ekf.setState(target_state);
}

// 将四元数转换为偏航角
double Tracker::orientationToYaw(const geometry_msgs::msg::Quaternion & q)
{
  // 获取装甲板的偏航角
  tf2::Quaternion tf_q;
  tf2::fromMsg(q, tf_q);
  double roll, pitch, yaw;
  tf2::Matrix3x3(tf_q).getRPY(roll, pitch, yaw);
  // 使偏航角变化连续（-pi~pi 到 -inf~inf）
  yaw = last_yaw_ + angles::shortest_angular_distance(last_yaw_, yaw);
  last_yaw_ = yaw;
  return yaw;
}

// 从状态向量中获取装甲板位置
Eigen::Vector3d Tracker::getArmorPositionFromState(const Eigen::VectorXd & x)
{
  // 计算当前装甲板的预测位置
  double xc = x(0), yc = x(2), za = x(4);
  double yaw = x(6), r = x(8);
  double xa = xc - r * cos(yaw);
  double ya = yc - r * sin(yaw);
  return Eigen::Vector3d(xa, ya, za);
}

}  // namespace rm_auto_aim