// Maintained by Shenglin Qin, Chengfu Zou
// Copyright (C) FYT Vision Group. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "armor_detector/light_corner_corrector.hpp"

#include <numeric>

namespace fyt::auto_aim {

void LightCornerCorrector::correctCorners(Armor &armor, const cv::Mat &gray_img) {
  // 如果灯光的宽度太小，则不进行校正
  constexpr int PASS_OPTIMIZE_WIDTH = 3;

  // 校正左灯光的角点
  if (armor.left_light.width > PASS_OPTIMIZE_WIDTH) {
    // 找到左灯光的对称轴
    SymmetryAxis left_axis = findSymmetryAxis(gray_img, armor.left_light);
    armor.left_light.center = left_axis.centroid; // 更新左灯光的中心
    armor.left_light.axis = left_axis.direction;  // 更新左灯光的轴方向

    // 找到左灯光的顶部角点
    if (cv::Point2f t = findCorner(gray_img, armor.left_light, left_axis, "top"); t.x > 0) {
      armor.left_light.top = t; // 更新左灯光的顶部角点
    }

    // 找到左灯光的底部角点
    if (cv::Point2f b = findCorner(gray_img, armor.left_light, left_axis, "bottom"); b.x > 0) {
      armor.left_light.bottom = b; // 更新左灯光的底部角点
    }
  }

  // 校正右灯光的角点
  if (armor.right_light.width > PASS_OPTIMIZE_WIDTH) {
    // 找到右灯光的对称轴
    SymmetryAxis right_axis = findSymmetryAxis(gray_img, armor.right_light);
    armor.right_light.center = right_axis.centroid; // 更新右灯光的中心
    armor.right_light.axis = right_axis.direction;  // 更新右灯光的轴方向

    // 找到右灯光的顶部角点
    if (cv::Point2f t = findCorner(gray_img, armor.right_light, right_axis, "top"); t.x > 0) {
      armor.right_light.top = t; // 更新右灯光的顶部角点
    }

    // 找到右灯光的底部角点
    if (cv::Point2f b = findCorner(gray_img, armor.right_light, right_axis, "bottom"); b.x > 0) {
      armor.right_light.bottom = b; // 更新右灯光的底部角点
    }
  }
}

SymmetryAxis LightCornerCorrector::findSymmetryAxis(const cv::Mat &gray_img, const Light &light) {
  constexpr float MAX_BRIGHTNESS = 25; // 最大亮度值
  constexpr float SCALE = 0.07; // 缩放比例

  // 缩放灯光的边界框
  cv::Rect light_box = light.boundingRect();
  light_box.x -= light_box.width * SCALE;
  light_box.y -= light_box.height * SCALE;
  light_box.width += light_box.width * SCALE * 2;
  light_box.height += light_box.height * SCALE * 2;

  // 检查边界框是否超出图像范围
  light_box.x = std::max(light_box.x, 0);
  light_box.x = std::min(light_box.x, gray_img.cols - 1);
  light_box.y = std::max(light_box.y, 0);
  light_box.y = std::min(light_box.y, gray_img.rows - 1);
  light_box.width = std::min(light_box.width, gray_img.cols - light_box.x);
  light_box.height = std::min(light_box.height, gray_img.rows - light_box.y);

  // 获取归一化的灯光图像
  cv::Mat roi = gray_img(light_box);
  float mean_val = cv::mean(roi)[0]; // 计算平均亮度值
  roi.convertTo(roi, CV_32F); // 转换为浮点类型
  cv::normalize(roi, roi, 0, MAX_BRIGHTNESS, cv::NORM_MINMAX); // 归一化亮度

  // 计算质心
  cv::Moments moments = cv::moments(roi, false);
  cv::Point2f centroid = cv::Point2f(moments.m10 / moments.m00, moments.m01 / moments.m00) +
                         cv::Point2f(light_box.x, light_box.y);

  // 初始化点云
  std::vector<cv::Point2f> points;
  for (int i = 0; i < roi.rows; i++) {
    for (int j = 0; j < roi.cols; j++) {
      for (int k = 0; k < std::round(roi.at<float>(i, j)); k++) {
        points.emplace_back(cv::Point2f(j, i));
      }
    }
  }
  cv::Mat points_mat = cv::Mat(points).reshape(1);

  // 主成分分析（PCA）
  auto pca = cv::PCA(points_mat, cv::Mat(), cv::PCA::DATA_AS_ROW);

  // 获取对称轴
  cv::Point2f axis =
    cv::Point2f(pca.eigenvectors.at<float>(0, 0), pca.eigenvectors.at<float>(0, 1));

  // 归一化对称轴
  axis = axis / cv::norm(axis);

  // 确保对称轴方向向上
  if (axis.y > 0) {
    axis = -axis;
  }

  // 返回对称轴信息
  return SymmetryAxis{.centroid = centroid, .direction = axis, .mean_val = mean_val};
}

cv::Point2f LightCornerCorrector::findCorner(const cv::Mat &gray_img,
                                             const Light &light,
                                             const SymmetryAxis &axis,
                                             std::string order) {
  // 定义搜索范围的起始和结束比例
  constexpr float START = 0.8 / 2;
  constexpr float END = 1.2 / 2;

  // 检查点是否在图像范围内的lambda函数
  auto inImage = [&gray_img](const cv::Point &point) -> bool {
    return point.x >= 0 && point.x < gray_img.cols && point.y >= 0 && point.y < gray_img.rows;
  };

  // 计算两点之间距离的lambda函数
  auto distance = [](float x0, float y0, float x1, float y1) -> float {
    return std::sqrt((x0 - x1) * (x0 - x1) + (y0 - y1) * (y0 - y1));
  };

  // 根据order确定搜索方向，"top"为1，"bottom"为-1
  int oper = order == "top" ? 1 : -1;
  float L = light.length; // 光条的长度
  float dx = axis.direction.x * oper; // 对称轴方向的x增量
  float dy = axis.direction.y * oper; // 对称轴方向的y增量

  std::vector<cv::Point2f> candidates; // 存储候选角点的向量

  // 选择多个角点候选，并取平均值作为最终角点
  int n = light.width - 2; // 遍历范围
  int half_n = std::round(n / 2); // 遍历范围的一半
  for (int i = -half_n; i <= half_n; i++) {
    // 计算当前遍历点的初始位置
    float x0 = axis.centroid.x + L * START * dx + i;
    float y0 = axis.centroid.y + L * START * dy;

    cv::Point2f prev = cv::Point2f(x0, y0); // 前一个点的坐标
    cv::Point2f corner = cv::Point2f(x0, y0); // 当前候选角点的坐标
    float max_brightness_diff = 0; // 最大亮度差
    bool has_corner = false; // 是否找到角点的标志

    // 沿对称轴方向搜索，找到亮度差最大的角点
    for (float x = x0 + dx, y = y0 + dy; distance(x, y, x0, y0) < L * (END - START);
         x += dx, y += dy) {
      cv::Point2f cur = cv::Point2f(x, y); // 当前点的坐标
      if (!inImage(cv::Point(cur))) { // 检查当前点是否在图像范围内
        break;
      }

      // 计算前一个点和当前点的亮度差
      float brightness_diff = gray_img.at<uchar>(prev) - gray_img.at<uchar>(cur);
      // 如果亮度差大于最大亮度差且前一个点的亮度大于平均亮度，则更新最大亮度差和候选角点
      if (brightness_diff > max_brightness_diff && gray_img.at<uchar>(prev) > axis.mean_val) {
        max_brightness_diff = brightness_diff;
        corner = prev;
        has_corner = true;
      }

      prev = cur; // 更新前一个点的坐标
    }

    if (has_corner) { // 如果找到角点，则将其添加到候选角点列表中
      candidates.emplace_back(corner);
    }
  }

  // 计算候选角点的平均值，作为最终角点
  if (!candidates.empty()) {
    cv::Point2f result = std::accumulate(candidates.begin(), candidates.end(), cv::Point2f(0, 0));
    return result / static_cast<float>(candidates.size());
  }

  // 如果没有找到候选角点，返回(-1, -1)表示未找到
  return cv::Point2f(-1, -1);
}

}  // namespace fyt::auto_aim