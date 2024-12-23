#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include "detector.hpp"
// 将图像转换为灰度图像并进行二值化
Detector::Detector(){

}
cv::Mat Detector::convertToAdaptiveBinary(const cv::Mat& img, const cv::Ptr<cv::CLAHE> clahe, const int& clipLimit) {
    originalImg = img; 
    // 将图像转换为红色通道减去蓝色通道的灰度图像
    cv::Mat channels[3];
    cv::split(originalImg, channels); // 分割图像为三个通道
    grayImg = channels[2] - 0.3 * channels[0]; // 红色通道减去蓝色通道

    // 对灰度图像进行亮度自适应（CLAHE）
    // 应用CLAHE到灰度图像
    clahe->apply(grayImg, equalizedImg);
    // imshow("equalizedImg", equalizedImg); 
    // cv::waitKey(30); 

    // 进行全局二值化
    cv::threshold(equalizedImg, binaryImg, clipLimit, 255, cv::THRESH_BINARY);
    // std::cout << (cv::getTickCount() - start) / cv::getTickFrequency() << "\n"; 

    // 去除小于9个像素的明亮噪点
    cv::Mat morphKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(binaryImg, binaryImg, cv::MORPH_OPEN, morphKernel); 
    // imshow("binaryImg", binaryImg);
    // cv::waitKey(30);

    return binaryImg;
}
// 处理轮廓
std::vector<cv::RotatedRect> Detector::processContours() {
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::RotatedRect> rectangles;

    // 查找轮廓
    cv::findContours(binaryImg, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    for (const auto& contour : contours) {
        // 跳过包围像素过少的轮廓
        if (cv::contourArea(contour) < 10 || cv::contourArea(contour) > 2000) {
            continue;
        }

        // 计算轮廓的最小外接可旋转矩形
        cv::RotatedRect minRect = cv::minAreaRect(contour);
        if(!this->isLight(minRect, contour)) {
            continue; 
        }

        // 判断矩形区域内的像素颜色
        if (!this->isRedDominant(minRect)) {
            continue; 
        }
        // 绘制矩形在原图上
        // cv::Point2f rectPoints[4];
        // minRect.points(rectPoints);
        // for (int j = 0; j < 4; j++) {
        //     cv::line(originalImg, rectPoints[j], rectPoints[(j + 1) % 4], cv::Scalar(0, 255, 0), 1);
        // }
        // 保存矩形在动态数组中
        rectangles.push_back(minRect);
    }

    return rectangles;
}
bool Detector::isLight(cv::RotatedRect& rect, const std::vector<cv::Point>& contour) {
    if (rect.size.width > rect.size.height) {
        std::swap(rect.size.width, rect.size.height); 
        rect.angle -= 90.0; // 调整角度，使其与短边一致
    }
    if(rect.size.height < 1.0 * rect.size.width) return false; 
    if(std::abs(rect.angle) > 40.0) return false; 

    // 计算最小外接矩形的面积
    double rectArea = rect.size.width * rect.size.height;
    // 计算轮廓的面积
    double contourArea = cv::contourArea(contour);
    // 判断面积和面积比
    if (rectArea < 50 && contourArea / rectArea <= 0.4) return false;
    if (rectArea >= 50 && contourArea / rectArea <= 0.6) return false;

    return true; 
}
bool Detector::isSimilarRotatedRect(const cv::RotatedRect& rect1, const cv::RotatedRect& rect2, bool &issmall) {
    // 计算旋转角度差
    if (std::abs(rect1.angle - rect2.angle) > 10.0) return false; 
    // 计算形状大小差异
    double height1 = rect1.size.height; 
    double height2 = rect2.size.height; 

    double size_diff_height = std::abs(height1 - height2) / std::max(height1, height2);
    if (size_diff_height > 0.3) return false; 

    // 计算两个矩形之间的间距与矩形长边的比
    double distance_ratio = cv::norm(rect1.center - rect2.center) / std::max(height1, height2);
    if (distance_ratio < 1.0 || distance_ratio > 6.0) return false; 
    if (distance_ratio < 3.0) issmall = true; 
    else issmall = false;

    // std::cout << "distance_ratio: " << distance_ratio << std::endl; 

    return true;
}
bool Detector::isRedDominant(const cv::RotatedRect& minRect) {
    // 获取旋转矩形的四个顶点
    cv::Point2f vertices[4];
    minRect.points(vertices);

    // 获取旋转矩形的边界矩形
    cv::Rect boundingRect = minRect.boundingRect();

    // 初始化红色通道和蓝色通道的和
    double redSum = 0;
    double blueSum = 0;

    // 枚举边界矩形中的每个点
    for (int y = boundingRect.y; y < boundingRect.y + boundingRect.height; ++y) {
        for (int x = boundingRect.x; x < boundingRect.x + boundingRect.width; ++x) {
            // 检查点是否在旋转矩形内
            if (cv::pointPolygonTest(std::vector<cv::Point2f>(vertices, vertices + 4), cv::Point2f(x, y), false) >= 0) {
                // 获取像素值
                cv::Vec3b pixel = originalImg.at<cv::Vec3b>(y, x);
                // 累加红色通道和蓝色通道的值
                redSum += pixel[2];  // 红色通道
                blueSum += pixel[0]; // 蓝色通道
            }
        }
    }

    // 判断红色通道的和是否大于蓝色通道的和
    return redSum > blueSum * 1.1;
}
cv::Mat Detector::performPCA(const cv::Mat& roiImage) {
    // 将ROI图像转换为浮点型
    cv::Mat floatRoiImage;
    roiImage.convertTo(floatRoiImage, CV_64F);

    // 归一化图像亮度值到0到1之间
    cv::normalize(floatRoiImage, floatRoiImage, 0, 25, cv::NORM_MINMAX);

    // 计算质心
    cv::Moments moments = cv::moments(floatRoiImage, false);
    cv::Point2f centroid = cv::Point2f(moments.m10 / moments.m00, moments.m01 / moments.m00);

    // 初始化点云
    std::vector<cv::Point2f> points = {}; 
    for (int i = 0; i < floatRoiImage.rows; i++) {
        for (int j = 0; j < floatRoiImage.cols; j++) {
            double intensity = floatRoiImage.at<double>(i, j);
            for (int k = 0; k < std::round(intensity); k++) {
                points.emplace_back(cv::Point2f(j, i));
            }
        }
    }

    // 确保有足够的点进行 PCA
    if (points.size() < 2) {
        return cv::Mat::zeros(1, 4, CV_64F);
    }

    // 将 points 转换为 cv::Mat
    cv::Mat data(points.size(), 2, CV_64F);
    for (size_t i = 0; i < points.size(); ++i) {
        data.at<double>(i, 0) = points[i].x;
        data.at<double>(i, 1) = points[i].y;
    }

    // 执行 PCA
    cv::PCA pca(data, cv::Mat(), cv::PCA::DATA_AS_ROW);
    cv::Mat eigenvector = pca.eigenvectors.row(0);

    // 将PCA得到的向量变为以质心为起点，方向不变的向量
    cv::Point2f pcaVector(eigenvector.at<double>(0, 0), eigenvector.at<double>(0, 1));

    // 返回以质心为起点的单位向量
    return (cv::Mat_<double>(1, 4) << centroid.x, centroid.y, pcaVector.x, pcaVector.y);
}

// findExtremePoints 函数实现
std::pair<cv::Point2f, cv::Point2f> Detector::findExtremePoints(const cv::Mat& roiImage, 
                                                                const cv::Point2f& symmetryAxis, 
                                                                const cv::Size2f& rectSize, 
                                                                const cv::Point2f& rectCenter, 
                                                                double mean_val) {
    cv::Mat roiImage64F;
    roiImage.convertTo(roiImage64F, CV_64F);
    cv::normalize(roiImage64F, roiImage64F, 0, 255, cv::NORM_MINMAX);
    mean_val = mean(roiImage64F)[0];
    // 初始化最大亮度变化值和对应的点坐标
    std::vector<cv::Point2f> topPoints, bottomPoints; 

    // 计算对称轴的单位向量
    cv::Point2f unitSymmetryAxis = symmetryAxis / cv::norm(symmetryAxis);

    // 遍历对称轴两侧一定宽度内的每条平行于对称轴的线
    for (double offset = -rectSize.width * 0.2; offset <= rectSize.width * 0.2; offset += 0.5) {
        double maxGradientTop = 0;
        double maxGradientBottom = 0;
        cv::Point2f topPoint, bottomPoint;

        for (double t = -rectSize.height / 2; t <= rectSize.height / 2; t += 1.0) {
            // 计算当前线上的点
            cv::Point2f point = rectCenter + t * unitSymmetryAxis + cv::Point2f(offset * unitSymmetryAxis.y, -offset * unitSymmetryAxis.x); 
            // 确保点在ROI范围内
            if (point.x >= 0 && point.x < roiImage64F.cols && point.y >= 0 && point.y < roiImage64F.rows) {
                // 获取当前点的亮度值
                double intensity = roiImage64F.at<double>(point);

                // 计算沿对称轴方向的梯度
                cv::Point2f gradientPoint = point + unitSymmetryAxis;
                if (gradientPoint.x >= 0 && gradientPoint.x < roiImage64F.cols && 
                    gradientPoint.y >= 0 && gradientPoint.y < roiImage64F.rows) {
                    double gradientIntensity = roiImage64F.at<double>(gradientPoint); 
                    double gradient = std::abs(gradientIntensity - intensity); 

                    // 如果t大于0，说明点在对称轴上方
                    if (t > 0 && gradient > maxGradientTop) {
                        maxGradientTop = gradient;
                        topPoint = point;
                    } 
                    // 如果t小于0，说明点在对称轴下方
                    else if (t < 0 && gradient > maxGradientBottom) {
                        maxGradientBottom = gradient;
                        bottomPoint = gradientPoint;
                    } 
                }
            }
        }

        // 记录找到的角点
        if (maxGradientTop > 0) {
            topPoints.push_back(topPoint);
        }
        if (maxGradientBottom > 0) {
            bottomPoints.push_back(bottomPoint);
        }
    }

    // 计算上下角点的平均值
    cv::Point2f avgTopPoint(0, 0), avgBottomPoint(0, 0);
    for (const auto& point : topPoints) {
        avgTopPoint += point; 
    }
    for (const auto& point : bottomPoints) {
        avgBottomPoint += point; 
    }
    if (!topPoints.empty()) {
        avgTopPoint /= static_cast<double>(topPoints.size());
    }
    if (!bottomPoints.empty()) {
        avgBottomPoint /= static_cast<double>(bottomPoints.size());
    }
    if(1) {
         // 将图像从 CV_64F 转换为 CV_8U
        cv::Mat roiImage8U;
        roiImage64F.convertTo(roiImage8U, CV_8U);

        // 将灰度图像转换为彩色图像
        cv::Mat roiImageColor;
        cv::cvtColor(roiImage8U, roiImageColor, cv::COLOR_GRAY2BGR);

        // 绘制极值点之间的线
        cv::line(roiImageColor, avgTopPoint, avgBottomPoint, cv::Scalar(0, 0, 255), 1);

        // 计算方向向量的终点
        cv::Point2f directionEndPoint = rectCenter + symmetryAxis * 50;


        // 边缘检测
        cv::Mat edges;
        cv::Canny(roiImage8U, edges, 150, 250); 

        // 将边缘绘制到彩色图像上
        for (int y = 0; y < edges.rows; y++) {
            for (int x = 0; x < edges.cols; x++) {
                if (edges.at<uchar>(y, x) > 0) {
                    roiImageColor.at<cv::Vec3b>(y, x) = cv::Vec3b(255, 0, 0); // 蓝色
                }
            }
        }

        // 绘制从质心开始的方向向量
        // cv::line(roiImageColor, rectCenter, directionEndPoint, cv::Scalar(0, 255, 0), 1);
        cv::imshow("roiImageColor", roiImageColor);
        cv::waitKey(0);
    }

    // 返回对称轴上方和下方亮度变化最大的点的平均值
    return std::make_pair(avgTopPoint, avgBottomPoint);
}

// mergeSimilarRects 函数实现
std::vector<cv::Point2f> Detector::mergeSimilarRects(const cv::RotatedRect& rect1, const cv::RotatedRect& rect2) {
    // 获取两个旋转矩形的四个顶点
    cv::Point2f vertices1[4];
    cv::Point2f vertices2[4];
    rect1.points(vertices1);
    rect2.points(vertices2); 

    // 扩展矩形的长宽为原本的1.2倍
    cv::Size2f expandedSize1(rect1.size.width * 1.2, rect1.size.height * 1.5);
    cv::Size2f expandedSize2(rect2.size.width * 1.2, rect2.size.height * 1.5);
    cv::RotatedRect expandedRect1(rect1.center, expandedSize1, rect1.angle);
    cv::RotatedRect expandedRect2(rect2.center, expandedSize2, rect2.angle);

    // 获取扩展后的ROI
    cv::Rect roi1 = expandedRect1.boundingRect(); 
    cv::Rect roi2 = expandedRect2.boundingRect(); 
    // 确保 ROI 在图像边界内
    roi1 &= cv::Rect(0, 0, grayImg.cols, grayImg.rows);
    roi2 &= cv::Rect(0, 0, grayImg.cols, grayImg.rows);

    cv::Mat mask1 = cv::Mat::zeros(roi1.size(), CV_8UC1); 
    cv::Mat mask2 = cv::Mat::zeros(roi2.size(), CV_8UC1);

    std::vector<cv::Point> contour1 = {vertices1[0] - cv::Point2f(roi1.x, roi1.y), 
                                       vertices1[1] - cv::Point2f(roi1.x, roi1.y), 
                                       vertices1[2] - cv::Point2f(roi1.x, roi1.y), 
                                       vertices1[3] - cv::Point2f(roi1.x, roi1.y)};
    std::vector<cv::Point> contour2 = {vertices2[0] - cv::Point2f(roi2.x, roi2.y),
                                       vertices2[1] - cv::Point2f(roi2.x, roi2.y),
                                       vertices2[2] - cv::Point2f(roi2.x, roi2.y),
                                       vertices2[3] - cv::Point2f(roi2.x, roi2.y)};
    cv::fillConvexPoly(mask1, contour1, cv::Scalar(255));
    cv::fillConvexPoly(mask2, contour2, cv::Scalar(255));

    cv::Mat roiImage1, roiImage2;
    grayImg(roi1).copyTo(roiImage1, mask1);
    grayImg(roi2).copyTo(roiImage2, mask2); 

    // 检查图像是否为空
    if (roiImage1.empty() || roiImage2.empty()) {
        std::cerr << "Error: ROI image is empty." << std::endl;
        return std::vector<cv::Point2f>();
    }

    // 分别对两个ROI进行主成分分析（PCA）
    cv::Mat symmetryAxisMat1 = performPCA(roiImage1);
    cv::Mat symmetryAxisMat2 = performPCA(roiImage2);

    // 将 cv::Mat 转换为 cv::Point2f
    cv::Point2f symmetryAxis1(symmetryAxisMat1.at<double>(0, 2), symmetryAxisMat1.at<double>(0, 3));
    cv::Point2f symmetryAxis2(symmetryAxisMat2.at<double>(0, 2), symmetryAxisMat2.at<double>(0, 3));
    cv::Size2f rectSize1(rect1.size.width, rect1.size.height * 1.5);
    cv::Size2f rectSize2(rect2.size.width, rect2.size.height * 1.5);
    cv::Point2f rectcenter1(symmetryAxisMat1.at<double>(0, 0), symmetryAxisMat1.at<double>(0, 1));
    cv::Point2f rectcenter2(symmetryAxisMat2.at<double>(0, 0), symmetryAxisMat2.at<double>(0, 1));

    // 计算矩形内的平均亮度
    double meanVal1 = cv::mean(roiImage1)[0];
    double meanVal2 = cv::mean(roiImage2)[0];

    // 在对称轴上找到上下两个亮度变化最大的点
    std::pair<cv::Point2f, cv::Point2f> extremePoints1 = findExtremePoints(roiImage1, symmetryAxis1, rectSize1, rectcenter1, meanVal1);
    std::pair<cv::Point2f, cv::Point2f> extremePoints2 = findExtremePoints(roiImage2, symmetryAxis2, rectSize2, rectcenter2, meanVal2);

    // 将 ROI 中的点坐标转换为原图像中的全局坐标
    cv::Point2f topPoint1 = extremePoints1.first + cv::Point2f(roi1.x, roi1.y);
    cv::Point2f bottomPoint1 = extremePoints1.second + cv::Point2f(roi1.x, roi1.y);
    cv::Point2f topPoint2 = extremePoints2.first + cv::Point2f(roi2.x, roi2.y);
    cv::Point2f bottomPoint2 = extremePoints2.second + cv::Point2f(roi2.x, roi2.y);

    if (rect1.size.width <= 4) {
        topPoint1 = (vertices1[0] + vertices1[1]) / 2 + cv::Point2f(roi1.x, roi1.y);
        bottomPoint1 = (vertices1[2] + vertices1[3]) / 2 + cv::Point2f(roi1.x, roi1.y);
    }
    if (rect2.size.width <= 4) {
        topPoint2 = (vertices2[0] + vertices2[1]) / 2 + cv::Point2f(roi2.x, roi2.y);
        bottomPoint2 = (vertices2[2] + vertices2[3]) / 2 + cv::Point2f(roi2.x, roi2.y);
    }
    // 沿着对称轴上下延长1倍，作为装甲板的四个顶点
    cv::Point2f center1 = (topPoint1 + bottomPoint1) / 2;
    cv::Point2f center2 = (topPoint2 + bottomPoint2) / 2;
    cv::Point2f extendedTop1 = center1 + (topPoint1 - center1) * 2.4;
    cv::Point2f extendedBottom1 = center1 + (bottomPoint1 - center1) * 2.4;
    cv::Point2f extendedTop2 = center2 + (topPoint2 - center2) * 2.4;
    cv::Point2f extendedBottom2 = center2 + (bottomPoint2 - center2) * 2.4;
    std::vector<cv::Point2f> armorPoints = {extendedTop1, extendedTop2, extendedBottom2, extendedBottom1};

    return armorPoints;
}
cv::Mat Detector::warpToRectangle(const cv::Mat& img, const std::vector<cv::Point2f>& quad, int width, int height) {
    // 定义矩形的四个顶点
    std::vector<cv::Point2f> rectangle = {
        cv::Point2f(0, 0),
        cv::Point2f(width - 1, 0),
        cv::Point2f(width - 1, height - 1),
        cv::Point2f(0, height - 1)
    };

    // 计算透视变换矩阵
    cv::Mat transformMatrix = cv::getPerspectiveTransform(quad, rectangle);

    // 进行透视变换
    cv::Mat warpedImg;
    cv::warpPerspective(img, warpedImg, transformMatrix, cv::Size(width, height));

    return warpedImg;
}