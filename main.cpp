#include <iostream>
#include <vector>
#include <chrono>
#include <map>
#include <opencv2/opencv.hpp>
#include "number_classifier.hpp"
#include "pnp_solver.hpp"
#include "detector.hpp"
#include "armor.hpp"
#include "tracker.hpp"
// /opt/MVS/bin/MVS.sh

Armor armor; // 装甲板结构体

// 函数声明
bool readVideo(const std::string& filename, cv::VideoCapture& cap); // 从文件中读取视频
void Draw(cv::Mat& frame, const Armor& armor); // 绘制矩形在原图上
void Draw1(cv::Mat& frame, const Armor& armor); 

int64 start, latest_num, frame_id;
// 初始化跟踪器列表
std::map<std::string, Tracker> trackers;

int main() {
    // 打开视频文件
    cv::VideoCapture cap;
    if (!readVideo("unity_n.mp4", cap)) {
        return -1;
    }
    // 创建CLAHE对象，用于均衡亮度
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
    clahe->setClipLimit(4.0);
    // 获取视频的帧率和帧大小
    int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS); 
    // 创建视频写入对象
    cv::VideoWriter video(std::string(ROOT) + "/img_output/output_video.mp4", cv::VideoWriter::fourcc('a','v','c','1'), fps, cv::Size(frame_width, frame_height));
    start = cv::getTickCount(); 
    cv::Mat frame; 
    PnPSolver pnp_solver; // 创建pnp解算对象

    while (cap.read(frame)) {
        frame_id ++; 
        std::cout << frame_id << std::endl;
        // 翻转蓝色和红色通道(可选,在敌方为蓝方时需要)
        // std::vector<cv::Mat> channels; 
        // cv::split(frame, channels); 
        // std::swap(channels[0], channels[2]); 
        // cv::merge(channels, frame); 
        // 将图像转换为灰度图像并进行二值化
        Detector detector; 
        std::map<std::string, std::vector<Armor>> armors;
        cv::Mat binaryImg = detector.convertToAdaptiveBinary(frame, clahe, 190);
        // imshow("binaryImg", binaryImg);
        // cv::waitKey(200);
        // 处理轮廓并获取最小外接可旋转矩形
        std::vector<cv::RotatedRect> rectangles = detector.processContours(); 
        // 判断两个旋转矩形是否相似
        for (int i = 0; i < rectangles.size(); i++) {
            for (int j = i + 1; j < rectangles.size(); j++) {
                bool issmall; 
                if (detector.isSimilarRotatedRect(rectangles[i], rectangles[j], issmall)) {
                    // 合并相似的矩形
                    std::vector<cv::Point2f> mergedRect;
                    mergedRect = rectangles[i].center.x < rectangles[j].center.x
                                    ? detector.mergeSimilarRects(rectangles[i], rectangles[j])
                                    : detector.mergeSimilarRects(rectangles[j], rectangles[i]); 
                    // 将四边形内容投影为长方形
                    cv::Mat squareImg; 
                    squareImg = issmall ? detector.warpToRectangle(frame, mergedRect, 34, 28)(cv::Rect(7, 0, 20, 28)) 
                                        : detector.warpToRectangle(frame, mergedRect, 58, 28)(cv::Rect(19, 0, 20, 28));
                    // 数字识别
                    NumberClassifier number_classifier("mlp.onnx", "label.txt", 0.5);  
                    std::pair<std::string, double> result = number_classifier.classifyNumber(squareImg, issmall); 
                    if(result.first == "negative"){
                        // imshow("squareImg", squareImg);
                        // cv::waitKey(200);
                        continue; 
                    } 
                    armor.is_small = issmall; 
                    armor.classification = result.first; 
                    armor.probability = result.second;  
                    // PnP解算相机外参
                    armor.ex_mat = pnp_solver.solvePnPWithIPPE(mergedRect, std::string(ROOT) + "/input/2BDFA1701242.yaml", issmall); 
                    armor.frame_id = frame_id; 
                    armor.calculatemergedRect(); 
                    armors[result.first].push_back(armor); 
                    // 绘制矩形在原图上
                    std::vector<cv::Point2f> points = armor.mergedRect;
                    for (int k = 0; k < 4; k++) {
                        cv::line(frame, points[k], points[(k + 1) % 4], cv::Scalar(0, 255, 0), 3); 
                    }
                }
            }
        }
        // 更新跟踪器
        for(auto& armor : armors) {
            for(auto& armor_ : armor.second){
                Draw(frame, armor_); 
            }
            if(armor.second.size() == 1){
                if(trackers.find(armor.first) == trackers.end()) trackers[armor.first] = Tracker(armor.second[0], 1.0 / fps);
                if(isSameArmor(trackers[armor.first], armor.second[0])) trackers[armor.first].update1(armor.second[0]); 
                else trackers[armor.first] = Tracker(armor.second[0], 1.0 / fps); 
            }
            if(armor.second.size() == 2){
                double yaw1 = armor.second[0].calculateYawAngle(); 
                double yaw2 = armor.second[1].calculateYawAngle();
                if(abs_yaw(yaw2 - yaw1) > CV_PI / 12) continue;
                if(yaw1 > yaw2 && yaw1 - yaw2 < CV_PI / 2){
                    auto temp = armor.second[0]; armor.second[0] = armor.second[1]; armor.second[1] = temp;
                }
                if(yaw1 < yaw2 && yaw2 - yaw1 > CV_PI / 2){
                    auto temp = armor.second[0]; armor.second[0] = armor.second[1]; armor.second[1] = temp;
                }
                if(trackers.find(armor.first) == trackers.end()){
                    trackers[armor.first] = Tracker(armor.second[0], 1.0 / fps); 
                    trackers[armor.first].update1(armor.second[1]); 
                }
                if(isSameArmor(trackers[armor.first], armor.second[0])) trackers[armor.first].update2(armor.second[0], armor.second[1]); 
                else{
                    trackers[armor.first] = Tracker(armor.second[0], 1.0 / fps); 
                    trackers[armor.first].update1(armor.second[1]); 
                }
            }
            
        }
        armors.clear(); 
        // 检查并删除过期的 Tracker
        auto it = trackers.begin();
        while (it != trackers.end()) {
            if (it->second.isExpired(frame_id, int64(fps))) {
                it = trackers.erase(it);
            } else {
                it->second.markLost(frame_id);
                ++it;
            }
        }
        // 预测并绘制结果
        for (auto& tracker : trackers) {
            std::cout << tracker.first << "\n"; 
            std::cout << tracker.second.getPosition() << " "; 
            std::cout << tracker.second.getVelocity() << std::endl; 
            cv::Mat prediction = tracker.second.predict(); 
            std::cout << tracker.second.getPosition() << " "; 
            std::cout << tracker.second.getVelocity() << std::endl; 
            std::vector<Armor> armors = calculateArmorPositions(tracker.second); 
            for(auto& armor : armors){
                Draw1(frame, armor); 
            }
            std::string text = "(" + std::to_string(tracker.second.getPosition().x) + 
                        ", " + std::to_string(tracker.second.getPosition().y) + 
                        ", " + std::to_string(tracker.second.getPosition().z) + ")";
            cv::putText(frame, text, cv::Point(0, 20), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2); 
            text = std::to_string(armors[0].calculateYawAngle() / CV_PI * 180); 
            cv::putText(frame, text, cv::Point(0, 50), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
            text = std::to_string(tracker.second.getR().first) + " " + std::to_string(tracker.second.getR().second);
            cv::putText(frame, text, cv::Point(0, 80), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
        }
        cv::putText(frame, std::to_string(frame_id), cv::Point(frame_width - 100, 20), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
        // 将处理后的帧写入视频文件
        video.write(frame); 
    }
    std::cout << (cv::getTickCount() - start) / cv::getTickFrequency() << "\n"; 
    cap.release();
    video.release();
    cv::destroyAllWindows();

    return 0;
}

// 从文件中读取视频
bool readVideo(const std::string& filename, cv::VideoCapture& cap) {
    cap.open(std::string(ROOT) + "/img_input/"+filename);

    // 检查视频是否成功打开
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open or find the video." << std::endl;
        return false;
    }
    return true;
} 
// 绘制矩形在原图上
void Draw(cv::Mat& frame, const Armor& armor) {
    cv::putText(frame, armor.classification, armor.mergedRect[3], cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
    cv::putText(frame, std::to_string(armor.calculateYawAngle() / CV_PI * 180), armor.mergedRect[2], cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
    for (int k = 0; k < 4; k++) {
        cv::line(frame, armor.mergedRect[k], armor.mergedRect[(k + 1) % 4], cv::Scalar(255, 255, 0), 2); 
    }
    std::string text = "(" + std::to_string(armor.ex_mat.at<double>(2, 3)) + 
                        ", " + std::to_string(armor.ex_mat.at<double>(0, 3)) + 
                        ", " + std::to_string(-armor.ex_mat.at<double>(1, 3)) + ")"; 
    cv::putText(frame, text, cv::Point(armor.mergedRect[0].x, armor.mergedRect[0].y - 20), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2); 
}
void Draw1(cv::Mat& frame, const Armor& armor) {
    for (int k = 0; k < 4; k++) {
        cv::line(frame, armor.mergedRect[k], armor.mergedRect[(k + 1) % 4], cv::Scalar(0, 255, 255), 2); 
    }
}