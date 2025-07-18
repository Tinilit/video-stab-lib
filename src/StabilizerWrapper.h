#pragma once

#include <opencv2/core.hpp>
#include <opencv2/videostab.hpp>
#include <vector>
#include <mutex>

class StabilizerWrapper {
public:
    StabilizerWrapper();
    ~StabilizerWrapper();

    void feedFrame(const cv::Mat& frame);
    bool getFrame(int index, cv::Mat& out);

private:
    bool processFrame();

    std::deque<cv::Mat> originalFrames;
    std::vector<cv::Mat> stabilizedFrames;
    std::mutex mutex;
    bool processed;
    
    // Відповідні поля в клас StabilizerWrapper (private):
    cv::Mat prevGray;  // попередній сірий кадр
    bool firstFrameProcessed = false;

    std::vector<double> dx, dy, da;            // трансформації між кадрами
    std::vector<double> trajectoryX, trajectoryY, trajectoryA;  // накопичені траєкторії
    std::vector<double> smoothedX, smoothedY, smoothedA;        // згладжені траєкторії

    const int SMOOTH_RADIUS = 5;

    int lastProcessedIndex = -1;  // Індекс останнього стабілізованого кадру

};