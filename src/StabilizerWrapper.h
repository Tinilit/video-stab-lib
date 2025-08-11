#pragma once

#include <opencv2/core.hpp>
#include <opencv2/videostab.hpp>
#include <vector>
#include <mutex>
#include "FrameSourceFromQueue.h"

class StabilizerWrapper {
public:
    StabilizerWrapper();
    ~StabilizerWrapper();

    void feedFrame(const cv::Mat& frame);
    bool getFrame(int index, cv::Mat& out);

private:
    std::deque<cv::Mat> originalFrames;
    std::thread processingThread;
    std::atomic<bool> stopFlag = false;
    cv::Mat latestStabilizedFrame;
    std::mutex mutex;
    std::condition_variable condVar;

    cv::Ptr<cv::videostab::OnePassStabilizer> stabilizer;
    cv::Ptr<cv::videostab::IFrameSource> frameSource;
};