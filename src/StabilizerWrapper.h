#pragma once

#include <opencv2/core.hpp>
#include <opencv2/videostab.hpp>
#include <vector>

class StabilizerWrapper {
public:
    StabilizerWrapper();
    ~StabilizerWrapper();

    void feedFrame(const cv::Mat& frame);
    void process();
    bool getFrame(int index, cv::Mat& out);

private:
    std::vector<cv::Mat> originalFrames;
    std::vector<cv::Mat> stabilizedFrames;
    bool processed;
};