#pragma once

#include <opencv2/videostab/frame_source.hpp>
#include <deque>
#include <mutex>

class FrameSourceFromQueue : public cv::videostab::IFrameSource {
public:
    FrameSourceFromQueue(std::deque<cv::Mat>& queue, std::mutex& mutex)
        : frameQueue(queue), queueMutex(mutex) {}

    cv::Mat FrameSourceFromQueue::nextFrame() {
        std::lock_guard<std::mutex> lock(queueMutex);
        if (frameQueue.empty()) return cv::Mat();
        cv::Mat frame = frameQueue.front().clone();
        frameQueue.pop_front();
        return frame;
    }

    void reset() override {
        
    }

private:
    std::deque<cv::Mat>& frameQueue;
    std::mutex& queueMutex;
};
