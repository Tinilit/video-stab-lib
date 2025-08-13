#pragma once

#include <opencv2/videostab/frame_source.hpp>
#include <deque>
#include <mutex>
#include <condition_variable>

class FrameSourceFromQueue : public cv::videostab::IFrameSource {
public:
    FrameSourceFromQueue(std::deque<cv::Mat>& queue, std::mutex& mutex, std::condition_variable& cv)
        : frameQueue(queue), queueMutex(mutex), condVar(cv) {}

    cv::Mat FrameSourceFromQueue::nextFrame() {
        std::unique_lock<std::mutex> lock(queueMutex);

        condVar.wait(lock, [this]() { 
            return !frameQueue.empty(); 
        });

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
    std::condition_variable& condVar;
};