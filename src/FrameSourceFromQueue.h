#pragma once

#include <opencv2/videostab/frame_source.hpp>
#include <deque>
#include <mutex>
#include <condition_variable>
#include <iostream>
#include <fstream>
#include <string>
#include "Logger.h"

class FrameSourceFromQueue : public cv::videostab::IFrameSource {
public:
    FrameSourceFromQueue(std::deque<cv::Mat>& queue, std::mutex& mutex, std::condition_variable& cv)
        : frameQueue(queue), queueMutex(mutex), condVar(cv) {}

    cv::Mat nextFrame() override {
        std::unique_lock<std::mutex> lock(queueMutex);
        Logger::logToFile("[nextFrame] Запит кадру. Розмір черги перед чеканням: " + std::to_string(frameQueue.size()));

        condVar.wait(lock, [this]() { 
            bool ready = !frameQueue.empty();
            if (!ready) {
                Logger::logToFile("[nextFrame] Черга порожня, чекаємо...");
            }
            return ready;
        });

        Logger::logToFile("[nextFrame] Прокинулись. Розмір черги: " + std::to_string(frameQueue.size()));

        if (frameQueue.empty()) {
            Logger::logToFile("[nextFrame] Черга порожня після пробудження — повертаю Mat()");
            return cv::Mat();
        }

        cv::Mat frame = frameQueue.front().clone();
        frameQueue.pop_front();

        Logger::logToFile("[nextFrame] Віддав кадр. Розмір черги після pop: " + std::to_string(frameQueue.size()));

        return frame;
    }

    void reset() override {
        
    }

private:
    std::deque<cv::Mat>& frameQueue;
    std::mutex& queueMutex;
    std::condition_variable& condVar;
};