#pragma once
#include <string>
#include <stdexcept>
#include <opencv2/videostab.hpp>

using namespace cv::videostab;

// Лише оголошення функції
MotionModel motionModel(const std::string &str);