#include "StabilizerWrapper.h"
#include <cstring>

extern "C" {

__declspec(dllexport) void* CreateStabilizer() {
    return new StabilizerWrapper();
}

__declspec(dllexport) void FreeStabilizer(void* ptr) {
    delete static_cast<StabilizerWrapper*>(ptr);
}

__declspec(dllexport) void FeedFrame(void* ptr, unsigned char* data, int width, int height, int stride) {
    if (!ptr) return;
    cv::Mat input(height, width, CV_8UC4, data, stride);
    static_cast<StabilizerWrapper*>(ptr)->feedFrame(input);
}

__declspec(dllexport) bool GetFrame(void* ptr, int index, unsigned char* outBuffer, int width, int height, int stride) {
    if (!ptr || !outBuffer) {
        Logger::logToFile("GetFrame: invalid ptr or outBuffer is null");
        return false;
    }

    Logger::logToFile("GetFrame: called with index = " + std::to_string(index) +
                      ", width = " + std::to_string(width) +
                      ", height = " + std::to_string(height) +
                      ", stride = " + std::to_string(stride));

    cv::Mat frame;
    if (!static_cast<StabilizerWrapper*>(ptr)->getFrame(index, frame)) {
        Logger::logToFile("GetFrame: getFrame failed for index = " + std::to_string(index));
        return false;
    }

    Logger::logToFile("GetFrame: got frame, size = " + std::to_string(frame.cols) +
                      "x" + std::to_string(frame.rows) +
                      ", channels = " + std::to_string(frame.channels()));

    cv::Mat rgba;
    cv::cvtColor(frame, rgba, cv::COLOR_BGR2RGBA);

    if (rgba.cols != width || rgba.rows != height) {
        Logger::logToFile("GetFrame: size mismatch. Expected " +
                          std::to_string(width) + "x" + std::to_string(height) +
                          ", got " + std::to_string(rgba.cols) + "x" + std::to_string(rgba.rows));
        return false;
    }

    for (int y = 0; y < height; ++y) {
        std::memcpy(outBuffer + y * stride, rgba.ptr(y), width * 4);
    }

    Logger::logToFile("GetFrame: frame copied successfully to outBuffer");
    return true;
}

}
