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
    if (!ptr || !outBuffer) return false;

    cv::Mat frame;
    if (!static_cast<StabilizerWrapper*>(ptr)->getFrame(index, frame))
        return false;

    cv::Mat rgba;
    cv::cvtColor(frame, rgba, cv::COLOR_BGR2RGBA);

    if (rgba.cols != width || rgba.rows != height)
        return false;

    for (int y = 0; y < height; ++y) {
        std::memcpy(outBuffer + y * stride, rgba.ptr(y), width * 4);
    }

    return true;
}

}
