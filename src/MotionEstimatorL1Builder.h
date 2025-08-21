#pragma once

#include <string>
#include <opencv2/videostab.hpp>
#include <opencv2/core/utility.hpp>
#include "MotionModelHelper.h"

using namespace cv;
using namespace cv::videostab;

// ====== Структура параметрів ======
struct Params {
    std::string model = "affine";           // змінили з rigid на affine
    std::string local_outlier_rejection = "no";
    std::string thresh_mode = "auto";
    float thresh = 3.0f;
    int nkps = 3000;

    int subset = 3;                         // мінімальна кількість точок для affine
    float min_inlier_ratio = 0.2f;
    float outlier_ratio = 0.5f;
};

// ====== Інтерфейс ======
class IMotionEstimatorBuilder {
public:
    virtual ~IMotionEstimatorBuilder() {}
    virtual Ptr<ImageMotionEstimatorBase> build() = 0;
protected:
    IMotionEstimatorBuilder() = default;
};

// ====== L1 Builder ======
class MotionEstimatorL1Builder : public IMotionEstimatorBuilder {
public:
    MotionEstimatorL1Builder(const Params& _params, bool use_gpu);
    virtual Ptr<ImageMotionEstimatorBase> build() override;

private:
    bool gpu;
    Params params;
};