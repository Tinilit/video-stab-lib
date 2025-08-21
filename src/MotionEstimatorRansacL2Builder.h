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
    std::string local_outlier_rejection = "yes";
    std::string thresh_mode = "fixed";
    float thresh = 2.5f;
    int nkps = 3000;

    int subset = 4;                         // мінімальна кількість точок для affine
    float min_inlier_ratio = 0.25f;
    float outlier_ratio = 0.3f;
};

// ====== Інтерфейс ======
class IMotionEstimatorBuilder {
public:
    virtual ~IMotionEstimatorBuilder() {}
    virtual Ptr<ImageMotionEstimatorBase> build() = 0;
protected:
    IMotionEstimatorBuilder() = default;
};

// ====== RansacL2 Builder ======
class MotionEstimatorRansacL2Builder : public IMotionEstimatorBuilder {
public:
    MotionEstimatorRansacL2Builder(const Params& _params, bool use_gpu);
    virtual Ptr<ImageMotionEstimatorBase> build() override;

private:
    bool gpu;
    Params params;
};