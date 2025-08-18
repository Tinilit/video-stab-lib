#pragma once

#include <string>
#include <opencv2/videostab.hpp>
#include <opencv2/core/utility.hpp>
#include "MotionModelHelper.h"

using namespace cv;
using namespace cv::videostab;

// ====== Структура параметрів ======
struct Params {
    std::string model = "rigid";
    std::string local_outlier_rejection = "no";
    std::string thresh_mode = "auto"; // або "fixed"
    float thresh = 1.0f;
    int nkps = 1000;

    // нові поля для RansacL2
    int subset = 20;                // кількість точок для оцінки моделі
    float min_inlier_ratio = 0.5f;  // мінімальний відсоток інлайєрів
    float outlier_ratio = 0.5f;     // очікуваний відсоток аутлайєрів
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