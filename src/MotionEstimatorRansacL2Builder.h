#pragma once

#include <string>
#include <opencv2/videostab.hpp>
#include <opencv2/core/utility.hpp>
#include "MotionModelHelper.h"

using namespace cv;
using namespace cv::videostab;

// ====== Структура параметрів ======
struct Params {
    std::string model = "affine";
    std::string local_outlier_rejection = "yes";
    std::string thresh_mode = "fixed";

    float thresh = 3.5f;        // було 1.5f → зроби 3.0–5.0 (більше толерантності до зсувів/ролу)
    int   nkps   = 5000;        // якщо тягне CPU/GPU — підніми з 3000 до 5000–8000

    int   subset = 10;           // було 3 → краще 4–5 для стійкішого affine
    float min_inlier_ratio = 0.15f; // було 0.4 → зменш, щоб RANSAC не душив складні кадри
    float outlier_ratio    = 0.3f;  // можна лишити
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