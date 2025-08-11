#include "MotionEstimationBuilder.h"

#include <stdexcept>
#include "opencv2/video.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/videostab.hpp"

using namespace std;
using namespace cv;
using namespace cv::videostab;

// ===== Реалізація motionModel() =====
MotionModel motionModel(const string &str)
{
    if (str == "transl") return MM_TRANSLATION;
    if (str == "transl_and_scale") return MM_TRANSLATION_AND_SCALE;
    if (str == "rigid") return MM_RIGID;
    if (str == "similarity") return MM_SIMILARITY;
    if (str == "affine") return MM_AFFINE;
    if (str == "homography") return MM_HOMOGRAPHY;
    throw runtime_error("unknown motion model: " + str);
}

// ===== Реалізація MotionEstimatorL1Builder =====
MotionEstimatorL1Builder::MotionEstimatorL1Builder(const Params& _params, bool use_gpu)
    : gpu(use_gpu), params(_params) {}

Ptr<ImageMotionEstimatorBase> MotionEstimatorL1Builder::build()
{
    Ptr<MotionEstimatorL1> est = makePtr<MotionEstimatorL1>(motionModel(params.model));

    Ptr<IOutlierRejector> outlierRejector = makePtr<NullOutlierRejector>();
    if (params.local_outlier_rejection == "yes")
    {
        Ptr<TranslationBasedLocalOutlierRejector> tblor = makePtr<TranslationBasedLocalOutlierRejector>();
        RansacParams ransacParams = tblor->ransacParams();
        if (params.thresh_mode != "auto")
            ransacParams.thresh = params.thresh;
        tblor->setRansacParams(ransacParams);
        outlierRejector = tblor;
    }

#if defined(HAVE_OPENCV_CUDAIMGPROC) && defined(HAVE_OPENCV_CUDAOPTFLOW)
    if (gpu)
    {
        Ptr<KeypointBasedMotionEstimatorGpu> kbest = makePtr<KeypointBasedMotionEstimatorGpu>(est);
        kbest->setOutlierRejector(outlierRejector);
        return kbest;
    }
#else
    CV_Assert(gpu == false && "CUDA modules are not available");
#endif

    Ptr<KeypointBasedMotionEstimator> kbest = makePtr<KeypointBasedMotionEstimator>(est);
    kbest->setDetector(GFTTDetector::create(params.nkps));
    kbest->setOutlierRejector(outlierRejector);
    return kbest;
}
