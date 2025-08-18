#include "MotionEstimatorRansacL2Builder.h"

#include <stdexcept>
#include "opencv2/video.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/videostab.hpp"

using namespace std;
using namespace cv;
using namespace cv::videostab;

// ===== Реалізація MotionEstimatorRansacL2Builder =====
MotionEstimatorRansacL2Builder::MotionEstimatorRansacL2Builder(const Params& _params, bool use_gpu)
    : gpu(use_gpu), params(_params) {}

Ptr<ImageMotionEstimatorBase> MotionEstimatorRansacL2Builder::build()
{
    // Створюємо RansacL2 motion estimator
    Ptr<MotionEstimatorRansacL2> est = makePtr<MotionEstimatorRansacL2>(motionModel(params.model));

    // Outlier rejector
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