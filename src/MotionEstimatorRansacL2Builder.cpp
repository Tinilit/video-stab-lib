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
    Ptr<MotionEstimatorRansacL2> est = makePtr<MotionEstimatorRansacL2>(motionModel(params.model));

    RansacParams ransac = est->ransacParams();
    ransac.size = params.subset;
    ransac.thresh = params.thresh;
    ransac.eps = params.outlier_ratio;
    est->setRansacParams(ransac);
    est->setMinInlierRatio(params.min_inlier_ratio);

#if defined(HAVE_OPENCV_CUDAIMGPROC) && defined(HAVE_OPENCV_CUDAOPTFLOW)
    if (gpu)
    {
        Ptr<KeypointBasedMotionEstimatorGpu> kbest = makePtr<KeypointBasedMotionEstimatorGpu>(est);
        return kbest;
    }
#else
    CV_Assert(gpu == false && "CUDA modules are not available");
#endif

    Ptr<KeypointBasedMotionEstimator> kbest = makePtr<KeypointBasedMotionEstimator>(est);
    kbest->setDetector(GFTTDetector::create(params.nkps));
    return kbest;
}