#include "StabilizerWrapper.h"

using namespace cv;
using namespace cv::videostab;

StabilizerWrapper::StabilizerWrapper() : processed(false) {}

StabilizerWrapper::~StabilizerWrapper() {}

void StabilizerWrapper::feedFrame(const Mat& frame) {
    originalFrames.push_back(frame.clone());
}

void StabilizerWrapper::process() {
    if (processed || originalFrames.empty()) return;

    std::vector<Mat> motions;
    std::vector<Mat> stabilized;

    Ptr<IFrameSource> source = makePtr<NullFrameSource>();

    OnePassStabilizer stabilizer;
    stabilizer.setMotionEstimator(makePtr<MotionEstimatorRansacL2>(MM_TRANSLATION_L2));
    stabilizer.setFrameSource(makePtr<VectorFrameSource>(originalFrames));
    stabilizer.setRadius(15);
    stabilizer.setTrimRatio(0.1);
    stabilizer.setBorderMode(BORDER_REPLICATE);

    for (size_t i = 0; i < originalFrames.size(); ++i) {
        Mat stabFrame = stabilizer.nextFrame();
        if (stabFrame.empty()) break;
        stabilizedFrames.push_back(stabFrame.clone());
    }

    processed = true;
}

bool StabilizerWrapper::getFrame(int index, Mat& out) {
    if (!processed || index < 0 || index >= stabilizedFrames.size())
        return false;

    out = stabilizedFrames[index].clone();
    return true;
}